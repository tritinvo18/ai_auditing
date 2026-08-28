import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tqdm
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Normalize images before passing to ResNet using standard ImageNet preprocessing means and standard deviations
imagenet_means = (0.485, 0.456, 0.406)
imagenet_stds = (0.229, 0.224, 0.225)

# Fix the seed for reproducibility and determines the execution device (GPU or CPU)
def setup_env():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

# Loads internal and external datasets and prepare the dataloader
def get_dataloaders(internal_path, external_path, batch_size=16, num_workers=1):
    # Load raw image datasets using ImageFolder (infers classes from subdirectories)
    test_internal_dataset = torchvision.datasets.ImageFolder(internal_path)
    test_external_dataset = torchvision.datasets.ImageFolder(external_path)
    
    # Define preprocessing pipeline: convert to tensor, resize to 224x224, and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        transforms.Normalize(imagenet_means, imagenet_stds)
    ])
    
    # Apply transformations to the datasets
    test_internal_dataset.transform = transform
    test_external_dataset.transform = transform
    
    # Wrap datasets in DataLoaders for batching during evaluation
    test_internal_loader = torch.utils.data.DataLoader(test_internal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_external_loader = torch.utils.data.DataLoader(test_external_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_internal_loader, test_external_loader, test_internal_dataset, test_external_dataset

# Initializes a ResNet18 model and updates the final layer for the specified number of classes.
def setup_model(num_classes, weights_path=None, device='cpu'):
    # Load the base ResNet18 architecture
    weights = None if weights_path else torchvision.models.ResNet18_Weights.DEFAULT
    backbone = torchvision.models.resnet18(weights=weights)
    
    # Replace the final fully connected layer to match the required number of classes
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    
    # Load pre-trained weights if provided
    if weights_path:
        backbone.load_state_dict(torch.load(weights_path, map_location=device))
        
    # Set model to evaluation mode (disables dropout, fixes batch normalization)
    backbone.eval()
    backbone = backbone.to(device)
    
    return backbone

# Runs inference on the provided dataloaders and collects all probabilities and ground truth labels
def evaluate_model(model, internal_loader, external_loader, device):
    all_predictions = {'internal': [], 'external': []}
    all_labels = {'internal': [], 'external': []}

    # Disable gradient tracking to save memory and speed up inference
    with torch.no_grad():
        for split, dataloader in zip(['internal', 'external'], [internal_loader, external_loader]):
            for data in tqdm.tqdm(dataloader, desc=f'Evaluating on {split} set'):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Convert raw logits to probabilities using Softmax
                probs = torch.softmax(outputs, dim=1)
                
                # Store results on CPU
                all_predictions[split].append(probs.cpu().numpy())
                all_labels[split].append(labels.cpu().numpy())
                
    # Concatenate batched lists into single numpy arrays
    for split in ['internal', 'external']:
        all_predictions[split] = np.concatenate(all_predictions[split], axis=0)
        all_labels[split] = np.concatenate(all_labels[split], axis=0)
        
    return all_predictions, all_labels

# Calculates True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
def confusion_matrix_counts(labels, probs, threshold=0.5):
    labels = labels.astype(bool)
    probs = probs.astype(float)
    
    # Binarize predictions based on the decision threshold
    preds = probs >= threshold
    
    tp = np.sum(preds & labels)
    fp = np.sum(preds & ~labels)
    tn = np.sum(~preds & ~labels)
    fn = np.sum(~preds & labels)
    
    return tp, fp, tn, fn

# Computes Precision, Recall, and F1-score
def compute_metrics(labels, probs, threshold=0.5):
    tp, fp, tn, fn = confusion_matrix_counts(labels, probs, threshold=threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Plot and save ROC and Precision-Recall (PR) curves
def plot_roc_pr_curves(labels, probs, split_name, save_prefix=''):
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR metrics
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    # Set up a 1x2 grid for the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal representing random guessing
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{split_name} - ROC Curve')
    axes[0].legend(loc="lower right")
    
    # Plot PR Curve
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{split_name} - PR Curve')
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}{split_name}_roc_pr.png')
    plt.show()

# Calculates the Expected Calibration Error (ECE)
def calculate_ece(labels, probs, n_bins=10):
    # Create bin boundaries (e.g., [0, 0.1, 0.2, ... 1.0])
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Identify which predictions fall into the current bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        # Only evaluate bins that contain predictions
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            
            # ECE is the weighted average of the absolute difference between confidence and accuracy
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

# Compute basic statistical metrics (brightness and contrast) for an entire dataset
def get_image_stats(dataset):
    brightness = []
    contrast = []
    
    for img, _ in dataset:
        # Reverse the ImageNet normalization to get pixels back to [0, 1] range
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(imagenet_stds) + np.array(imagenet_means)
        img_np = np.clip(img_np, 0, 1)
        
        # Convert RGB to Grayscale using standard luminosity method
        gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Brightness is the mean pixel intensity, contrast is the standard deviation
        brightness.append(np.mean(gray))
        contrast.append(np.std(gray))
        
    return brightness, contrast

# Calculates and plots histograms comparing the brightness and contrast distributions between the internal and external datasets
def plot_image_stats(internal_dataset, external_dataset, save_prefix=''):
    int_b, int_c = get_image_stats(internal_dataset)
    ext_b, ext_c = get_image_stats(external_dataset)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Brightness histogram
    axes[0].hist(int_b, bins=30, alpha=0.5, label='Internal')
    axes[0].hist(ext_b, bins=30, alpha=0.5, label='External')
    axes[0].set_title('Brightness Distribution')
    axes[0].legend()
    
    # Contrast histogram
    axes[1].hist(int_c, bins=30, alpha=0.5, label='Internal')
    axes[1].hist(ext_c, bins=30, alpha=0.5, label='External')
    axes[1].set_title('Contrast Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}dataset_stats.png')
    plt.show()

# Reverses the ImageNet normalization applied during preprocessing.
def denormalize_image(image):
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array(imagenet_stds) + np.array(imagenet_means)
    return np.clip(image, 0, 1)

# Computes the Class Activation Map (CAM) given convolutional feature maps and fully connected layer weights
def compute_cam(conv_feature_map, fc_weights, image_size=(224, 224)):
    # Calculate weighted combination of convolutional feature maps
    cam = torch.sum(fc_weights[:, None, None] * conv_feature_map, dim=0)
    
    # Apply ReLU / clamp negative activations to zero
    cam = torch.clamp(cam, min=0)
    
    # Normalize CAM values to [0, 1] range
    if torch.max(cam) > 0:
        cam = cam / torch.max(cam)
        
    # Resize CAM to the target image dimensions using bilinear interpolation
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    return cam

