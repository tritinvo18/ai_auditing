import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score

# Imnagenet preprocessing
imagenet_means = (0.485, 0.456, 0.406)
imagenet_stds = (0.229, 0.224, 0.225)

def setup_env():
    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_dataloaders(internal_path, external_path, batch_size=16, num_workers=1):
    test_internal_dataset = torchvision.datasets.ImageFolder(internal_path)
    test_external_dataset = torchvision.datasets.ImageFolder(external_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        transforms.Normalize(imagenet_means, imagenet_stds)
    ])
    
    test_internal_dataset.transform = transform
    test_external_dataset.transform = transform
    
    test_internal_loader = torch.utils.data.DataLoader(test_internal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_external_loader = torch.utils.data.DataLoader(test_external_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_internal_loader, test_external_loader, test_internal_dataset, test_external_dataset

def setup_model(num_classes, weights_path=None, device='cpu'):
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    if weights_path:
        backbone.load_state_dict(torch.load(weights_path, map_location=device))
    backbone.eval()
    backbone = backbone.to(device)
    return backbone

def evaluate_model(model, internal_loader, external_loader, device):
    all_predictions = {'internal': [], 'external': []}
    all_labels = {'internal': [], 'external': []}

    with torch.no_grad():
        for split, dataloader in zip(['internal', 'external'], [internal_loader, external_loader]):
            for data in tqdm.tqdm(dataloader, desc=f'Evaluating on {split} set'):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_predictions[split].append(probs.cpu().numpy())
                all_labels[split].append(labels.cpu().numpy())
                
    for split in ['internal', 'external']:
        all_predictions[split] = np.concatenate(all_predictions[split], axis=0)
        all_labels[split] = np.concatenate(all_labels[split], axis=0)
    return all_predictions, all_labels

def confusion_matrix_counts(labels, probs, threshold=0.5):
    labels = labels.astype(bool)
    probs = probs.astype(float)
    preds = probs >= threshold
    tp = np.sum(preds & labels)
    fp = np.sum(preds & ~labels)
    tn = np.sum(~preds & ~labels)
    fn = np.sum(~preds & labels)
    return tp, fp, tn, fn

def compute_metrics(labels, probs, threshold=0.5):
    tp, fp, tn, fn = confusion_matrix_counts(labels, probs, threshold=threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def plot_roc_pr_curves(labels, probs, split_name, save_prefix=''):
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{split_name} - ROC Curve')
    axes[0].legend(loc="lower right")
    
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

def calculate_ece(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def get_image_stats(dataset):
    brightness = []
    contrast = []
    for img, _ in dataset:
        # img is normalized, let's reverse to [0,1] for stats
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(imagenet_stds) + np.array(imagenet_means)
        img_np = np.clip(img_np, 0, 1)
        
        gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
        brightness.append(np.mean(gray))
        contrast.append(np.std(gray))
    return brightness, contrast

def plot_image_stats(internal_dataset, external_dataset, save_prefix=''):
    int_b, int_c = get_image_stats(internal_dataset)
    ext_b, ext_c = get_image_stats(external_dataset)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(int_b, bins=30, alpha=0.5, label='Internal')
    axes[0].hist(ext_b, bins=30, alpha=0.5, label='External')
    axes[0].set_title('Brightness Distribution')
    axes[0].legend()
    
    axes[1].hist(int_c, bins=30, alpha=0.5, label='Internal')
    axes[1].hist(ext_c, bins=30, alpha=0.5, label='External')
    axes[1].set_title('Contrast Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}dataset_stats.png')
    plt.show()

def denormalize_image(image):
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array(imagenet_stds) + np.array(imagenet_means)
    return np.clip(image, 0, 1)
