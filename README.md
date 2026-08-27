# ai_auditing

In this assesment, you are acting as a Lead AI Auditor for your firm. You have inherited three pre-trained image classification models based on the CelebA dataset.Links to an external site. While previous engineering teams reported high accuracy, field reports suggest these models are failing in real-world deployment. Your task is to identify the root cause of these failures and propose actionable recommendations to solve the identified problem.

For each case, you will have access to:

    - **Trained model**: the developed model architecture code and trained weights. A start-up notebook showing how to load the model and datasets will also be provided.

    - **Internal test set**: an internal validation set used to evaluate the model during development. The dataset used to train the model, which is not provided for auditing, has similar data distribution.

    - **External test set**: A sample of the field dataset data where the model has exhibited failure.

Your task is to inspect the datasets, evaluate the model to identify possible issues, and then report your findings and suggestions. You should try to compute metrics, plots, saliency checks, and so on. You’ll also submit a .ipynb notebook containing the plots and analysis you have run.

Note that you are **not required to fix or retrain the models**. Your task is to act as an AI auditor: identify the likely root cause of the model's failure, provide evidence to support your diagnosis, and recommend appropriate measures to address it.

## Case details

Work through the following three cases to complete this challenge and produce your report and implementation notebooks:

### Case 1

For a high-end grooming application, the engineering team developed a model to discern clean-shaven faces. During the 'Lab Phase,' the model performed exceptionally well, trained and validated on studio-quality images. It achieved a near-perfect accuracy on the Internal test Set. However, once deployed in a mobile application, the model’s accuracy significantly decreased. Users have reported that the model frequently fails. What happened during the transition from the 'Studio' to the 'Mobile App'?

Supplement files for this challenge:

    - Case1Dataset.zip

    - resnet_frozen_best_case1.pth

    - case1.ipynb

### Case 2

For an automated retail kiosk that recommends a new look based on a user’s current appearance, the development team developed a classifier to detect whether an individual is wearing eyeglasses, enabling the system to suggest suitable frame styles as part of the recommendation. During internal 'Lab Phase' testing, the model achieved near-perfect accuracy on the company’s validation set. Engineers also observed that its performance remained stable across different lighting conditions and head poses. However, within weeks of deployment, the results from the kiosks were puzzling. While the model worked reliably for some users, its accuracy dropped significantly for others. Even more confusingly, these errors occurred in clear, well-lit images similar to those used during internal testing.  

Download the following files for this challenge:

    - Case2Dataset.zip

    - resnet_frozen_best_case2.pth

    - case2.ipynb

### Case 3

For an automated social media tagging system designed to predict whether a person appears young in facial images, the developers implemented a model that assigns a probability score to each prediction. To streamline the workflow, a 'High-Confidence Bypass' was added: predictions with a probability of 90% or higher were automatically published without human review. During the controlled 'Lab Phase,' this approach performed impressively, producing accurate tags with minimal intervention. Since deployment, however, the system has drawn criticism. In practice, it occasionally publishes incorrect tags while still reporting near-certain confidence, leading to unexpected errors despite the apparent reliability suggested by its high-probability outputs.

Download the following files for this challenge:

    - Case3Dataset.zip

    - resnet_frozen_best_case3.pth

    - case3.ipynb

## Submission Instructions

### Report

You must submit a single professional report in PDF format (max 2 pages) on this page. This document should be written with technical precision and clarity. For each of the three cases, your report must include:

    - **Diagnosis:** A clear identification of the specific technical flaw (e.g., spurious correlation or texture-dependency) responsible for the observed model failure.

    - **Experiments and evidence:** Describe the experiments performed to investigate and validate your diagnosis, including the relevant results. Include key plots, visualisations, or metrics that support your diagnosis (e.g., dataset comparisons, saliency maps, performance metrics, confusion matrices, or reliability diagrams).

    - **Recommendations:** Provide specific and actionable recommendations for the engineering team to address the identified problem. Recommendations should be directly linked to your diagnosis and supported by your experimental evidence.

### Project Code

You will submit a Zipped folder named project4_code.zip. This folder must include:

    - A Jupyter Notebook for each case (three notebooks in total) named case1.ipynb, case2.ipynb, and case3.ipynb. These are the technical appendices to your report. Every plot, table, or metric presented in your PDF report must be directly reproducible by running the code in these notebooks. Please include any auxiliary files required for this process, such as pickle files containing losses, predictions, metrics or other necessary data. Ensure all notebooks are compatible with the IFN680 computing environment. The grader will run this notebook to verify reproducibility. Note that this project does not use development.ipynb or main_report.ipynb like previous projects.
