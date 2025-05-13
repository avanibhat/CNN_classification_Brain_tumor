# Brain Tumor Classification Using CNN

A deep learning project focused on classifying brain tumors from MRI scans using a custom Convolutional Neural Network (CNN). This project explores the impact of different data augmentation strategies on model performance and includes visual evaluations of predictions.

---

## Table of Contents


- [Brain Tumor Classification Using CNN](#brain-tumor-classification-using-cnn)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
    - [ResNet18 (Transfer Learning)](#resnet18-transfer-learning)
  - [Experiments and Results](#experiments-and-results)
    - [ResNet18:](#resnet18)
  - [Visualizing Predictions](#visualizing-predictions)
    - [Summary of Batch-Wise Predictions](#summary-of-batch-wise-predictions)
  - [Grad-CAM Visualization](#grad-cam-visualization)
  - [Confusion Matrix and Classification Report](#confusion-matrix-and-classification-report)
    - [Confusion Matrix (ResNet18)](#confusion-matrix-resnet18)
    - [Classification Report Highlights](#classification-report-highlights)
  - [Folder Structure](#folder-structure)
  - [Future Work](#future-work)
  - [License](#license)

---

## Project Overview

Brain tumors are complex and vary widely in structure and appearance. This project aims to build a custom CNN to classify MRI scans into four types of brain tumors. It explores multiple augmentation strategies to improve generalization and includes visual checks for model performance.

---

## Dataset

- **Source**: [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- The dataset contains pre-split `Training` and `Testing` folders.
- Classes:
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`
  - `no_tumor`

---

## Model Architecture

The CNN model consists of:
- Three convolutional layers with ReLU and MaxPooling
- Dropout layer for regularization
- Two fully connected layers
- Final softmax classification for 4 classes

### ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Final fully connected layer replaced to output 4 classes
- Fine-tuned only the final layers for stability
- Used ImageNet normalization and controlled augmentation


---

## Experiments and Results

| Experiment               | Epochs | Augmentation       | Train Accuracy | Test Accuracy |
|--------------------------|--------|--------------------|----------------|---------------|
| No Augmentation          | 5      | None               | 98.22%         | 71.07%        |
| Full Augmentation        | 5      | Flip + Rotation    | 91.99%         | 64.21%        |
| Simplified Augmentation  | 10     | Flip only          | 99.23%         | **72.59%**    |

**Observation**: Full augmentation reduced performance, likely due to over-distortion. Simplified augmentation (horizontal flip only) with more epochs yielded the best result.

### ResNet18:

| Epochs | Train Accuracy | Test Accuracy | Best Test Accuracy |
|--------|----------------|----------------|---------------------|
| 5      | 97.18%         | 70.05%         | 70.05%              |

**Observation**: ResNet18 converged quickly with high train accuracy, but performance varied based on augmentation intensity and class imbalance.


---

## Visualizing Predictions

To validate model performance, we visualized multiple batches from the test set.

### Summary of Batch-Wise Predictions

- **Glioma Tumor**
  - 0/8 correct in one batch  
  - Frequently confused with `meningioma_tumor` and `no_tumor`

- **Meningioma Tumor**
  - 8/8 correct in two separate batches  
  - High accuracy and confidence

- **Pituitary Tumor**
  - 6/8 correct, 2 predicted as `no_tumor`  
  - Moderate confusion observed

This evaluation suggests class-specific biases — strong performance on meningioma, weaker on glioma.


## Grad-CAM Visualization

Grad-CAM was implemented to understand what regions of the MRI influenced the model's decisions.
- Visualizations showed strong attention on actual tumor regions in meningioma and pituitary cases.
- Glioma predictions often had weak or misplaced focus, explaining poor performance.

---

## Confusion Matrix and Classification Report

### Confusion Matrix (ResNet18)
- **Glioma tumors** were frequently misclassified as meningioma or no tumor.
- **Meningioma and No Tumor** classes were classified accurately.
- **Pituitary tumors** showed moderate confusion with meningiomas.

### Classification Report Highlights

| Class              | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| Glioma            | 0.87      | 0.20   | 0.33     | 100     |
| Meningioma        | 0.57      | 0.99   | 0.73     | 115     |
| No Tumor          | 0.77      | 0.94   | 0.85     | 105     |
| Pituitary Tumor   | 1.00      | 0.58   | 0.74     | 74      |

**Observation**: Despite high accuracy overall (70%), poor glioma recall shows a clear performance gap that needs addressing.

---

## Folder Structure

```bash
Brain-Tumor-Classification/
├── .venv/                            # Python virtual environment
├── data/
│   └── brain_tumor_dataset/
│       ├── Training/                # Training images by class
│       └── Testing/                 # Testing images by class
├── notebooks/
│   ├── load_and_visual_of_batch.ipynb      # Data loading and batch visualization
│   ├── test_cnn.ipynb                      # Running and testing trained CNN
│   └── visualize_gradcam.ipynb
├── outputs/
│   ├── Model_results.ipynb          # Training metrics and accuracy summary
│   └── visualize_predictions.ipynb  # True vs predicted label visualization
├── src/
│   ├── __init__.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── __pycache__/
├── cnn_model.pth                    # Trained model weights
├── best_model.pth                   
├── requirements.txt                 # Dependencies for environment setup
├── setup_project.py                 # Optional setup file (e.g., for initializing folders)
├── .gitignore                       # Files and folders to exclude from GitHub
└── README.md                        # Project summary and results
```


---

## Future Work

1. Improve glioma classification using weighted loss or focal loss
2. Apply SMOTE or other techniques to balance training data
3. Extend training with ResNet34 or EfficientNet
4. Integrate segmentation before classification

---

## License

This project is for educational purposes and is licensed under the MIT License.
