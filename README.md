# Brain Tumor Classification Using CNN

A deep learning project focused on classifying brain tumors from MRI scans using a custom Convolutional Neural Network (CNN). This project explores the impact of different data augmentation strategies on model performance and includes visual evaluations of predictions.

---

## Table of Contents

- [Brain Tumor Classification Using CNN](#brain-tumor-classification-using-cnn)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Experiments and Results](#experiments-and-results)
  - [Visualizing Predictions](#visualizing-predictions)
    - [Summary of Batch-Wise Predictions](#summary-of-batch-wise-predictions)
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

---

## Experiments and Results

| Experiment               | Epochs | Augmentation       | Train Accuracy | Test Accuracy |
|--------------------------|--------|--------------------|----------------|---------------|
| No Augmentation          | 5      | None               | 98.22%         | 71.07%        |
| Full Augmentation        | 5      | Flip + Rotation    | 91.99%         | 64.21%        |
| Simplified Augmentation  | 10     | Flip only          | 99.23%         | **72.59%**    |

**Observation**: Full augmentation reduced performance, likely due to over-distortion. Simplified augmentation (horizontal flip only) with more epochs yielded the best result.

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
│   └── test_cnn.ipynb                       # Running and testing trained CNN
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
├── requirements.txt                 # Dependencies for environment setup
├── setup_project.py                 # Optional setup file (e.g., for initializing folders)
├── .gitignore                       # Files and folders to exclude from GitHub
└── README.md                        # Project summary and results
```


---

## Future Work

- Replace CNN with transfer learning using ResNet18 or other pretrained models
- Hyperparameter tuning for better generalization
- Use Grad-CAM or saliency maps to interpret predictions
- Deploy the trained model via a web interface

---

## License

This project is for educational purposes and is licensed under the MIT License.
