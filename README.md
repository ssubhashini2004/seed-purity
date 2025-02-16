# Seed Purity Analysis Using CNN

## Overview
This project implements an **automated seed purity analysis system** using **Convolutional Neural Networks (CNNs)**. The system classifies seed images into four categories: **Pure, Broken, Discolored, and Silkcut**. By leveraging deep learning, this project aims to replace traditional manual inspection methods with a more accurate, scalable, and efficient approach.

The model is trained on a labeled dataset of seed images and optimized using **Google Colab with TPU acceleration** to improve computational efficiency.

## Features
- **Deep Learning-Based Classification:** Uses a CNN model for accurate seed purity classification.
- **Automated Image Processing:** Preprocessing techniques such as resizing, normalization, and data augmentation improve model performance.
- **TPU Acceleration:** Leverages TPU v2 for efficient training.
- **High Accuracy:** Optimized using techniques like dropout and data augmentation.
- **Scalable Solution:** Can be integrated into agricultural quality control systems.

## Methodology
1. **Dataset Preparation:**
   - Images of seeds were collected and labeled as **Pure, Broken, Discolored, or Silkcut**.
   - Data augmentation was applied to increase dataset diversity.
   
2. **Model Architecture:**
   - A CNN model was designed with multiple **convolutional, pooling, and fully connected layers**.
   - Used **ReLU activation**, **softmax classifier**, and **dropout regularization**.
   
3. **Training and Optimization:**
   - Model trained using **Google Colab TPU v2**.
   - Categorical cross-entropy loss and Adam optimizer used.
   - Hyperparameters fine-tuned to improve accuracy.
   
4. **Evaluation:**
   - Performance tested on validation dataset.
   - Accuracy, loss, and confusion matrix analyzed.

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- TensorFlow & Keras
- OpenCV (for image processing)
- NumPy, Matplotlib, Pandas
- Google Colab (for TPU support)
## Acknowledgments and Dataset Used
- Data sourced from https://www.kaggle.com/datasets/linxvandu/corn-seed-image-classification. Special thanks to Kaggle for providing access to the training data.
### Clone the Repository
```bash
git clone https://github.com/ssubhashini2004/seed-purity.git
cd seed-purity-analysis
