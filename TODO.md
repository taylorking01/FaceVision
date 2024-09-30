# FaceVision TODO List

## Current Model Limitations

- **Potential Overfitting**: The SVM model achieved 100% accuracy on the current unseen test data, which suggests that the model might be overfitting. More unseen data should be tested.

## Completed Improvements

- Implemented SVM classifier for face detection.
- Achieved 100% accuracy on unseen test data.

## Planned Improvements

### 1. Additional Testing with Unseen Data
- Test the SVM model on a much larger unseen dataset to ensure that the model is not overfitting.

### 2. Fine-Tune SVM Parameters
- Adjust the C (regularization) and gamma parameters to explore better generalization capabilities.

### 3. Data Augmentation
- Continue exploring data augmentation techniques, including advanced transformations such as perspective shifts and Gaussian noise.

### 4. Explore Deep Learning Models
- Consider experimenting with CNNs for improved feature extraction and face detection performance.

### 5. Optimize Training with Multithreading
- Investigate further multithreading optimizations to speed up data loading and training phases.

---

## Long-Term Goals

### 1. Implement Deep Learning Models
- Transition from traditional machine learning techniques to deep learning (e.g., CNNs) for better face detection performance.

### 2. Integration with Real-World Applications
- Explore integration with real-time applications such as security systems, face recognition, etc.

---
