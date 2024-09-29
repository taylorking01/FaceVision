# FaceVision TODO List

## Current Model Limitations

- **Face Detection Bias**: The model has a bias towards detecting faces facing right and struggles with faces looking straight or left.
- **Generalization**: The model's generalization is limited due to insufficient variation in face orientations, lighting conditions, and background diversity.

## Completed Improvements

- Implemented data augmentation techniques (rotating, flipping, scaling, brightness/contrast adjustments) to enhance model generalization.

## Planned Improvements

### 1. Additional Training Data
- Train the model on a more diverse dataset with varied face orientations, lighting, and backgrounds.

### 2. Data Augmentation
- Continue exploring data augmentation techniques, including advanced transformations such as:
  - Perspective transformations
  - Gaussian noise addition

### 3. Use More Complex Classifiers
- Experiment with more complex models such as:
  - Support Vector Machines (SVM)
  - Convolutional Neural Networks (CNNs)

### 4. Hyperparameter Tuning
- Perform Grid Search or Random Search to optimize the following parameters:
  - Number of estimators in AdaBoost
  - Learning rate
  - Depth of decision trees

### 5. Optimize Training with Multithreading
- Implement multithreading and simultaneous processing to take advantage of available memory and computational power.
- Investigate how to optimize memory usage during training to reduce training time without maxing out system resources.
---

## Long-Term Goals

### 1. Implement Deep Learning Models
- Transition from traditional machine learning techniques to deep learning (e.g., CNNs) for better face detection performance.

### 2. Integration with Real-World Applications
- Explore integration with real-time applications such as security systems, face recognition, etc.

