# **FaceVision**

## **Introduction**

FaceVision is a computer vision project aimed at developing a face detection system using machine learning techniques. The project leverages the Labeled Faces in the Wild (LFW) dataset to train a model that distinguishes between human faces and non-human images. The goal is to explore the implementation of a face detection algorithm, evaluate its performance on unseen data, and identify areas for improvement.

---

## **Project Structure**

The project is organized into several Python scripts, each responsible for a specific functionality:

- **`data_loader.py`**: Loads and preprocesses the dataset, including both face images and non-face images.
- **`feature_extractor.py`**: Extracts features from images to be used by the classifier.
- **`classifier.py`**: Contains functions to train the machine learning classifier, now using Support Vector Machines (SVM).
- **`train.py`**: Orchestrates the training process, including data loading, feature extraction, and model training with start and finish times.
- **`main.py`**: Opens the webcam to run real-time face detection using the trained model.
- **`evaluate_model.py`**: Loads the trained model and evaluates its performance on unseen test data.
- **`utils.py`**: Provides utility functions, such as plotting sample images.

---

## **Mathematical Background**

### **Feature Extraction**

The feature extraction process involves transforming images into a format suitable for machine learning algorithms. Currently, Histogram of Oriented Gradients (HOG) is used to capture essential features such as edges and gradients for face detection.

### **Classification Algorithm: SVM**

We have implemented a Support Vector Machine (SVM) using a Radial Basis Function (RBF) kernel.

- **Kernel**: RBF is used as it performs well in most image recognition tasks by creating non-linear decision boundaries.
- **C (Regularization)**: Controls the trade-off between classifying training points correctly and having a smooth decision boundary.
- **Gamma**: Controls the influence of individual data points, currently set to `'scale'`, which is recommended for SVM in scikit-learn.

---

## **Data Used**

### **Training Data**

- **Face Images**: Sourced from the Labeled Faces in the Wild (LFW) dataset, resized to \( 62 	imes 47 \) pixels.
- **Non-Face Images**: Sourced from the CIFAR-100 dataset, providing varied non-face objects such as animals and vehicles.

### **Test Data**

- **Human Faces**: A subset of the LFW dataset not used in training.
- **Non-Human Images**: Selected from the CIFAR-10 dataset, including classes such as 'airplane', 'automobile', 'ship', and 'truck'.

---

## **Model Training**

The training process involves several steps:

1. **Data Loading**: Face and non-face images are loaded and combined into a single dataset.
2. **Data Augmentation**: Data augmentation techniques such as horizontal flipping, rotation, scaling, and adjusting brightness/contrast are applied.
3. **Feature Extraction**: Histogram of Oriented Gradients (HOG) features are extracted from images.
4. **Data Splitting**: The dataset is split into training and testing sets using an 80/20 split.
5. **Model Training**: A Support Vector Machine (SVM) classifier is trained on the training set.
6. **Model Saving**: The trained model is saved to `face_detection_model_svm.joblib`.

---

## **Performance Evaluation**

### SVM Model Results:
- **Accuracy on Unseen Data**: 100.00%
- **Confusion Matrix**:

  \[
  egin{bmatrix}
  50 & 0 \
  0 & 50 \
  \end{bmatrix}
  \]

### **Classification Report**:

```
              precision    recall  f1-score   support

   Non-Human       1.00      1.00      1.00        50
       Human       1.00      1.00      1.00        50

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100
```

---

## **Current Limitations**

- The model achieved 100% accuracy, which could suggest **overfitting**. This means the model may not generalize well to real-world data or larger datasets. To verify this, we should test on a larger unseen dataset and consider tuning regularization parameters (C) in SVM.

---

## **Future Improvements**

### **1. Additional Unseen Data**
- Test the model on a significantly larger dataset of unseen data to check for overfitting.

### **2. Fine-Tune Hyperparameters**
- Experiment with the C and gamma parameters in SVM to see if more generalizable models can be achieved.

### **3. Use More Complex Classifiers**
- Investigate deep learning models such as Convolutional Neural Networks (CNNs) for more complex face detection tasks.

### **4. Explore Data Augmentation**
- Further explore advanced data augmentation techniques such as perspective transformations.

### **5. Multithreading Optimization**
- Continue optimizing training time by experimenting with multithreading for both data loading and model training.

---

## **Conclusion**

The FaceVision project demonstrates a solid foundation for understanding face detection using machine learning. The implementation of SVM has shown excellent results, but further steps are necessary to validate the model’s generalization and ensure robustness.

---

## **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/taylorking01/FaceVision.git
```

### **2. Install the Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Train the Model**

```bash
python train.py
```

### **4. Evaluate the Model**

```bash
python evaluate_model.py
```

---

## **Dependencies**

- **Python 3.x**
- **NumPy**
- **Scikit-learn**
- **Scikit-image**
- **Matplotlib**
- **Joblib**
- **Torchvision**

---
