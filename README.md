# **FaceVision**

## **Introduction**

FaceVision is a computer vision project aimed at developing a face detection system using machine learning techniques. The project leverages the Labeled Faces in the Wild (LFW) dataset to train a model that distinguishes between human faces and non-human images. The goal is to explore the implementation of a face detection algorithm, evaluate its performance on unseen data, and identify areas for improvement.

---

## **Project Structure**

The project is organized into several Python scripts, each responsible for a specific functionality:

- **`data_loader.py`**: Loads and preprocesses the dataset, including both face images and non-face images.
- **`feature_extractor.py`**: Extracts features from images to be used by the classifier.
- **`classifier.py`**: Contains functions to train the machine learning classifier.
- **`train.py`**: Orchestrates the training process, including data loading, feature extraction, and model training.
- **`main.py`**: Opens the webcam to run real-time face detection using the trained model.
- **`evaluate_model.py`**: Loads the trained model and evaluates its performance on unseen test data.
- **`utils.py`**: Provides utility functions, such as plotting sample images.

---

## **Mathematical Background**

### **Feature Extraction**

The feature extraction process involves transforming images into a format suitable for machine learning algorithms. Initially, images are processed using Histogram of Oriented Gradients (HOG), which captures essential features such as edges and gradients for face detection:

\[
	ext{HOG Features} = 	ext{compute\_hog}(	ext{Image})
\]

### **Classification Algorithm**

The classifier used is an AdaBoost ensemble method with decision stumps (trees with a maximum depth of 1) as weak learners. The AdaBoost algorithm combines multiple weak learners to form a strong classifier.

- **AdaBoost Algorithm**:

  For \( m = 1 \) to \( M \):
  
  1. Train a weak learner \( h_m(x) \) using weighted training data.
  2. Compute the weak learner's weight \( lpha_m \):
     \[
     lpha_m = rac{1}{2} \ln\left(rac{1 - \epsilon_m}{\epsilon_m}
ight)
     \]
     where \( \epsilon_m \) is the weighted error rate.
  3. Update the data weights \( w_i \) for each training sample:
     \[
     w_i \leftarrow w_i \exp\left(-lpha_m y_i h_m(x_i)
ight)
     \]
  4. Normalize the weights.

- **Final Strong Classifier**:
  \[
  H(x) = 	ext{sign}\left(\sum_{m=1}^{M} lpha_m h_m(x)
ight)
  \]

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
2. **Feature Extraction**: Histogram of Oriented Gradients (HOG) features are extracted from images.
3. **Data Splitting**: The dataset is split into training and testing sets using an 80/20 split.
4. **Model Training**: An AdaBoost classifier with decision stumps is trained on the training set.
5. **Model Saving**: The trained model is saved to `face_detection_model.joblib` for later use.

---

## **Performance Evaluation**

### Initial Model:
- **Accuracy**: 50.00% (indicating poor performance on the unseen data).

### Improved Model:
- **Accuracy**: 94.00% (after implementing HOG feature extraction and using a more diverse dataset).

---

## **Current Limitations**

- The model shows bias toward detecting faces facing a specific direction (e.g., faces facing right are more likely to be detected, while faces looking straight or left are often missed).
- More diverse training data with varying face orientations, lighting, and background conditions are needed to improve generalization.

---

## **Future Improvements**

### **1. Additional Datasets**
- Train the model on a larger dataset that includes more face orientations (left, right, straight), varied lighting conditions, and different backgrounds.

### **2. Use More Complex Classifiers**
- Consider switching to classifiers that can capture non-linear relationships:
  - **Support Vector Machines (SVM)**
  - **Random Forests**
  - **Convolutional Neural Networks (CNNs)**

### **3. Data Augmentation**
- Augment the training dataset by rotating, flipping, and scaling the images to create more variation.

### **4. Hyperparameter Tuning**
- Fine-tune the model’s parameters to optimize performance.

---

## **Conclusion**

The FaceVision project serves as a foundation for understanding face detection using machine learning. By enhancing feature extraction methods, utilizing more sophisticated classifiers, and expanding the dataset, future iterations of the project can achieve better performance.

---

## **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/your_username/FaceVision.git
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
