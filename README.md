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
- **`main.py`**: Loads the trained model and evaluates its performance on unseen test data.
- **`utils.py`**: Provides utility functions, such as plotting sample images.

---

## **Mathematical Background**

### **Feature Extraction**

The feature extraction process involves transforming images into a format suitable for machine learning algorithms. Initially, images are flattened into one-dimensional arrays:

\[
	ext{Flattened Image} = 	ext{reshape}(	ext{Image Matrix}, (1, 	ext{height} 	imes 	ext{width}))
\]

This simple approach may not capture the complex features necessary for distinguishing faces from non-faces.

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
- **Non-Face Images**: Generated as random noise images matching the dimensions of the face images.

### **Test Data**

- **Human Faces**: A subset of the LFW dataset not used in training.
- **Non-Human Images**: Selected from the CIFAR-10 dataset, including classes such as 'airplane', 'automobile', 'ship', and 'truck'.

---

## **Model Training**

The training process involves several steps:

1. **Data Loading**: Face and non-face images are loaded and combined into a single dataset.
2. **Feature Extraction**: Images are flattened to create feature vectors.
3. **Data Splitting**: The dataset is split into training and testing sets using an 80/20 split.
4. **Model Training**: An AdaBoost classifier with decision stumps is trained on the training set.
5. **Model Saving**: The trained model is saved to `face_detection_model.joblib` for later use.

---

## **Performance Evaluation**

When evaluated on unseen test data, the model's performance was suboptimal:

- **Accuracy**: 50.00%
- **Classification Report**:

  ```
                precision    recall  f1-score   support

     Non-Human       0.50      1.00      0.67        50
         Human       0.00      0.00      0.00        50

      accuracy                           0.50       100
     macro avg       0.25      0.50      0.33       100
  weighted avg       0.25      0.50      0.33       100
  ```

- **Confusion Matrix**:

  \[
  egin{bmatrix}
  50 & 0 \
  50 & 0 \
  \end{bmatrix}
  \]

The model predicts all samples as 'Non-Human', indicating that it fails to generalize to new human face images.

---

## **Discussion**

The poor performance suggests that the current model and feature extraction method are insufficient for effective face detection. Possible reasons include:

- **Inadequate Feature Representation**: Flattening images may not capture the necessary features for distinguishing faces.
- **Model Complexity**: The AdaBoost classifier with decision stumps may be too simplistic for this task.
- **Data Imbalance**: The training data may not be adequately balanced or diverse.

---

## **Future Improvements**

To enhance the model's performance, the following steps are proposed:

### **1. Advanced Feature Extraction**

Implement Histogram of Oriented Gradients (HOG) to capture edge and shape information:

\[
	ext{HOG Features} = 	ext{compute\_hog}(	ext{Image})
\]

### **2. Use More Complex Classifiers**

Experiment with classifiers that can capture non-linear relationships:

- **Support Vector Machines (SVM)**
- **Random Forests**
- **Gradient Boosting Machines**

### **3. Hyperparameter Tuning**

Optimize model parameters using techniques like Grid Search or Random Search to find the best settings for:

- **Number of Estimators**
- **Learning Rate**
- **Max Depth of Trees**

### **4. Increase Training Data**

- **Data Augmentation**: Rotate, flip, or scale images to create more training samples.
- **Additional Datasets**: Incorporate other face datasets to improve diversity.

### **5. Implement Deep Learning Models**

Consider using Convolutional Neural Networks (CNNs) for feature extraction and classification, which are well-suited for image data.

---

## **Conclusion**

The FaceVision project serves as a foundation for understanding face detection using machine learning. While the initial model did not perform as expected on unseen data, it provides valuable insights into areas requiring improvement. By enhancing feature extraction methods, utilizing more sophisticated classifiers, and expanding the dataset, future iterations of the project can achieve better performance.

---

## **How to Run the Project**

To get started with FaceVision, follow these steps:

### **1. Clone the Repository**

Start by cloning the FaceVision repository from GitHub to your local machine or cloud environment (such as Google Colab):

```bash
git clone https://github.com/your_username/FaceVision.git
```

### **2. Navigate to the Project Directory**

Once cloned, navigate into the project directory:

```bash
cd FaceVision
```

### **3. Install the Dependencies**

Before running the project, you need to install the required dependencies. These are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

### **4. Train the Model**

To train the model, execute the `train.py` script. This will load the dataset, extract features, train the classifier, and save the trained model to a file:

```bash
python train.py
```

### **5. Test the Model on Unseen Data**

Once the model is trained, you can evaluate its performance on unseen test data using the `main.py` script:

```bash
python main.py
```

This script will load the saved model, preprocess the test data (human and non-human images), and output the model's performance.

---

## **Dependencies**

- **Python 3.x**
- **NumPy**
- **Scikit-learn**
- **Scikit-image**
- **Matplotlib**
- **Joblib**
- **Torchvision**

To install these dependencies, simply run:

```bash
pip install -r requirements.txt
```

---

## **Contact**

For any questions or suggestions, feel free to contact [your_email@example.com](mailto:your_email@example.com).
