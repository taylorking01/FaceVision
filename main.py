import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Load the trained model
clf = joblib.load('face_detection_model.joblib')
print("Model loaded successfully.")

# Function to preprocess each frame and extract HOG features
def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to the same dimensions as used during training (62x47)
    resized_frame = cv2.resize(gray_frame, (62, 47))
    
    # Extract HOG features
    features = hog(resized_frame, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    return np.array([features])

# Function to draw a bounding box if a face is detected
def draw_bounding_box(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Preprocess the frame
        features = preprocess_frame(frame)
        
        # Predict using the trained model
        prediction = clf.predict(features)
        
        # If the model predicts a face (label 1), draw a bounding box and print to console
        if prediction == 1:
            print("Face detected!")  # Output to console when a face is detected
            height, width = frame.shape[:2]
            draw_bounding_box(frame, int(width * 0.3), int(height * 0.3), int(width * 0.4), int(height * 0.4))
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
