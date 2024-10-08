import cv2
import torch
from torchvision import transforms
from cnn_model import FaceDetectionCNN

# Load the trained model
model = FaceDetectionCNN()
model.load_state_dict(torch.load('face_detection_cnn.pth'))
model.eval()
print("Model loaded successfully.")

# Define preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    # Open the webcam
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

        # Convert the frame to grayscale and preprocess
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame
        img_tensor = preprocess(frame_rgb).unsqueeze(0)  # Add batch dimension

        # Optionally, print the shape of the input tensor
        # print(f"Input tensor shape: {img_tensor.shape}")

        # Predict using the trained model
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # If a face is detected, draw a bounding box
        if predicted.item() == 1:
            print("Face detected!")
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (int(width * 0.3), int(height * 0.3)),
                          (int(width * 0.7), int(height * 0.7)), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
