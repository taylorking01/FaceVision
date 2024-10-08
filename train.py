import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data
from cnn_model import FaceDetectionCNN
from sklearn.model_selection import train_test_split
from datetime import datetime

def train_model():
    # Record the start time
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%H:%M:%S')}")

    # Load and preprocess data
    X, y = load_data()
    print("Data loaded successfully.")

    # Convert lists to tensors
    print(f"Shape of first image tensor: {X[0].shape}")
    print(f"Shape of last image tensor: {X[-1].shape}")
    X = torch.stack(X)  # Shape: [num_samples, channels, height, width]
    y = torch.tensor(y).long()  # Labels as LongTensor for CrossEntropyLoss

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss function, optimizer
    model = FaceDetectionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'face_detection_cnn.pth')
    print("Model saved to face_detection_cnn.pth")

    # Record the end time
    end_time = datetime.now()
    print(f"Training finished at: {end_time.strftime('%H:%M:%S')}")

    # Calculate total training time
    training_time = end_time - start_time
    hours, remainder = divmod(training_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {hours} hrs, {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    train_model()
