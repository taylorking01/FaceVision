import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def plot_samples(X, y, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(X):
                axes[i, j].imshow(X[idx], cmap='gray')
                axes[i, j].axis('off')
                axes[i, j].set_title(f"Label: {int(y[idx])}")
                idx += 1
    plt.tight_layout()
    plt.show()

def load_images_from_folder(folder, image_size=(62, 47)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize(image_size, Image.ANTIALIAS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
    return images
