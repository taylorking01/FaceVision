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
