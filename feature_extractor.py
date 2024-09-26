import numpy as np

def extract_features(X):
    # Flatten the images
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat

def extract_features_single(img):
    # Flatten a single image
    img_flat = img.flatten()
    return img_flat
