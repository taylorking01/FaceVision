import numpy as np
from skimage.feature import hog

def extract_features(X):
    # Compute HOG features for each image in X
    hog_features = []
    for img in X:
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

def extract_features_single(img):
    # Compute HOG features for a single image
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return fd
