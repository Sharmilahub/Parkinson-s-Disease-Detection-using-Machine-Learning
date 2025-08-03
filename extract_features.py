import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

def extract_all_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))

    # GLCM features
    glcm = graycomatrix(img, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # LBP features
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # HOG features
    hog_features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    feature_vector = [contrast, dissimilarity, homogeneity, energy, correlation]
    feature_vector.extend(hist)
    feature_vector.extend(hog_features[:20])  # Limit HOG size for efficiency

    return np.array(feature_vector)