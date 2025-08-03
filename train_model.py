import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from extract_features import extract_all_features

def load_dataset(base_dir):
    X, y = [], []
    for category in ["healthy", "parkinson"]:
        label = 0 if category == "healthy" else 1
        folder = os.path.join(base_dir, category)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                features = extract_all_features(path)
                X.append(features)
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)