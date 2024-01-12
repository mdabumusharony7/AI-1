import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

feature_vectors = np.array([])
labels = np.array([])

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

accuracy = svm_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
