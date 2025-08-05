import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = '/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/WEEK8/ImageDataset'
IMAGE_SIZE = (64, 64)       # Resize all images to 64x64
CLASSES = ['baseball', 'bmx']  # Folder names = class names


X = []
y = []

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, cls)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            X.append(img.flatten())  # flatten to 1D vector
            y.append(label)

X = np.array(X)
y = np.array(y)

pca = PCA(n_components=100)  # Reduce to 100 features
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
model = GridSearchCV(svm.SVC(), param_grid, cv=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Best Parameters:", model.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
