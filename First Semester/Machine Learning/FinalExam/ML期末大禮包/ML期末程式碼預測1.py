
# ML_Final_Function_Extension_Solution.py
# 解答版：完整功能擴充

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torchvision import models
import torch
import torch.nn as nn

# ===== 初始資料與 KMeans 分群 =====
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# KMeans 建模
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 畫出 KMeans 結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('KMeans Clustering Result')
plt.show()

# ===== 功能 1: 使用 SVM 並進行交叉驗證 =====
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0)
svm_model = SVC(kernel='linear')
cv_scores = cross_val_score(svm_model, X, labels, cv=5)
print(f"✅ SVM 5-fold CV 平均準確率: {np.mean(cv_scores):.4f}")

# ===== 功能 2: Decision Tree vs Random Forest =====
dtree = DecisionTreeClassifier(criterion='entropy')
rf = RandomForestClassifier(n_estimators=10)
dtree.fit(X_train, y_train)
rf.fit(X_train, y_train)

acc_tree = dtree.score(X_test, y_test)
acc_rf = rf.score(X_test, y_test)
print(f"✅ Decision Tree Test Accuracy: {acc_tree:.4f}")
print(f"✅ Random Forest Test Accuracy: {acc_rf:.4f}")

# ===== 功能 3: 模擬 VGG16 特徵提取（假設 X 為圖像向量）=====
# 模擬 2D 轉為圖像格式：[N, C, H, W] = [300, 1, 10, 10]
X_image = X[:, :2]  # 僅用前兩維
X_padded = np.zeros((300, 100))
X_padded[:, :2] = X_image
X_reshaped = torch.tensor(X_padded, dtype=torch.float32).view(-1, 1, 10, 10)

# 使用 VGG 特徵提取
vgg_model = models.vgg16(pretrained=True)
for param in vgg_model.features.parameters():
    param.requires_grad = False
vgg_features = vgg_model.features[:2](X_reshaped.repeat(1, 3, 1, 1))  # 模擬特徵提取
print("✅ VGG 特徵張量維度 (模擬):", vgg_features.shape)

# ===== 功能 4: Bonus - Silhouette Score 與視覺化 =====
silhouette = silhouette_score(X, labels)
print(f"✅ Silhouette Score (k=3): {silhouette:.4f}")

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
for i, center in enumerate(centers):
    plt.text(center[0], center[1], f"Cluster {i}", fontsize=12, color='black')
plt.title("KMeans + Cluster Center Labels")
plt.show()
