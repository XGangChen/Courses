
# marketing_analysis_embedded.py
# 解答程式碼（內嵌資料版本）

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===== 資料內嵌於程式碼中 =====
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8,
                    176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6,
           112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2,
                  123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6,
              16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6],
    'Class': ['Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Medium',
              'Low', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Medium', 'Low']
})

# ===== 1. KMeans 分群與視覺化 =====
features = data[['SocialMedia', 'TV', 'Billboard']]
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)
data['Cluster'] = labels

sns.scatterplot(data=data, x='SocialMedia', y='Sales', hue='Cluster', palette='Set2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], c='black', marker='x', s=200)
plt.title("KMeans Clustering")
plt.show()

# ===== 2. 使用 SVM / Decision Tree / Random Forest 分類 Class =====
X = features
y = LabelEncoder().fit_transform(data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ {name} Accuracy: {acc:.4f}")

# ===== 3. 示意 Transfer Learning 架構（假設為圖像特徵）=====
print("🔄 假設此表格資料經轉換為影像格式後可餵入 VGG16 提取特徵，再用全連接層分類（此處略過實作）")

# ===== 4. Silhouette Score 求最佳 k 值 =====
scores = []
for k in range(2, 6):
    kmeans_k = KMeans(n_clusters=k, random_state=42).fit(features)
    score = silhouette_score(features, kmeans_k.labels_)
    scores.append((k, score))

best_k = max(scores, key=lambda x: x[1])
print("✅ Best K by Silhouette Score:", best_k[0], "Score:", best_k[1])

plt.plot([x[0] for x in scores], [x[1] for x in scores], marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()
