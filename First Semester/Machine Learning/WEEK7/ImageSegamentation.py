import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

# ======================
# Custom KMeans (from scratch)
# ======================
class KMeansScratch:
    def __init__(self, n_clusters=4, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):
            print(f"Iteration {iteration + 1}/{self.max_iters}")  # progress

            # Efficient squared distance computation
            distances = (
                np.sum(X**2, axis=1, keepdims=True)
                - 2 * X @ self.centroids.T
                + np.sum(self.centroids**2, axis=1)
            )
            self.labels_ = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                print("Converged.")
                break

            self.centroids = new_centroids

        # Compute total inertia
        self.inertia_ = sum(
            np.sum((X[self.labels_ == i] - self.centroids[i]) ** 2)
            for i in range(self.n_clusters)
        )
        return self

# ======================
# Load and downsample image
# ======================
img = io.imread('/home/xgang/WinShared_D/Graduation/First_Year/Machine-Learning/WEEK7/beach.jpg')
img = resize(img, (150, 150), anti_aliasing=True, preserve_range=True).astype(np.uint8)  # Downsample
rows, cols, ch = img.shape

# Create feature vector: [R, G, B, x, y]
X = []
for i in range(rows):
    for j in range(cols):
        r, g, b = img[i, j]
        X.append([r, g, b, i / rows * 255, j / cols * 255])  # Normalize positions
X = np.array(X)

# ======================
# Elbow method to find optimal k
# ======================
inertias = []
K_range = range(1, 11)
for k in K_range:
    print(f"\nRunning KMeans for k={k}")
    kmeans = KMeansScratch(n_clusters=k, random_state=0).fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure()
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# ======================
# Final KMeans clustering
# ======================
optimal_k = 4
kmeans = KMeansScratch(n_clusters=optimal_k, random_state=0).fit(X)
labels = kmeans.labels_

# Reshape labels to image
segmented_img = np.zeros((rows, cols, 3), dtype=np.uint8)
label_colors = np.random.randint(0, 255, size=(optimal_k, 3))

for idx, label in enumerate(labels):
    i = idx // cols
    j = idx % cols
    segmented_img[i, j] = label_colors[label]

# ======================
# Display results
# ======================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image (Downsampled)')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Segmented Image (k={optimal_k})')
plt.imshow(segmented_img)
plt.axis('off')
plt.tight_layout()
plt.show()
