import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    """Calcule la distance euclidienne entre deux vecteurs."""
    return np.sqrt(np.sum((a - b) ** 2))


class DynamicClouds:
    """Implémentation from scratch de l'algorithme des nuées dynamiques."""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.labels = None
        self.inertia_history = []

    def initialize_centers(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.centers = X[indices].copy()

    def assign_clusters(self, X):
        distances = np.array([
            [euclidean_distance(x, center) for center in self.centers]
            for x in X
        ])
        return np.argmin(distances, axis=1)

    def update_centers(self, X, labels):
        new_centers = np.zeros_like(self.centers)

        for k in range(self.n_clusters):
            points = X[labels == k]

            if len(points) == 0:
                # Réinitialisation d'un nuage vide avec un point aléatoire
                rng = np.random.default_rng(self.random_state + k)
                new_centers[k] = X[rng.integers(len(X))]
            else:
                new_centers[k] = np.mean(points, axis=0)

        return new_centers

    def compute_inertia(self, X, labels):
        total = 0.0
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) > 0:
                total += np.sum((points - self.centers[k]) ** 2)
        return total

    def fit(self, X):
        self.initialize_centers(X)

        for _ in range(self.max_iter):
            labels = self.assign_clusters(X)
            new_centers = self.update_centers(X, labels)

            self.centers = new_centers
            inertia = self.compute_inertia(X, labels)
            self.inertia_history.append(inertia)

            if self.labels is not None and np.array_equal(labels, self.labels):
                self.labels = labels
                break

            if np.max(np.linalg.norm(new_centers - self.centers, axis=1)) < self.tol:
                self.labels = labels
                break

            self.labels = labels

        return self


def generate_data(random_state=42):
    """Génère trois nuages de points 2D pour l'expérimentation."""
    rng = np.random.default_rng(random_state)

    cloud1 = rng.normal(loc=[2, 2], scale=0.8, size=(120, 2))
    cloud2 = rng.normal(loc=[7, 7], scale=0.9, size=(120, 2))
    cloud3 = rng.normal(loc=[2, 8], scale=0.7, size=(120, 2))

    return np.vstack([cloud1, cloud2, cloud3])


if __name__ == "__main__":
    X = generate_data()

    model = DynamicClouds(
        n_clusters=3,
        max_iter=100,
        tol=1e-4,
        random_state=42
    )

    model.fit(X)

    print("Centres finaux :")
    print(model.centers)

    print("\nNombre d'itérations :", len(model.inertia_history))

    # Représentation des nuées finales
    plt.figure(figsize=(8, 6))
    for k in range(model.n_clusters):
        points = X[model.labels == k]
        plt.scatter(points[:, 0], points[:, 1], label=f"Nuée {k + 1}")

    plt.scatter(
        model.centers[:, 0],
        model.centers[:, 1],
        marker="X",
        s=200,
        label="Centres"
    )

    plt.title("Implémentation de l'algorithme des nuées dynamiques")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.savefig("clusters.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Courbe de convergence
    plt.figure(figsize=(8, 5))
    plt.plot(model.inertia_history, marker="o")
    plt.title("Convergence de l'algorithme")
    plt.xlabel("Itération")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.savefig("convergence.png", dpi=300, bbox_inches="tight")
    plt.show()
