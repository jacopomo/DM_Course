import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, Checkbox, Button, VBox, HBox, Output
from IPython.display import display, clear_output

class Point:
    """A point in 2D space with methods to generate random coordinates and assign to clusters."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def random_normal(self, mean_x, mean_y, sigma_x, sigma_y):
        self.x = np.random.normal(mean_x, sigma_x)
        self.y = np.random.normal(mean_y, sigma_y)

class Cluster:
    """A cluster represented by its centroid."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.points = []

    def add_point(self, point):
        self.points.append(point)

    def mean(self):
        if not self.points:
            return self.x, self.y
        mean_x = np.mean([p.x for p in self.points])
        mean_y = np.mean([p.y for p in self.points])
        return mean_x, mean_y
    
    def update_centroid(self):
        self.x, self.y = self.mean()

    def square_mean_error(self):
        return np.sum([(p.x - self.x)**2 + (p.y - self.y)**2 for p in self.points])

class Gaussian:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

def generate_points_normal(mean_x, mean_y, sigma_x, sigma_y, points_per_cluster=100):
    points = []
    for _ in range(points_per_cluster):
        p = Point(0, 0)
        p.random_normal(mean_x, mean_y, sigma_x, sigma_y)
        points.append(p)
    return points

def generate_clustered_data(gaussians, points_per_cluster=100):
    all_points = []
    for gaussian in gaussians:
        cluster_points = generate_points_normal(
            mean_x=gaussian[0].mean,
            mean_y=gaussian[1].mean,
            sigma_x=gaussian[0].sigma,
            sigma_y=gaussian[1].sigma,
            points_per_cluster=points_per_cluster
        )
        all_points.extend(cluster_points)
    return all_points

def generate_gaussians(num_clusters=3, mean_range=(0, 10), sigma_range=(0.5, 1.5)):
    gaussians = []
    for _ in range(num_clusters):
        mean_x = np.random.uniform(*mean_range)
        mean_y = np.random.uniform(*mean_range)
        sigma_x = np.random.uniform(*sigma_range)
        sigma_y = np.random.uniform(*sigma_range)
        gaussians.append((Gaussian(mean_x, sigma_x), Gaussian(mean_y, sigma_y)))
    return gaussians

def k_means_alg(points, k, max_iters=5):
    n_points = len(points)
    if n_points == 0:
        return []
    # clamp k to number of available points to avoid np.random.choice errors
    k = min(k, n_points)
    indices = np.random.choice(n_points, k, replace=False)
    centroids = [Cluster(points[i].x, points[i].y) for i in indices]

    for _ in range(max_iters):
        for c in centroids:
            c.points = []

        for point in points:
            distances = np.sqrt((point.x - np.array([c.x for c in centroids]))**2 +
                                (point.y - np.array([c.y for c in centroids]))**2)
            nearest = np.argmin(distances)
            centroids[nearest].add_point(point)

        for c in centroids:
            c.update_centroid()

    return centroids

def incremental_k_means(points, min_k, max_k, max_iters=5):
    sse = []
    n_points = len(points)
    for k in range(min_k, max_k + 1):
        if k <= 0:
            sse.append(np.nan)
            continue
        if n_points == 0:
            sse.append(np.nan)
            continue
        if k > n_points:
            # cannot form more clusters than points; append NaN to indicate invalid
            sse.append(np.nan)
            continue
        centroids = k_means_alg(points, k=k, max_iters=max_iters)
        sse.append(np.sum([c.square_mean_error() for c in centroids]))
    return sse

def plot_clusters(points, centroids, gaussians, ax):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    ax.clear()

    for idx, centroid in enumerate(centroids):
        cluster_points = centroid.points
        ax.scatter([p.x for p in cluster_points], [p.y for p in cluster_points],
                   color=colors[idx % len(colors)], label=f'Cluster {idx+1}', alpha=0.5)

    ax.scatter([c.x for c in centroids], [c.y for c in centroids],
               color='black', marker='X', s=200, label='Centroids')

    for idx, gaussian in enumerate(gaussians):
        ax.scatter(gaussian[0].mean, gaussian[1].mean, color='orange', marker='D', s=100)
        ellipse = patches.Ellipse(
            (gaussian[0].mean, gaussian[1].mean),
            2*gaussian[0].sigma,
            2*gaussian[1].sigma,
            color='orange', fill=False, linestyle='dashed'
        )
        ax.add_patch(ellipse)

    ax.set_title('K-Means Clustering')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)
    ax.legend()


def plot_sse(sse, min_k, ax):
    ax.clear()
    ax.plot(range(min_k, min_k + len(sse)), sse, marker='o')
    ax.set_title('SSE vs Number of Clusters (k)')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Sum of Squared Errors (SSE)')
    ax.grid(True)

# ✅ Interactive K-Means runner with Regenerate button
def run_interactive_kmeans():
    # --- Widgets ---
    num_clusters = IntSlider(min=1, max=10, step=1, value=5, description='Clusters')
    points_per_cluster = IntSlider(min=10, max=500, step=10, value=100, description='Points/Cluster')
    mean_min = FloatSlider(min=-10, max=10, step=0.5, value=0, description='Mean min')
    mean_max = FloatSlider(min=0, max=50, step=0.5, value=10, description='Mean max')
    sigma_min = FloatSlider(min=0.1, max=5, step=0.1, value=1.0, description='Sigma min')
    sigma_max = FloatSlider(min=1, max=10, step=0.1, value=2.0, description='Sigma max')
    k = IntSlider(min=1, max=10, step=1, value=5, description='K')
    max_iters = IntSlider(min=1, max=50, step=1, value=10, description='Iterations')
    min_k = IntSlider(min=1, max=10, step=1, value=1, description='Min K')
    max_k = IntSlider(min=1, max=20, step=1, value=15, description='Max K')
    show_sse = Checkbox(value=True, description='Show SSE Plot')
    regenerate = Button(description='🔁 Regenerate Data', button_style='info')

    out = Output()

    # --- Main logic ---
    def update_plot(*args):
        with out:
            clear_output(wait=True)
            plt.close('all')  # close all old figures

            mean_range = (min(mean_min.value, mean_max.value), max(mean_min.value, mean_max.value))
            sigma_range = (min(sigma_min.value, sigma_max.value), max(sigma_min.value, sigma_max.value))

            gaussians = generate_gaussians(num_clusters.value, mean_range, sigma_range)
            points = generate_clustered_data(gaussians=gaussians, points_per_cluster=points_per_cluster.value)
            centroids = k_means_alg(points, k=k.value, max_iters=max_iters.value)

            # Create one figure with two axes (clusters + optional SSE)
            if show_sse.value:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            else:
                fig, ax1 = plt.subplots(figsize=(10, 8))
                ax2 = None

            plot_clusters(points, centroids, gaussians, ax1)

            if show_sse.value and ax2 is not None:
                sse = incremental_k_means(points, min_k=min_k.value, max_k=max_k.value, max_iters=max_iters.value)
                plot_sse(sse, min_k=min_k.value, ax=ax2)

            plt.tight_layout()
            plt.show()

    # --- Event bindings ---
    regenerate.on_click(update_plot)
    for w in [num_clusters, points_per_cluster, mean_min, mean_max, sigma_min, sigma_max, k, max_iters, min_k, max_k, show_sse]:
        w.observe(update_plot, names='value')

    # --- Layout ---
    controls = VBox([
        HBox([num_clusters, points_per_cluster]),
        HBox([mean_min, mean_max]),
        HBox([sigma_min, sigma_max]),
        HBox([k, max_iters]),
        HBox([min_k, max_k]),
        HBox([show_sse, regenerate])
    ])

    display(controls, out)
    update_plot()

# ✅ Run only when inside an IPython kernel (e.g. Jupyter)
try:
    import IPython
    if IPython.get_ipython() is not None:
        run_interactive_kmeans()
except Exception:
    # not running in IPython environment — skip auto-run
    pass
