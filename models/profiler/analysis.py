import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class LatentVisualizer:
    """
    Collects latent opponent profiles (z_opp) during evaluation rollouts and 
    projects them into 2D space to verify that the profiler accurately separates 
    different opponent archetypes.
    """
    def __init__(self):
        self.z_vectors = []
        self.labels = []

    def record(self, z_opp: torch.Tensor, opponent_label: str):
        """
        Records a latent vector and its corresponding opponent label.
        
        Args:
            z_opp (torch.Tensor): The latent profile of shape (1, z_dim) or (z_dim,).
            opponent_label (str): Human-readable name of the opponent strategy 
                                  (e.g., "Aggressive", "Random", "Rule-Based").
        """
        # Detach from graph, move to CPU, and flatten to a 1D array
        if isinstance(z_opp, torch.Tensor):
            z_opp = z_opp.detach().cpu().numpy().flatten()
            
        self.z_vectors.append(z_opp)
        self.labels.append(opponent_label)

    def clear(self):
        """Resets the stored data."""
        self.z_vectors = []
        self.labels = []

    def plot_clusters(self, method: str = 'tsne', save_path: str = None):
        """
        Reduces the dimensionality of the collected latent vectors to 2D and 
        plots them as a color-coded scatter plot.
        
        Args:
            method (str): 'tsne' or 'pca'. t-SNE is generally better for showing 
                          non-linear cluster separation.
            save_path (str): If provided, saves the figure to this filepath.
        """
        if len(self.z_vectors) < 5:
            print("Not enough data points to cluster. Run more evaluation episodes.")
            return

        X = np.stack(self.z_vectors)
        y = np.array(self.labels)

        # Dimensionality Reduction
        if method.lower() == 'tsne':
            # Adjust perplexity based on dataset size to avoid errors on small datasets
            perplexity = min(30, max(1, len(X) - 1))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title = "t-SNE Projection of Opponent Latent Profiles"
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title = "PCA Projection of Opponent Latent Profiles"
        else:
            raise ValueError("Method must be 'tsne' or 'pca'.")

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Create a visually distinct scatter plot
        sns.scatterplot(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1], 
            hue=y, 
            palette="deep", 
            s=60, 
            alpha=0.8,
            edgecolor="k"
        )
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        
        # Place legend outside the plot to avoid covering data points
        plt.legend(title="Opponent Archetype", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            # Ensure the target directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to {save_path}")
        else:
            plt.show()