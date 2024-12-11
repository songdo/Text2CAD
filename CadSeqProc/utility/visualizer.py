from CadSeqProc.utility.utils import ensure_dir
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from CadSeqProc.OCCUtils.Common import color

class TSNEVisualizer():
    def __init__(self):
        pass
    
    def tsne_embed(self,pc,embedding,n_components=2,eps=0.5,min_samples=5,filename=None,output_dir=None):
        tensor_embedded=self.get_tsne_embed(embedding) # Shape: (N, 2), 2D t-SNE transformed data
        cluster_labels=self.cluster_data(tensor_embedded,eps,min_samples) # Cluster labels for each point

        # Assign colors to clusters
        colors = self.assign_colors(cluster_labels)
        self.plot_tsne(tensor_embedded,colors,filename)

        # Save the point cloud with colors
        if filename and output_dir is not None:
            save_path = os.path.join(output_dir,filename+".ply")
            #ensure_dir(save_path)
            self.save_point_cloud(pc, colors, save_path)
        else:
            return pc,colors

    def get_tsne_embed(self,embedding):
        """
        embedding: tensor of shape (N,E)
        """
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tensor_embedded = tsne.fit_transform(embedding)

        return tensor_embedded

    def cluster_data(self,tensor_embedded, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(tensor_embedded)
        return cluster_labels

    def assign_colors(self,cluster_labels):
        num_clusters = len(np.unique(cluster_labels))
        colormap = plt.cm.get_cmap('tab20', num_clusters)
        colors = colormap(cluster_labels)
        return colors

    def save_point_cloud(self,pc, colors, save_path):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # We only use the RGB part of the colormap
        o3d.io.write_point_cloud(save_path, point_cloud)
    
    def plot_tsne(self,tsne_embedding,colors,title="tsne"):
        plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],color=colors)
        plt.title(title)
        plt.show()


class TSNEVisualizerWithSubplots(TSNEVisualizer):
    def plot_multiple_tsne(self, embeddings, colors, titles, filename=None,saveDir=None):
        num_plots = len(embeddings)
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        axs = axs.ravel()

        for i, (embedding, color, title) in enumerate(zip(embeddings, colors, titles)):
            ax = axs[i]
            ax.scatter(embedding[:, 0], embedding[:, 1], color=color)
            ax.set_title(title)

        # Remove empty subplots
        for i in range(num_plots, rows * cols):
            fig.delaxes(axs[i])

        plt.tight_layout()
        if filename is None or saveDir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveDir,filename+".jpg"),bbox_inches="tight")