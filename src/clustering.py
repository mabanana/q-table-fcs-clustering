"""
Clustering Module

Provides unified interface for clustering with GPU (cuML) and CPU (sklearn) support.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# Try to import cuML for GPU acceleration
try:
    from cuml.cluster import KMeans as cuMLKMeans
    from cuml.metrics import silhouette_score as cuml_silhouette_score
    import cudf
    GPU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("cuML detected - GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("cuML not available - using CPU-only sklearn")

# Import sklearn as fallback
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    Unified clustering interface supporting both GPU (cuML) and CPU (sklearn).
    """
    
    def __init__(self, use_gpu: bool = True, fallback_to_cpu: bool = True):
        """
        Initialize the clustering engine.
        
        Args:
            use_gpu: Whether to attempt using GPU acceleration
            fallback_to_cpu: Whether to fall back to CPU if GPU unavailable
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.fallback_to_cpu = fallback_to_cpu
        
        if use_gpu and not GPU_AVAILABLE and not fallback_to_cpu:
            raise RuntimeError(
                "GPU acceleration requested but cuML not available. "
                "Install cuML or set fallback_to_cpu=True"
            )
        
        if self.use_gpu:
            logger.info("Clustering engine initialized with GPU acceleration")
        else:
            logger.info("Clustering engine initialized with CPU-only mode")
    
    def kmeans(
        self,
        data: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
        max_iter: int = 300,
        init_centers: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, object]:
        """
        Perform K-means clustering.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
            init_centers: Optional explicit initial centers (shape: n_clusters x n_features)
            
        Returns:
            Tuple of (labels, inertia, model)
        """
        if self.use_gpu and init_centers is None:
            try:
                # Convert to cuDF DataFrame for GPU processing
                data_gpu = cudf.DataFrame(data)
                
                # Perform GPU K-means
                model = cuMLKMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    max_iter=max_iter
                )
                model.fit(data_gpu)
                
                # Get results
                labels = model.labels_.to_numpy()
                inertia = float(model.inertia_)
                
                logger.debug(
                    f"GPU K-means: n_clusters={n_clusters}, "
                    f"inertia={inertia:.2f}"
                )
                
                return labels, inertia, model
                
            except Exception as e:
                if self.fallback_to_cpu:
                    logger.warning(
                        f"GPU clustering failed ({str(e)}), falling back to CPU"
                    )
                else:
                    raise

        if init_centers is not None:
            init_centers = np.asarray(init_centers, dtype=float)
            if init_centers.shape != (n_clusters, data.shape[1]):
                raise ValueError(
                    "init_centers has invalid shape. "
                    f"Expected {(n_clusters, data.shape[1])}, got {init_centers.shape}"
                )
        
        # CPU K-means (sklearn)
        if init_centers is not None:
            model = SKLearnKMeans(
                n_clusters=n_clusters,
                init=init_centers,
                random_state=random_state,
                max_iter=max_iter,
                n_init=1
            )
        else:
            model = SKLearnKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=10
            )
        labels = model.fit_predict(data)
        inertia = model.inertia_
        
        logger.debug(
            f"CPU K-means: n_clusters={n_clusters}, "
            f"inertia={inertia:.2f}"
        )
        
        return labels, inertia, model
    
    def gaussian_mixture(
        self,
        data: np.ndarray,
        n_components: int,
        random_state: int = 42,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, float, object]:
        """
        Perform Gaussian Mixture Model clustering.
        
        Note: Currently only supports CPU (sklearn) implementation.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            n_components: Number of mixture components
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (labels, bic, model)
        """
        model = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter
        )
        labels = model.fit_predict(data)
        bic = model.bic(data)
        
        logger.debug(
            f"GMM clustering: n_components={n_components}, "
            f"BIC={bic:.2f}"
        )
        
        return labels, bic, model
    
    def silhouette_score(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calculate silhouette score for clustering results.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            labels: Cluster labels
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        # Need at least 2 clusters for silhouette score
        n_unique_labels = len(np.unique(labels))
        if n_unique_labels < 2:
            logger.warning("Cannot calculate silhouette score with < 2 clusters")
            return 0.0
        
        # Need more samples than clusters
        if len(labels) <= n_unique_labels:
            logger.warning("Too few samples for silhouette score")
            return 0.0
        
        if self.use_gpu and GPU_AVAILABLE:
            try:
                # Convert to GPU arrays
                data_gpu = cudf.DataFrame(data)
                labels_gpu = cudf.Series(labels)
                
                score = float(cuml_silhouette_score(data_gpu, labels_gpu))
                
                logger.debug(f"GPU silhouette score: {score:.4f}")
                return score
                
            except Exception as e:
                if self.fallback_to_cpu:
                    logger.warning(
                        f"GPU silhouette score failed ({str(e)}), falling back to CPU"
                    )
                else:
                    raise
        
        # CPU silhouette score
        score = sklearn_silhouette_score(data, labels)
        
        logger.debug(f"CPU silhouette score: {score:.4f}")
        return score
    
    def calculate_inertia(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        centers: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate inertia (within-cluster sum of squares).
        
        Args:
            data: Data array of shape (n_samples, n_features)
            labels: Cluster labels
            centers: Optional cluster centers. If None, will be calculated.
            
        Returns:
            Inertia value
        """
        if centers is None:
            # Calculate centers as mean of each cluster
            unique_labels = np.unique(labels)
            centers = np.array([
                data[labels == label].mean(axis=0)
                for label in unique_labels
            ])
        
        # Calculate sum of squared distances to nearest center
        inertia = 0.0
        for i, label in enumerate(labels):
            center = centers[label]
            dist_squared = np.sum((data[i] - center) ** 2)
            inertia += dist_squared
        
        return inertia


def cluster_and_evaluate(
    data: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    use_gpu: bool = True,
    random_state: int = 42
) -> dict:
    """
    Convenience function to cluster data and evaluate results.
    
    Args:
        data: Data array of shape (n_samples, n_features)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'gmm')
        use_gpu: Whether to use GPU acceleration
        random_state: Random seed
        
    Returns:
        Dictionary with clustering results and metrics
    """
    engine = ClusteringEngine(use_gpu=use_gpu)
    
    if method == "kmeans":
        labels, inertia, model = engine.kmeans(
            data,
            n_clusters,
            random_state=random_state
        )
        
        # Calculate silhouette score
        silhouette = engine.silhouette_score(data, labels)
        
        return {
            'labels': labels,
            'inertia': inertia,
            'silhouette_score': silhouette,
            'model': model,
            'n_clusters': n_clusters,
            'method': 'kmeans'
        }
    
    elif method == "gmm":
        labels, bic, model = engine.gaussian_mixture(
            data,
            n_clusters,
            random_state=random_state
        )
        
        # Calculate silhouette score
        silhouette = engine.silhouette_score(data, labels)
        
        return {
            'labels': labels,
            'bic': bic,
            'silhouette_score': silhouette,
            'model': model,
            'n_clusters': n_clusters,
            'method': 'gmm'
        }
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def assign_diagnosis_from_clusters(
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    positive_class: str = "positive"
) -> Tuple[dict, np.ndarray]:
    """
    Map cluster labels to diagnosis (HIV+ or HIV-) using majority voting.
    
    Args:
        labels: Cluster labels
        true_labels: Optional true diagnosis labels for training
        positive_class: Label for positive class in true_labels
        
    Returns:
        Tuple of (cluster_to_diagnosis mapping, predicted diagnoses)
    """
    if true_labels is None:
        # Without true labels, cannot determine mapping
        # Return default mapping (not very useful)
        unique_clusters = np.unique(labels)
        cluster_to_diagnosis = {
            cluster: "positive" if cluster == 0 else "negative"
            for cluster in unique_clusters
        }
        predictions = np.array([
            cluster_to_diagnosis[label] for label in labels
        ])
        return cluster_to_diagnosis, predictions
    
    # Count positive/negative samples in each cluster
    unique_clusters = np.unique(labels)
    cluster_to_diagnosis = {}
    
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        # Count positives in this cluster
        n_positive = np.sum(cluster_true_labels == positive_class)
        n_negative = np.sum(cluster_true_labels != positive_class)
        
        # Assign diagnosis based on majority
        if n_positive > n_negative:
            cluster_to_diagnosis[cluster] = "positive"
        else:
            cluster_to_diagnosis[cluster] = "negative"
        
        logger.debug(
            f"Cluster {cluster}: {n_positive} positive, {n_negative} negative "
            f"-> {cluster_to_diagnosis[cluster]}"
        )
    
    # Generate predictions
    predictions = np.array([
        cluster_to_diagnosis[label] for label in labels
    ])
    
    return cluster_to_diagnosis, predictions
