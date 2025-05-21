import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform

def herc_portfolio(returns, correlation_method='pearson', linkage_method='single', risk_measure='MV', 
                   risk_free_rate=0, k=None, max_k=10):
    """
    Implements the Hierarchical Equal Risk Contribution (HERC) portfolio optimization algorithm.
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns DataFrame with assets as columns and observations as rows
    correlation_method : str
        Method used to calculate correlation matrix ('pearson', 'spearman', 'kendall')
    linkage_method : str
        Linkage method for hierarchical clustering
    risk_measure : str
        Risk measure used ('MV' for variance, 'vol' for standard deviation, 'equal' for equal weighting)
    risk_free_rate : float
        Risk-free rate used in risk calculations
    k : int, optional
        Number of clusters (if None, it's determined using the two-difference gap statistic)
    max_k : int
        Maximum number of clusters to consider when calculating optimal k
        
    Returns:
    --------
    DataFrame
        Portfolio weights for each asset
    """
    
    # Step 1: Calculate correlation matrix
    corr = returns.corr(method=correlation_method)
    
    # Step 2: Convert correlation to distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Step 3: Hierarchical clustering
    # Convert distance matrix to condensed form
    dist_condensed = squareform(dist)
    
    # Perform hierarchical clustering
    Z = hr.linkage(dist_condensed, method=linkage_method)
    
    # Step 4: Determine optimal number of clusters if not provided
    if k is None:
        k = _optimal_number_of_clusters(dist, Z, max_k)
    
    # Step 5: Get clusters
    clustering_inds = hr.fcluster(Z, k, criterion="maxclust")
    clusters = {i: [] for i in range(1, max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)
    
    # Calculate covariance matrix
    cov = returns.cov()
    assets = returns.columns.tolist()
    
    # Step 6: Initialize weights
    weights = pd.Series(1.0, index=assets)  # Set initial weights to 1
    
    # Step 7: Recursive bisection at the clusters level
    # Transform linkage to tree
    root, nodes = hr.to_tree(Z, rd=True)
    nodes = np.array(nodes)
    nodes_dist = np.array([i.dist for i in nodes])
    idx = np.argsort(nodes_dist)
    nodes = nodes[idx][::-1].tolist()
    
    # Loop through nodes until k-1 bifurcations
    for i in nodes[:k-1]:
        if not i.is_leaf():  # skip leaf-nodes
            left = i.get_left().pre_order()  # get left cluster
            right = i.get_right().pre_order()  # get right cluster
            left_set = set(left)
            right_set = set(right)
            left_risk = 0
            right_risk = 0
            left_cluster = []
            right_cluster = []
            
            # Allocate weight to clusters
            if risk_measure == "equal":
                alpha = 0.5
            else:
                # Calculate risk for each cluster
                for j in clusters.keys():
                    cluster_indices = clusters[j]
                    
                    # Check if cluster is in left branch
                    if set(cluster_indices).issubset(left_set):
                        # Risk calculation for left cluster
                        cluster_cov = cov.iloc[cluster_indices, cluster_indices]
                        cluster_returns = returns.iloc[:, cluster_indices]
                        
                        # Simple inverse risk weights within cluster
                        if risk_measure == "vol":
                            cluster_vols = np.sqrt(np.diag(cluster_cov))
                            cluster_weights = 1 / cluster_vols
                            cluster_weights = cluster_weights / np.sum(cluster_weights)
                            cluster_risk = np.sqrt(cluster_weights @ cluster_cov @ cluster_weights)
                        else:  # MV (Variance)
                            cluster_vols = np.diag(cluster_cov)
                            cluster_weights = 1 / cluster_vols
                            cluster_weights = cluster_weights / np.sum(cluster_weights)
                            cluster_risk = cluster_weights @ cluster_cov @ cluster_weights
                        
                        left_risk += cluster_risk
                        left_cluster += cluster_indices
                    
                    # Check if cluster is in right branch
                    elif set(cluster_indices).issubset(right_set):
                        # Risk calculation for right cluster
                        cluster_cov = cov.iloc[cluster_indices, cluster_indices]
                        cluster_returns = returns.iloc[:, cluster_indices]
                        
                        # Simple inverse risk weights within cluster
                        if risk_measure == "vol":
                            cluster_vols = np.sqrt(np.diag(cluster_cov))
                            cluster_weights = 1 / cluster_vols
                            cluster_weights = cluster_weights / np.sum(cluster_weights)
                            cluster_risk = np.sqrt(cluster_weights @ cluster_cov @ cluster_weights)
                        else:  # MV (Variance)
                            cluster_vols = np.diag(cluster_cov)
                            cluster_weights = 1 / cluster_vols
                            cluster_weights = cluster_weights / np.sum(cluster_weights)
                            cluster_risk = cluster_weights @ cluster_cov @ cluster_weights
                        
                        right_risk += cluster_risk
                        right_cluster += cluster_indices
                
                # Calculate allocation proportion
                alpha = 1 - left_risk / (left_risk + right_risk)
            
            # Update weights
            indices_left = [i for i in range(len(assets)) if i in left]
            indices_right = [i for i in range(len(assets)) if i in right]
            weights.iloc[indices_left] *= alpha
            weights.iloc[indices_right] *= (1 - alpha)
    
    # Step 8: Calculate weights within each cluster
    for i in range(1, k+1):
        cluster_indices = clusters[i]
        cluster_cov = cov.iloc[cluster_indices, cluster_indices]
        cluster_returns = returns.iloc[:, cluster_indices]
        
        # Calculate intra-cluster weights
        if risk_measure == "equal":
            # Equal weights
            cluster_weights = np.ones(len(cluster_indices)) / len(cluster_indices)
        else:
            # Inverse risk weights
            if risk_measure == "vol":
                cluster_vols = np.sqrt(np.diag(cluster_cov))
                cluster_weights = 1 / cluster_vols
            else:  # MV (Variance)
                cluster_vols = np.diag(cluster_cov)
                cluster_weights = 1 / cluster_vols
            
            # Normalize to sum to 1
            cluster_weights = cluster_weights / np.sum(cluster_weights)
        
        # Update final weights
        for j, idx in enumerate(cluster_indices):
            weights.iloc[idx] *= cluster_weights[j]
    
    # Format the output
    final_weights = pd.DataFrame(weights, columns=['weights'])
    
    return final_weights


def _optimal_number_of_clusters(dist_matrix, Z, max_k=10):
    """
    Determines the optimal number of clusters using the two-difference gap statistic.
    
    Parameters:
    -----------
    dist_matrix : DataFrame
        Distance matrix
    Z : ndarray
        Linkage matrix from hierarchical clustering
    max_k : int
        Maximum number of clusters to consider
        
    Returns:
    --------
    int
        Optimal number of clusters
    """
    # Calculate gap statistic for different numbers of clusters
    gaps = []
    for k in range(1, max_k + 1):
        clustering_inds = hr.fcluster(Z, k, criterion="maxclust")
        
        # Calculate within-cluster dispersion
        wcd = 0
        for i in range(1, k + 1):
            cluster_indices = [j for j, x in enumerate(clustering_inds) if x == i]
            if len(cluster_indices) > 1:
                cluster_dist = dist_matrix.iloc[cluster_indices, cluster_indices]
                wcd += np.sum(cluster_dist.values) / (2 * len(cluster_indices))
        
        gaps.append(wcd)
    
    # Calculate the two-difference gap statistic
    two_diff = np.diff(np.diff(gaps))
    
    # The optimal k is where the two-difference is maximized
    # Add 3 because we start with k=1 and need to compensate for two diff operations
    optimal_k = np.argmax(two_diff) + 3
    
    return min(optimal_k, max_k)  # Ensure we don't exceed max_k