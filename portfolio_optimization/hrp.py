
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform

def hrp_portfolio(returns, correlation_method='pearson', linkage_method='single', risk_measure='MV', risk_free_rate=0):
    """
    Implements the Hierarchical Risk Parity portfolio optimization algorithm.
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns DataFrame with assets as columns and observations as rows
    correlation_method : str
        Method used to calculate correlation matrix ('pearson', 'spearman', 'kendall')
    linkage_method : str
        Linkage method for hierarchical clustering
    risk_measure : str
        Risk measure used ('MV' for variance, 'vol' for standard deviation)
    risk_free_rate : float
        Risk-free rate used in risk calculations
        
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
    clusters = hr.linkage(dist_condensed, method=linkage_method)
    
    # Step 4: Quasi-diagonalization (seriation)
    # Reorder assets based on hierarchical clustering
    sort_order = hr.leaves_list(clusters)
    assets = returns.columns.tolist()
    asset_order = [assets[i] for i in sort_order]
    
    # Step 5: Recursive bisection
    weights = pd.Series(1.0, index=assets)  # set initial weights to 1
    items = [sort_order]
    
    # Calculate covariance matrix
    cov = returns.cov()
    
    while len(items) > 0:
        # Split clusters until all clusters have been bisected
        items = [
            i[j:k] for i in items 
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        
        # For each pair of clusters
        for i in range(0, len(items), 2):
            if i + 1 >= len(items):
                continue
                
            left_cluster = items[i]
            right_cluster = items[i + 1]
            
            # Left cluster
            left_cov = cov.iloc[left_cluster, left_cluster]
            left_returns = returns.iloc[:, left_cluster]
            
            # Simple inverse volatility weights for elements in left cluster
            if risk_measure == "vol":
                left_vols = np.sqrt(np.diag(left_cov))
                left_weights = 1 / left_vols
                left_weights = left_weights / np.sum(left_weights)
                left_risk = np.sqrt(left_weights @ left_cov @ left_weights)
            else:  # MV (Variance)
                left_vols = np.diag(left_cov)
                left_weights = 1 / left_vols
                left_weights = left_weights / np.sum(left_weights)
                left_risk = left_weights @ left_cov @ left_weights
                
            # Right cluster
            right_cov = cov.iloc[right_cluster, right_cluster]
            right_returns = returns.iloc[:, right_cluster]
            
            # Simple inverse volatility weights for elements in right cluster
            if risk_measure == "vol":
                right_vols = np.sqrt(np.diag(right_cov))
                right_weights = 1 / right_vols
                right_weights = right_weights / np.sum(right_weights)
                right_risk = np.sqrt(right_weights @ right_cov @ right_weights)
            else:  # MV (Variance)
                right_vols = np.diag(right_cov)
                right_weights = 1 / right_vols
                right_weights = right_weights / np.sum(right_weights)
                right_risk = right_weights @ right_cov @ right_weights
                
            # Allocate weights between the two clusters
            alpha = 1 - left_risk / (left_risk + right_risk)
            
            # Update weights
            weights.iloc[left_cluster] *= alpha
            weights.iloc[right_cluster] *= (1 - alpha)
    
    # Format the output
    final_weights = pd.DataFrame(weights, columns=['weights'])
    
    return final_weights