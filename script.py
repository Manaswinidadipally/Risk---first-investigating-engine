# main_hrp.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.spatial.distance as ssd

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Step 1: Load & Clean Data
# ---------------------------------------------------
def load_data(filepath):
    """Load and clean S&P500 price data."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Drop columns with more than 30% missing values
    threshold = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=threshold)
    
    # Fill remaining missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def compute_returns(price_df):
    """Compute daily returns from price data."""
    return price_df.pct_change().dropna()

# ---------------------------------------------------
# Step 2: Apply PCA
# ---------------------------------------------------
def apply_pca(returns, n_components=0.9):
    """
    Apply PCA on returns. Keeps enough components to explain 90% of variance by default.
    """
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)

    pca = PCA(n_components=n_components)
    reduced_returns = pca.fit_transform(scaled_returns)

    reduced_df = pd.DataFrame(
        reduced_returns,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(reduced_returns.shape[1])]
    )
    
    print(f"PCA reduced returns shape: {reduced_df.shape}")
    return reduced_df, pca.explained_variance_ratio_

# ---------------------------------------------------
# Step 3: Hierarchical Clustering
# ---------------------------------------------------
def hierarchical_clustering(pca_df, method="ward", plot=True):
    """
    Perform hierarchical clustering on PCA-transformed data.
    """
    dist_matrix = ssd.pdist(pca_df.values, metric="euclidean")
    linkage_matrix = linkage(dist_matrix, method=method)

    if plot:
        plt.figure(figsize=(12, 4))
        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90.,
            leaf_font_size=10.
        )
        plt.title("Hierarchical Clustering Dendrogram (truncated)")
        plt.xlabel("Cluster size")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    return linkage_matrix

def get_quasi_diag(link):
    """Sort clustered items by distance using the linkage matrix."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    num_items = link[-1, 3]  # total number of original items

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()

    return sort_ix.tolist()

def get_cluster_var(cov, cluster_items):
    """Compute the variance of a cluster."""
    cov_ = cov.loc[cluster_items, cluster_items]
    w_ = 1. / np.diag(cov_)
    w_ /= w_.sum()
    return np.dot(np.dot(w_, cov_), w_.T)

def hrp_allocation(cov, ordered_assets):
    """Perform recursive bisection to allocate weights."""
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = int(len(cluster) / 2)
            c1 = cluster[:split]
            c2 = cluster[split:]

            var1 = get_cluster_var(cov, c1)
            var2 = get_cluster_var(cov, c2)
            alpha = 1 - var1 / (var1 + var2)

            weights[c1] *= alpha
            weights[c2] *= (1 - alpha)

            new_clusters += [c1, c2]
        clusters = new_clusters

    return weights


def backtest_portfolio(returns, weights):
    """Simulate portfolio performance using static weights."""
    # Align columns (in case assets were dropped in PCA/cov step)
    returns = returns[weights.index]

    # Daily portfolio returns
    port_returns = (returns * weights).sum(axis=1)

    # Cumulative returns
    cum_returns = (1 + port_returns).cumprod()

    # Performance metrics
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = (cum_returns.iloc[-1]) ** (252 / len(cum_returns)) - 1
    annualized_vol = port_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol
    max_drawdown = ((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max()

    print("\n📊 Performance Metrics:")
    print(f"Cumulative Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot
    plt.figure(figsize=(10, 5))
    cum_returns.plot(title="HRP Portfolio Cumulative Returns")
    plt.ylabel("Portfolio Value (normalized)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return cum_returns





def equal_weight_allocation(returns):
    """Assign equal weights to all assets."""
    n_assets = returns.shape[1]
    return pd.Series(1 / n_assets, index=returns.columns)

def markowitz_allocation(returns, long_only=True, weight_clip=0.05):
    """Compute mean-variance (Markowitz) weights with optional clipping."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    inv_cov = np.linalg.pinv(cov_matrix.values)

    raw_weights = inv_cov @ mean_returns.values
    norm_weights = raw_weights / np.sum(raw_weights)

    weights = pd.Series(norm_weights, index=returns.columns)

    if long_only:
        weights = weights.clip(lower=0)
        weights /= weights.sum()  # re-normalize
    else:
        # Optional: clip extremes if needed
        weights = weights.clip(lower=-weight_clip, upper=weight_clip)
        weights /= weights.sum()

    return weights




# ---------------------------------------------------
# Main execution block
# ---------------------------------------------------
if __name__ == "__main__":
    # 1. Load and clean data
    df = load_data("SP500_2022_2024.csv")
    returns = compute_returns(df)
    returns = returns["2022-01-01":"2024-12-31"]

    print("Data loaded and returns computed:", returns.shape)

    # 2. Apply PCA
    pca_returns, pca_variance = apply_pca(returns)
    print("Cumulative explained variance:", np.cumsum(pca_variance))

    # 3. Run hierarchical clustering
    linkage_matrix = hierarchical_clustering(returns.T)
    cov_matrix = returns.cov()
    assets = returns.columns.tolist()
    sorted_indices = get_quasi_diag(linkage_matrix)
    ordered_assets = [assets[i] for i in sorted_indices]
    hrp_weights = hrp_allocation(cov_matrix, ordered_assets)

    print("\nTop 10 HRP Portfolio Weights:")
    print(hrp_weights.sort_values(ascending=False).head(10))

    # 4. Backtest HRP Portfolio
    hrp_cum = backtest_portfolio(returns, hrp_weights)
    print(f"Return period: {returns.index[0].date()} to {returns.index[-1].date()}")

    # 5. Equal Weight Portfolio
    ew_weights = equal_weight_allocation(returns)
    print("\nTop 10 Equal Weight Portfolio Weights:")
    print(ew_weights.sort_values(ascending=False).head(10))
    ew_cum = backtest_portfolio(returns, ew_weights)
    # 6. Markowitz Portfolio
    mv_weights = markowitz_allocation(returns)
    print("\nTop 10 Markowitz Portfolio Weights:")
    print(mv_weights.sort_values(ascending=False).head(10))
    mv_cum = backtest_portfolio(returns, mv_weights)

        # 7. Plot all three portfolios
    plt.figure(figsize=(10, 6))
    hrp_cum.plot(label="HRP")
    ew_cum.plot(label="Equal Weight")
    mv_cum.plot(label="Markowitz")
    plt.title("Portfolio Cumulative Returns (2022–2024)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


