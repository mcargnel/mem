import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = Path('book/images/capitulo_2')

def sim_data(N):
    np.random.seed(42)
    
    # Parameters
    beta_intercept = np.random.uniform(0, 2)
    true_beta = np.random.uniform(0, 2, 5) # Shape (5,)
    
    # 1. Generate core features (N rows, 4 columns)
    x_non_outlier = np.random.uniform(-5, 5, (N, 4))
    
    # 2. Generate outlier feature
    # Problem source: This resulted in shape (N,), a flat 1D array
    p_outlier = 0.05
    is_outlier = np.random.binomial(1, p_outlier, N)
    x_outlier_raw = (1 - is_outlier) * np.random.randn(N) + is_outlier * (100 * np.random.randn(N))
    
    # CORRECTION 1: Reshape to (N, 1) so it becomes a 2D column vector
    x_outlier = x_outlier_raw.reshape(-1, 1) 
    
    # CORRECTION 2: Use axis=1 to stack columns (features), not rows (observations)
    x_complete = np.concatenate((x_non_outlier, x_outlier), axis=1)
    
    # Generate Y
    noise = np.random.normal(0, 1, N)
    y = beta_intercept + np.dot(x_complete, true_beta) + noise
    
    return x_complete, y

def histogram_outliers(y):
    logger.info("Generating histogram of outliers...")
    fig = plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, color='skyblue', edgecolor='black')
    plt.title("Histograma de y (con outliers)", fontsize=16)
    plt.xlabel("Valor", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    return fig

def fit_lm(x, y):
    logger.info("Fitting linear regression model...")
    lm = LinearRegression()
    lm.fit(x, y)
    return lm

def fit_tree(x, y):
    logger.info("Fitting decision tree model...")
    tree = DecisionTreeRegressor(max_depth=3) # Constrain depth for better visualization
    tree.fit(x, y)
    return tree

def visualize_tree(tree_model, x):
    logger.info("Generating tree visualization...")
    fig = plt.figure(figsize=(15, 8))
    plot_tree(
        tree_model,
        feature_names=[f"X{i+1}" for i in range(x.shape[1])],
        filled=True, 
        rounded=True, 
        impurity=False,
        fontsize=11,
        precision=2,
        proportion=True
    )
    plt.title("Estructura del Árbol de Regresión", fontsize=20)
    return fig


def save_results(fig, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot to: {save_path}")

def main():
    logger.info("Starting analysis...")
    x, y = sim_data(100)
    lm_model = fit_lm(x, y)
    tree_model = fit_tree(x, y)
    
    fig_tree = visualize_tree(tree_model, x)
    fig_histogram_outliers = histogram_outliers(y)
    
    save_results(fig_tree, OUTPUT_DIR / 'tree_outliers.pdf')
    save_results(fig_histogram_outliers, OUTPUT_DIR / 'histogram_outliers.pdf')    
    logger.info("Analysis complete.")


if __name__ == '__main__':
    main()