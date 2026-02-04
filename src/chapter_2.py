import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'output/chapter_2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_abalone_data():
    """
    Fetch the Abalone dataset from UCI repository and prepare it for regression tasks.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Fetching Abalone dataset from UCI repository...")
    
    # Fetch the abalone dataset
    abalone = fetch_ucirepo(id=1)

    # Prepare features and targets: drop the 'Sex' column and ensure y is 1D.
    X = abalone.data.features.drop('Sex', axis=1)
    # Convert y to a 1D array (using .values.ravel() or .squeeze())
    y = (
        abalone.data.targets.values.ravel() 
        if hasattr(abalone.data.targets, 'values') 
        else abalone.data.targets.squeeze()
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Abalone dataset loaded - Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def fit_regression_tree(X_train, y_train) -> DecisionTreeRegressor:
    """
    Train a decision tree regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        DecisionTreeRegressor: Trained regression tree model
    """
    logger.info("Training decision tree regressor (max_depth=3)...")
    
    regression_tree = DecisionTreeRegressor(random_state=42, max_depth=3)
    regression_tree.fit(X_train, y_train)
    
    # Log model performance on training data
    train_score = regression_tree.score(X_train, y_train)
    logger.info(f"Training R² score: {train_score:.4f}")
    logger.info(f"Tree depth: {regression_tree.get_depth()}")
    logger.info(f"Number of leaves: {regression_tree.get_n_leaves()}")
    
    return regression_tree

def plot_tree_model(regression_tree, X_train, save_path: Path = None) -> plt.Figure:
    """
    Visualize the regression tree structure.
    
    Args:
        regression_tree: Trained decision tree model
        X_train: Training features (for column names)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.pyplot: Plot object
    """
    logger.info("Creating tree visualization...")
    
    plt.figure(figsize=(15, 8))
    plot_tree(
        regression_tree, 
        feature_names=X_train.columns.tolist(),
        filled=True, 
        rounded=True, 
        impurity=False,
        fontsize=11,
        precision=2,
        proportion=True
    )
    plt.title("Árbol de regresión", fontsize=20)
    
    if save_path:
        save_path_pdf = str(save_path).replace('.png', '.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        logger.info(f"Tree plot saved to: {save_path_pdf}")
    
    return plt

def calc_tree_alpha(X_train, X_test, y_train, y_test, save_plot_path: Path = None) -> tuple:
    """
    Calculate and plot MSE for different tree complexities.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        save_plot_path: Optional path to save the MSE plot
        
    Returns:
        tuple: (results_df, plt) - DataFrame with results and plot object
    """
    logger.info("Evaluating tree complexity vs. MSE...")
    
    # Define range of tree complexities to test
    max_leaf_nodes_list = [i for i in range(2, 250, 10)]
    logger.info(f"Testing {len(max_leaf_nodes_list)} different complexity levels")

    # Train models with different complexities and calculate MSE
    test_mse_scores = []
    for idx, max_leaf_nodes in enumerate(max_leaf_nodes_list):
        reg = DecisionTreeRegressor(random_state=42, max_leaf_nodes=max_leaf_nodes)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        test_mse_scores.append(mse)
        
        if (idx + 1) % 5 == 0:  # Log progress every 5 iterations
            logger.debug(f"Evaluated {idx + 1}/{len(max_leaf_nodes_list)} models")
    
    # Find optimal complexity
    optimal_idx = np.argmin(test_mse_scores)
    optimal_nodes = max_leaf_nodes_list[optimal_idx]
    optimal_mse = test_mse_scores[optimal_idx]
    logger.info(f"Optimal max_leaf_nodes: {optimal_nodes} (MSE: {optimal_mse:.4f})")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'max_leaf_nodes': max_leaf_nodes_list,
        'Error Cuadrático Medio': test_mse_scores
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(max_leaf_nodes_list, test_mse_scores, marker='o', linewidth=2)
    plt.axvline(x=optimal_nodes, color='r', linestyle='--', alpha=0.7, 
                label=f'Óptimo: {optimal_nodes} nodos')
    plt.xlabel('Número Máximo de Nodos Hoja', fontsize=12)
    plt.ylabel('Error Cuadrático Medio', fontsize=12)
    plt.title('Error Cuadrático Medio vs. Complejidad del Árbol (Máx. Nodos Hoja)', 
              fontsize=14)
    plt.legend()
    
    if save_plot_path:
        save_plot_path_pdf = str(save_plot_path).replace('.png', '.pdf')
        plt.savefig(save_plot_path_pdf, dpi=300, bbox_inches='tight')
        logger.info(f"MSE plot saved to: {save_plot_path_pdf}")
    
    return results_df, plt

def main():
    """
    Main execution function for regression tree analysis.
    """
    logger.info("="*60)
    logger.info("Starting Regression Tree Analysis")
    logger.info("="*60)
    
    # Generate data
    X_train, X_test, y_train, y_test = fetch_abalone_data()
    
    # Train regression tree
    regression_tree = fit_regression_tree(X_train, y_train)
    
    # Evaluate on test set
    y_pred_test = regression_tree.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = regression_tree.score(X_test, y_test)
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test R² score: {test_r2:.4f}")
    
    # Visualize tree and save
    tree_plot_path = OUTPUT_DIR / 'regression_tree_structure.pdf'
    tree_plot = plot_tree_model(regression_tree, X_train, save_path=tree_plot_path)
    
    # Analyze complexity vs performance and save
    mse_plot_path = OUTPUT_DIR / 'mse_vs_complexity.pdf'
    results_df, mse_plot = calc_tree_alpha(
        X_train, X_test, y_train, y_test, 
        save_plot_path=mse_plot_path
    )
    
    # Save results DataFrame
    results_tex_path = OUTPUT_DIR / 'tree_complexity_results.tex'
    results_df.to_latex(results_tex_path, index=False)
    logger.info(f"Results DataFrame saved to: {results_tex_path}")
    logger.info(f"Results DataFrame preview:\n{results_df.head()}")
    logger.info(f"Results summary:\n{results_df.describe()}")
    
    logger.info("="*60)
    logger.info("Regression Tree Analysis Completed Successfully")
    logger.info(f"All outputs saved to: {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()