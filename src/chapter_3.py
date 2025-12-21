import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'output/chapter_3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def simulate_data():
    """Generate synthetic regression data with complex relationships.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Generating synthetic data...")
    
    np.random.seed(73815)
    n = 1000

    # Generate all X features at once
    X = np.random.uniform(0, 1, (n, 10))
    X_cols = [f'X_{i+1}' for i in range(10)]

    # Calculate y using vectorized operations where possible
    y = (
        X[:, 0]**2 + 
        np.sin(X[:, 1]) - 
        np.exp(X[:, 2]) + 
        np.log1p(X[:, 3]) + 
        X[:, 4] * X[:, 5] + 
        X[:, 6] * X[:, 7] * X[:, 8] - 
        np.sqrt(X[:, 9]) + 
        X[:, 0] * X[:, 1] - 
        X[:, 2] * X[:, 3] + 
        X[:, 4] * X[:, 5] * X[:, 6] +
        np.random.normal(0, 0.1, n)
    )

    # Create DataFrame with all columns
    df = pd.DataFrame(X, columns=X_cols)
    df['y'] = y

    X = df.drop(columns='y')
    y = df['y']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Generated {n} samples - Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

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

def get_predictions_for_different_M(X_train, y_train, X_test_point, max_trees=100):
    """
    Trains a RandomForestRegressor with an increasing number of trees and
    records the prediction for a given test point.
    
    Parameters:
        X_train (array-like): Training features, shape (n_samples, n_features).
        y_train (array-like): Training targets, shape (n_samples,).
        X_test_point (array-like): A single test point with multiple features,
                                   shape (n_features,) or (1, n_features).
        max_trees (int): The maximum number of trees to try.
        
    Returns:
        tree_numbers (np.array): Array of tree counts used.
        predictions (np.array): Array of predictions corresponding to each model.
    """
    predictions = []
    tree_numbers = []
    
    # If X_train is a DataFrame, ensure X_test_point is also a DataFrame with the same columns.
    if hasattr(X_train, 'columns'):
        # If X_test_point is a Series, convert it to a DataFrame with one row.
        if isinstance(X_test_point, pd.Series):
            X_test_point = X_test_point.to_frame().T
        # If X_test_point is a NumPy array, convert it to a DataFrame.
        elif isinstance(X_test_point, np.ndarray):
            X_test_point = pd.DataFrame(X_test_point, columns=X_train.columns)
    else:
        # Otherwise, ensure X_test_point is 2D.
        X_test_point = np.atleast_2d(X_test_point)
    
    # Loop over a range of tree counts (stepping by 10)
    for n_trees in tqdm.tqdm(range(1, max_trees + 1, 10), desc="Calculando predicciones"):
        # Create and train the Random Forest model with n_trees trees
        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict for the test point
        pred = rf.predict(X_test_point)[0]
        
        predictions.append(pred)
        tree_numbers.append(n_trees)
    
    return np.array(tree_numbers), np.array(predictions)


def calcul_rf_predictions(X_train, X_test, y_train):    
    """
    Calculate Random Forest predictions for a single test point
    as the number of trees increases.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        
    Returns:
        tuple: (trees, preds, running_std)
    """
    logger.info("Calculating RF predictions for different tree counts...")
    
    # Select a single test point.
    # Using iloc with [0:1] ensures the result is a DataFrame (preserving column names).
    test_point = X_test.iloc[0:1]

    # Get predictions for different numbers of trees
    trees, preds = get_predictions_for_different_M(
        X_train, y_train, test_point, max_trees=1000
    )

    # Calculate the running standard deviation of the predictions
    running_std = []
    for i in range(1, len(preds)):
        running_std.append(np.std(preds[:i]))
    
    logger.info(f"Calculated predictions for {len(trees)} different tree counts")
    return trees, preds, running_std



def plot_rf_predictions(trees, preds, running_std, save_path=None):    
    """
    Plot Random Forest predictions and their running standard deviation
    as the number of trees increases.
    
    Args:
        trees: Array of tree counts
        preds: Array of predictions
        running_std: Array of running standard deviations
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating RF predictions plot...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot: Prediction vs. Number of Trees
    ax1.plot(trees, preds, color='blue', linewidth=2)
    ax1.set_xlabel('Número de Árboles (M)', fontsize=11)
    ax1.set_ylabel('Valor Predicho', fontsize=11)
    ax1.set_title('Convergencia de las Predicciones', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot: Running Standard Deviation vs. Number of Trees
    ax2.plot(trees[1:], running_std, color='green', linewidth=2)
    ax2.set_xlabel('Número de Árboles (M)', fontsize=11)
    ax2.set_ylabel('Desviación Estándar', fontsize=11)
    ax2.set_title('Variabilidad en las Predicciones', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RF predictions plot saved to: {save_path}")
    
    return fig


def calc_loss_functions():
    """
    Calculate and plot different loss functions: MSE, MAE, and Huber Loss.
    """
        
    #| label: calc-loss-functions
    # Generate sample predictions and true values
    y_true = 0
    predictions = np.linspace(-4, 4, 1000)

    # Calculate different loss functions
    mse_loss = 0.5 * (predictions - y_true)**2
    mae_loss = np.abs(predictions - y_true)

    # Calculate Huber loss for different deltas
    delta_values = [1.0, 0.5, 0.25]
    huber_losses = []
    for delta in delta_values:
        huber_loss = np.where(np.abs(predictions - y_true) <= delta,
                            0.5 * (predictions - y_true)**2,
                            delta * np.abs(predictions - y_true) - 0.5 * delta**2)
        huber_losses.append(huber_loss)
    return predictions, mse_loss, mae_loss, delta_values, huber_losses

def plot_loss_functions(predictions, mse_loss, mae_loss, delta_values, huber_losses, save_path=None):
    """
    Plot different loss functions: MSE, MAE, and Huber Loss.
    
    Args:
        predictions: Array of prediction values
        mse_loss: Array of MSE loss values
        mae_loss: Array of MAE loss values
        delta_values: List of delta values for Huber loss
        huber_losses: List of Huber loss arrays
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating loss functions plot...")
    
    fig = plt.figure(figsize=(12, 6))

    # Plot the loss functions
    plt.plot(predictions, mse_loss, label='MSE Loss', linewidth=2)
    plt.plot(predictions, mae_loss, label='MAE Loss', linewidth=2)
    
    # Plot all Huber losses
    linestyles = ['-', '--', '-.']
    colors = ['green', 'darkgreen', 'lightgreen']
    for (delta, huber_loss), ls, color in zip(
        zip(delta_values, huber_losses), linestyles, colors
    ):
        plt.plot(
            predictions, huber_loss, 
            label=f'Huber Loss (δ={delta})', 
            linewidth=2, linestyle=ls, color=color
        )

    # Add vertical line at y_true = 0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Customize the plot
    plt.xlabel('Predicción f(x)', fontsize=11)
    plt.ylabel('Pérdida L(y, f(x))', fontsize=11)
    plt.title('Comparación de Funciones de Pérdida', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss functions plot saved to: {save_path}")
    
    return fig


def calc_gbm_depth_effect(X_train, X_test, y_train, y_test):
    """
    Demonstrate the effect of tree depth in Gradient Boosting Machines (GBM).
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        tuple: (depths, mse_scores)
    """
    logger.info("Calculating GBM depth effect...")
    
    np.random.seed(42)

    # Test different max_depths
    depths = list(range(1, 20))
    n_estimators = 100
    learning_rate = 0.1

    mse_scores = []
    for d in tqdm.tqdm(depths, desc="Calculating errors for different depths"):
        gbm = GradientBoostingRegressor(
            max_depth=d,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        gbm.fit(X_train, y_train)
        y_pred = gbm.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
    
    optimal_depth = depths[np.argmin(mse_scores)]
    logger.info(f"Optimal depth: {optimal_depth} (MSE: {min(mse_scores):.4f})")
    
    return depths, mse_scores

def plot_gbm_depth_effect(depths, mse_scores, save_path=None):
    """
    Plot the effect of tree depth in Gradient Boosting Machines (GBM).
    
    Args:
        depths: List of depth values
        mse_scores: List of MSE scores
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating GBM depth effect plot...")
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(depths, mse_scores, marker='o', linewidth=2, markersize=6)
    
    # Mark optimal depth
    optimal_idx = np.argmin(mse_scores)
    plt.scatter(
        depths[optimal_idx], mse_scores[optimal_idx], 
        color='red', s=100, zorder=5, label='Óptimo'
    )
    
    plt.xlabel('Profundidad máxima del árbol', fontsize=11)
    plt.ylabel('Error cuadrático medio', fontsize=11)
    plt.title('Error vs Profundidad del árbol en GBM', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"GBM depth effect plot saved to: {save_path}")
    
    return fig

def calc_shrinkage_effect(X_train, X_test, y_train, y_test):
    """
    Demonstrate the effect of shrinkage (learning rate) in Gradient Boosting Machines (GBM).
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        tuple: (shrinkage_values, n_estimators, train_scores, test_scores)
    """
    logger.info("Calculating shrinkage effect...")
    
    # Define different shrinkage values to compare
    shrinkage_values = [1.0, 0.5, 0.1, 0.05]
    n_estimators = 250
    max_depth = 3

    # Train models with different shrinkage values
    train_scores = []
    test_scores = []

    for shrinkage in shrinkage_values:
        logger.info(f"Training model with shrinkage={shrinkage}...")
        gbm = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=shrinkage,
            random_state=42
        )
        
        # Save error at each iteration
        train_score = []
        test_score = []
        
        for i in tqdm.tqdm(
            range(1, n_estimators + 1, 10), 
            desc=f"Training model with shrinkage {shrinkage}"
        ):
            gbm.set_params(n_estimators=i)
            gbm.fit(X_train, y_train)
            train_score.append(mean_squared_error(y_train, gbm.predict(X_train)))
            test_score.append(mean_squared_error(y_test, gbm.predict(X_test)))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    logger.info(f"Completed shrinkage effect analysis for {len(shrinkage_values)} values")
    return shrinkage_values, n_estimators, train_scores, test_scores

def reshape_shrinkage_data(shrinkage_values, n_estimators, train_scores, test_scores):
    """
    Reshape the shrinkage effect data for plotting.
    
    Args:
        shrinkage_values: List of shrinkage values
        n_estimators: Maximum number of estimators
        train_scores: List of training scores
        test_scores: List of test scores
        
    Returns:
        pd.DataFrame: Reshaped data
    """
    logger.info("Reshaping shrinkage data for plotting...")
    
    iterations = range(1, n_estimators + 1, 10)

    # Create long format DataFrame for train scores
    train_data = []
    for i, shrinkage in enumerate(shrinkage_values):
        for iter_num, score in zip(iterations, train_scores[i]):
            train_data.append({
                'n_estimators': iter_num,
                'shrinkage': shrinkage,
                'mse': score,
                'set': 'train'
            })

    # Create long format DataFrame for test scores        
    test_data = []
    for i, shrinkage in enumerate(shrinkage_values):
        for iter_num, score in zip(iterations, test_scores[i]):
            test_data.append({
                'n_estimators': iter_num,
                'shrinkage': shrinkage,
                'mse': score,
                'set': 'test'
            })
    return pd.DataFrame(train_data + test_data)

def plot_shrinkage_effect(all_data, shrinkage_values, save_path=None):
    """
    Plot the effect of shrinkage (learning rate) in Gradient Boosting Machines (GBM).
    
    Args:
        all_data: DataFrame with combined train and test data
        shrinkage_values: List of shrinkage values
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating shrinkage effect plot...")

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot train data
    for shrinkage in shrinkage_values:
        subset = all_data[all_data['shrinkage'] == shrinkage]
        train = subset[subset['set'] == 'train']
        ax1.plot(train['n_estimators'], train['mse'],
                label=f'λ={shrinkage}')
        # Add point at minimum
        min_score = train['mse'].min()
        min_idx = train.loc[train['mse'].idxmin(), 'n_estimators']
        ax1.scatter(min_idx, min_score, color='red', zorder=5)

    ax1.set_xlabel('Número de iteraciones')
    ax1.set_ylabel('Error cuadrático medio')
    ax1.set_title('Error de Entrenamiento')
    ax1.legend()

    # Plot test data
    for shrinkage in shrinkage_values:
        subset = all_data[all_data['shrinkage'] == shrinkage]
        test = subset[subset['set'] == 'test']
        ax2.plot(test['n_estimators'], test['mse'],
                label=f'λ={shrinkage}')
        # Add point at minimum
        min_score = test['mse'].min()
        min_idx = test.loc[test['mse'].idxmin(), 'n_estimators']
        ax2.scatter(min_idx, min_score, color='red', zorder=5)

    ax2.set_xlabel('Número de iteraciones', fontsize=11)
    ax2.set_ylabel('Error cuadrático medio', fontsize=11)
    ax2.set_title('Error de Test', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Shrinkage effect plot saved to: {save_path}")
    
    return fig

def main():
    """
    Main execution function for ensemble methods analysis.
    """
    logger.info("="*60)
    logger.info("Starting Ensemble Methods Analysis")
    logger.info("="*60)
    
    # Fetch Abalone data
    X_train, X_test, y_train, y_test = fetch_abalone_data()
    
    # 1. Random Forest predictions analysis
    logger.info("\n" + "="*60)
    logger.info("1. Random Forest Convergence Analysis")
    logger.info("="*60)
    trees, preds, running_std = calcul_rf_predictions(X_train, X_test, y_train)
    rf_plot_path = OUTPUT_DIR / 'rf_predictions_convergence.png'
    rf_fig = plot_rf_predictions(trees, preds, running_std, save_path=rf_plot_path)
    plt.close(rf_fig)
    
    # 2. Loss functions comparison
    logger.info("\n" + "="*60)
    logger.info("2. Loss Functions Comparison")
    logger.info("="*60)
    predictions, mse_loss, mae_loss, delta_values, huber_losses = calc_loss_functions()
    loss_plot_path = OUTPUT_DIR / 'loss_functions_comparison.png'
    loss_fig = plot_loss_functions(
        predictions, mse_loss, mae_loss, delta_values, huber_losses, 
        save_path=loss_plot_path
    )
    plt.close(loss_fig)
    
    # 3. GBM depth effect
    logger.info("\n" + "="*60)
    logger.info("3. GBM Depth Effect Analysis")
    logger.info("="*60)
    depths, depth_mse_scores = calc_gbm_depth_effect(X_train, X_test, y_train, y_test)
    depth_plot_path = OUTPUT_DIR / 'gbm_depth_effect.png'
    depth_fig = plot_gbm_depth_effect(depths, depth_mse_scores, save_path=depth_plot_path)
    plt.close(depth_fig)
    
    # 4. Shrinkage effect
    logger.info("\n" + "="*60)
    logger.info("4. GBM Shrinkage Effect Analysis")
    logger.info("="*60)
    shrinkage_values, n_estimators, train_scores, test_scores = calc_shrinkage_effect(
        X_train, X_test, y_train, y_test
    )
    shrinkage_data = reshape_shrinkage_data(
        shrinkage_values, n_estimators, train_scores, test_scores
    )
    shrinkage_plot_path = OUTPUT_DIR / 'gbm_shrinkage_effect.png'
    shrinkage_fig = plot_shrinkage_effect(
        shrinkage_data, shrinkage_values, save_path=shrinkage_plot_path
    )
    plt.close(shrinkage_fig)
    
    # Save shrinkage results
    shrinkage_csv_path = OUTPUT_DIR / 'gbm_shrinkage_results.csv'
    shrinkage_data.to_csv(shrinkage_csv_path, index=False)
    logger.info(f"Shrinkage results saved to: {shrinkage_csv_path}")

    logger.info("\n" + "="*60)
    logger.info("Ensemble Methods Analysis Completed Successfully")
    logger.info(f"All outputs saved to: {OUTPUT_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    main()