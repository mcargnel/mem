import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'output/chapter_4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_data() -> tuple:
    """
    Generate synthetic regression data with complex relationships.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Generating synthetic data...")
    
    np.random.seed(73815)
    n = 1000

    # Generate all X features at once
    X = np.random.uniform(0, 1, (n, 10))
    X_cols = [f'X_{i+1}' for i in range(10)]

    # Calculate y using a mix of original and redundant features
    y = (
        X[:, 0]**2 + 
        np.sin(X[:, 1]) - 
        np.exp(X[:, 2]) + 
        np.log1p(X[:, 3]) + 
        X[:, 4] * X[:, 5] + 
        X[:, 6] * X[:, 7] * X[:, 8] - 
        np.sqrt(X[:, 9]) + 
        (X[:, 0]**2 + np.random.normal(0, 0.05, n)) + 
        (np.sin(X[:, 1]) + np.random.normal(0, 0.05, n)) - 
        (np.exp(X[:, 2]) + np.random.normal(0, 0.05, n)) + 
        (np.log1p(X[:, 3]) + np.random.normal(0, 0.05, n)) + 
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

def fetch_abalone_data() -> tuple:
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

def fit_gbm_model(X_train, y_train) -> GridSearchCV:
    """
    Train a Gradient Boosting Regressor model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        GridSearchCV: Fitted grid search object
    """
    logger.info("Fitting GBM model with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': list(range(100, 1000, 100)),
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5],
        'loss': ['squared_error', 'absolute_error', 'huber']
    }

    # Calculate total iterations
    n_candidates = len(list(ParameterGrid(param_grid)))
    cv_folds = 5
    total_iterations = n_candidates * cv_folds
    logger.info(f"Grid search: {total_iterations} total iterations ({n_candidates} parameter combinations × {cv_folds} folds)")

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    
    best_rmse = np.sqrt(-grid_search.best_score_)
    logger.info(f"Best model RMSE: {best_rmse:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")

    return grid_search

def calculate_importance(grid_search: GridSearchCV) -> tuple:
    """
    Calculate importance metrics from grid search results.
    
    Args:
        grid_search: Fitted GridSearchCV object
        
    Returns:
        tuple: (models, rmse_values, top_params)
    """
    logger.info("Calculating model importances...")
    
    # Best model from grid search
    rf1 = grid_search.best_estimator_

    # Number of top models to evaluate
    N = 15

    # Get the top N best parameter sets from grid search results
    top_params = pd.DataFrame(grid_search.cv_results_)
    top_params = top_params.sort_values('rank_test_score').iloc[:N]

    # Create dictionaries to store models and their CV errors
    models = {}
    rmse_values = {}

    # Best model
    models['RF1'] = rf1
    rmse_values['RF1'] = np.sqrt(-top_params.iloc[0]['mean_test_score'])

    # Loop through the top parameter sets (2nd to Nth best)
    for i in range(1, N):
        model_name = f'RF{i+1}'
        model = GradientBoostingRegressor(random_state=42, **top_params.iloc[i]['params'])
        models[model_name] = model
        rmse = np.sqrt(-top_params.iloc[i]['mean_test_score'])
        rmse_values[model_name] = rmse
    
    logger.info(f"Calculated importance for {len(models)} models")
    return models, rmse_values, top_params

def display_model_summary(
    models: dict, rmse_values: dict, top_params: pd.DataFrame
) -> pd.DataFrame:
    """
    Display summary of top models with their hyperparameters and RMSE values.
    
    Args:
        models: Dictionary of models
        rmse_values: Dictionary of RMSE values
        top_params: DataFrame with top parameters
        
    Returns:
        pd.DataFrame: Summary DataFrame
    """
    logger.info("Creating model summary...")
    
    # Create a DataFrame to store optimal hyperparameters and RMSE for each model
    model_info = []

    for i, model_name in enumerate(models.keys()):
        # Get the model's hyperparameters
        if i == 0:
            params = top_params.iloc[0]['params']
        else:
            params = top_params.iloc[i]['params']
        
        # Create a dictionary with model name, RMSE, and hyperparameters
        model_data = {
            'Model': model_name,
            'RMSE': rmse_values[model_name],
            **params
        }
        model_info.append(model_data)

    # Convert to DataFrame
    model_summary_df = pd.DataFrame(model_info)
    model_summary_df.columns = [
        'Modelo', 'RMSE', 'Shrinkage', 'Pérdida', 'Profundidad máx', 'Árboles'
    ]

    model_summary_df['RMSE'] = model_summary_df['RMSE'].apply(lambda x: f"{x:.4f}")
    model_summary_df['Shrinkage'] = model_summary_df['Shrinkage'].apply(
        lambda x: f"{x:.2f}"
    )
    
    logger.info(f"Model summary created with {len(model_summary_df)} models")
    return model_summary_df


def calculate_permutation_importance(
    models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, 
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Calculate permutation feature importance for each model.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_test: Test target
        
    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    logger.info("Calculating permutation importance for all models...")
    
    # Calculate permutation importance for each model
    importances = {}
    for idx, (name, model) in enumerate(models.items()):
        logger.info(f"Model {idx+1}/{len(models)}: Calculating permutation importance for {name}...")
        
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances[name] = result.importances_mean
    
    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame(
        {name: imp for name, imp in importances.items()},
        index=X_train.columns
    )
    logger.info("Permutation importance calculation completed")
    
    return importance_df

def plot_feature_importances(importance_df: pd.DataFrame, save_path=None) -> plt.Figure:
    """
    Plot feature importances across models.
    
    Args:
        importance_df: DataFrame with feature importances
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating feature importance plot...")
    
    # Create a better layout for feature importance visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot with matplotlib
    for col in importance_df.columns:
        ax.plot(
            importance_df.index, importance_df[col], 
            marker='o', label=col, linewidth=2, markersize=6
        )

    # Improve title and labels with better font sizes
    ax.set_title('Importancia de las variables comparadas entre modelos', fontsize=14)
    ax.set_ylabel('Importancia', fontsize=12)
    ax.set_xlabel('Variable', fontsize=12)

    # Improve readability of feature names
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title='Models', title_fontsize=11, fontsize=9, loc='best')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")
    
    return fig

def plot_importance_heatmap(importance_df: pd.DataFrame, save_path=None) -> plt.Figure:
    """
    Plot importance rankings as heatmap (uses seaborn for heatmap).
    
    Args:
        importance_df: DataFrame with feature importances
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Creating importance ranking heatmap...")
    
    # Create a dataframe with rankings (1 = most important)
    rank_df = importance_df.rank(ascending=False)

    # Create figure
    fig = plt.figure(figsize=(10, 8))

    # Create heatmap with seaborn (appropriate for heatmaps)
    sns.heatmap(
        rank_df, annot=True, cmap='YlGnBu', fmt='.0f',
        linewidths=0.5, linecolor='white'
    )

    # Improve title and labels
    plt.title('Ranking de importancia de variables entre modelos', fontsize=14)
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Modelos', fontsize=12)

    # Adjust font sizes for axes
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Importance heatmap saved to: {save_path}")
    
    return fig
def train_gbm_for_pdp(X_train: pd.DataFrame, y_train: np.ndarray) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting Regressor for partial dependence plots.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        GradientBoostingRegressor: Trained model
    """
    logger.info("Training GBM model for PDP analysis...")
    
    fitted_gbm = GradientBoostingRegressor(random_state=42, n_estimators=100)
    fitted_gbm.fit(X_train, y_train)
    
    logger.info("GBM model trained successfully")
    return fitted_gbm

def figure_partial_dependence(
    X_train: pd.DataFrame, fitted_gbm: GradientBoostingRegressor, 
    save_path=None, feature_idx: int = 3
) -> plt.Figure:
    """
    Create partial dependence plot for a specific feature.
    
    Args:
        X_train: Training features
        fitted_gbm: Trained GBM model
        save_path: Optional path to save the plot
        feature_idx: Index of feature to plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info(f"Creating partial dependence plot for feature {feature_idx}...")
    
    # Create and display partial dependence plot
    fig, ax = plt.subplots(figsize=(10, 7))
    feature_name = X_train.columns[feature_idx]

    # Using PartialDependenceDisplay for modern implementation
    display = PartialDependenceDisplay.from_estimator(
        fitted_gbm,
        X_train,
        [feature_name], 
        ax=ax,
        grid_resolution=50,
        random_state=42
    )

    # Customize the plot
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Partial Dependence Plot for {feature_name}', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Partial dependence plot saved to: {save_path}")
    
    return fig

def figure_ice_plots(
    X_train: pd.DataFrame, fitted_gbm: GradientBoostingRegressor,
    save_path=None, n_features: int = 3
) -> plt.Figure:
    """
    Create Individual Conditional Expectation (ICE) plots.
    
    Args:
        X_train: Training features
        fitted_gbm: Trained GBM model
        save_path: Optional path to save the plot
        n_features: Feature index to plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info(f"Creating ICE plots for feature {n_features}...")
    
    # Create and display partial dependence plot
    fig, ax = plt.subplots(figsize=(10, 7))
    feature_name = X_train.columns[n_features]

    # Using PartialDependenceDisplay for modern implementation
    display = PartialDependenceDisplay.from_estimator(
        fitted_gbm,
        X_train,
        [feature_name], 
        ax=ax,
        kind='individual',
        grid_resolution=50,
        random_state=42
    )

    # Customize the plot
    ax.grid(True, alpha=0.3)
    ax.set_title(f'ICE Plot for {feature_name}', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ICE plot saved to: {save_path}")
    
    return fig

def figure_centered_ice_plots(
    X_train: pd.DataFrame, fitted_gbm: GradientBoostingRegressor,
    save_path=None, n_features: int = 3
) -> plt.Figure:
    """
    Create Centered Individual Conditional Expectation (ICE) plots.
    
    Args:
        X_train: Training features
        fitted_gbm: Trained GBM model
        save_path: Optional path to save the plot
        n_features: Feature index to plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info(f"Creating centered ICE plots for feature {n_features}...")
    
    feature_name = X_train.columns[n_features]
    # Create centered ICE plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Centered ICE plot
    display_centered = PartialDependenceDisplay.from_estimator(
        fitted_gbm,
        X_train,
        [feature_name], 
        kind='both',  # Shows both ICE and PDP
        centered=True,  # Centers the ICE curves
        ax=ax,
        grid_resolution=50,
        random_state=42,
        ice_lines_kw={'alpha': 0.3, 'color': 'tab:blue'},
        pd_line_kw={'color': 'tab:red', 'linewidth': 2}
    )

    ax.set_title(f'Centered ICE Plots for {feature_name}', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Centered ICE plot saved to: {save_path}")
    
    return fig

def main():
    """
    Main execution function for model interpretation analysis.
    """
    logger.info("="*60)
    logger.info("Starting Model Interpretation Analysis")
    logger.info("="*60)
    
    # Fetch Abalone data
    X_train, X_test, y_train, y_test = fetch_abalone_data()
    
    # 1. Fit GBM model with hyperparameter tuning
    logger.info("\n" + "="*60)
    logger.info("1. Hyperparameter Tuning")
    logger.info("="*60)
    grid_search = fit_gbm_model(X_train, y_train)
    
    # 2. Calculate importances
    logger.info("\n" + "="*60)
    logger.info("2. Model Importance Analysis")
    logger.info("="*60)
    models, rmse_values, top_params = calculate_importance(grid_search)
    
    # Display model summary
    model_summary_df = display_model_summary(models, rmse_values, top_params)
    logger.info(f"\nModel Summary:\n{model_summary_df.to_string()}")
    
    # 3. Calculate permutation importance
    logger.info("\n" + "="*60)
    logger.info("3. Feature Importance Calculation")
    logger.info("="*60)
    importance_df = calculate_permutation_importance(
        models, X_train, X_test, y_test
    )
    
    # 4. Plot feature importances
    logger.info("\n" + "="*60)
    logger.info("4. Creating Visualization Plots")
    logger.info("="*60)
    importance_plot_path = OUTPUT_DIR / 'feature_importances.png'
    importance_fig = plot_feature_importances(
        importance_df, save_path=importance_plot_path
    )
    plt.close(importance_fig)
    
    # Plot importance heatmap
    heatmap_plot_path = OUTPUT_DIR / 'importance_heatmap.png'
    heatmap_fig = plot_importance_heatmap(
        importance_df, save_path=heatmap_plot_path
    )
    plt.close(heatmap_fig)
    
    # 5. Partial Dependence Plots
    logger.info("\n" + "="*60)
    logger.info("5. Partial Dependence Analysis")
    logger.info("="*60)
    fitted_gbm = train_gbm_for_pdp(X_train, y_train)
    
    pdp_plot_path = OUTPUT_DIR / 'partial_dependence_plot.png'
    pdp_fig = figure_partial_dependence(
        X_train, fitted_gbm, save_path=pdp_plot_path, feature_idx=3
    )
    plt.close(pdp_fig)
    
    # 6. ICE Plots
    ice_plot_path = OUTPUT_DIR / 'ice_plot.png'
    ice_fig = figure_ice_plots(
        X_train, fitted_gbm, save_path=ice_plot_path, n_features=3
    )
    plt.close(ice_fig)
    
    # 7. Centered ICE Plots
    centered_ice_plot_path = OUTPUT_DIR / 'centered_ice_plot.png'
    centered_ice_fig = figure_centered_ice_plots(
        X_train, fitted_gbm, save_path=centered_ice_plot_path, n_features=3
    )
    plt.close(centered_ice_fig)
    
    # Save model summary to CSV
    summary_csv_path = OUTPUT_DIR / 'model_summary.csv'
    model_summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Model summary saved to: {summary_csv_path}")
    
    # Save importance dataframe
    importance_csv_path = OUTPUT_DIR / 'feature_importances.csv'
    importance_df.to_csv(importance_csv_path)
    logger.info(f"Feature importances saved to: {importance_csv_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Model Interpretation Analysis Completed Successfully")
    logger.info(f"All outputs saved to: {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()