import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'output/application'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


dt_param_grid: Dict[str, List[int]] = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf_param_grid: Dict[str, List[int]] = {
    'n_estimators': list(range(100, 2000, 100)),
}

gb_param_grid: Dict[str, List[float]] = {
    'n_estimators': list(range(100, 2000, 100)),
    'learning_rate': [0.01, 0.1, 0.2],
}

def fetch_split_ucirepo(
    dataset_id: int, dep_var: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Fetch a UCI dataset and return train/test splits with dependent variable.

    Args:
        dataset_id: UCI repository dataset ID
        dep_var: Name of dependent variable
        test_size: Proportion for test split

    Returns:
        Tuple with X_train, X_test, y_train, y_test, df_train, df_test
    """
    logger.info(f"Fetching dataset id={dataset_id} from UCI repository...")

    dataset = fetch_ucirepo(id=dataset_id)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data.features,
        dataset.data.targets.squeeze(),
        test_size=test_size,
        random_state=42,
    )

    # Ensure targets are 1D arrays
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Create complete dataframes for train and test sets
    df_train = X_train.copy()
    df_train[dep_var] = y_train

    df_test = X_test.copy()
    df_test[dep_var] = y_test

    logger.info(
        f"Dataset loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test, df_train, df_test

def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> Dict[str, float | str]:
    """Compute evaluation metrics for a model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    logger.info(
        f"{model_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
    )

    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
    }

def corr_calc(df: pd.DataFrame, save_path: Optional[Path] = None) -> Figure:
    """Plot correlation heatmap (uses seaborn for heatmap)."""
    logger.info("Creating correlation heatmap...")

    corr_matrix = df.corr()

    fig = plt.figure(figsize=(10, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
    )
    plt.title('Correlación de las variables')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to: {save_path}")

    return fig


def init_models(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]:
    """Train baseline and tuned models and return best estimators."""
    logger.info("Initializing and fitting models (LR, DT, RF, GBM)...")

    y_train_1d = y_train.ravel()

    dt_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    gbm_model = GradientBoostingRegressor(random_state=42)

    dt_grid = GridSearchCV(
        dt_model, dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    gbm_grid = GridSearchCV(
        gbm_model, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_1d)
    logger.info("Linear Regression model fitted")

    dt_grid.fit(X_train, y_train_1d)
    dt_best = dt_grid.best_estimator_
    logger.info("Decision Tree model fitted")

    rf_grid.fit(X_train, y_train_1d)
    rf_best = rf_grid.best_estimator_
    logger.info("Random Forest model fitted")

    gbm_grid.fit(X_train, y_train_1d)
    gbm_best = gbm_grid.best_estimator_
    logger.info("Gradient Boosting model fitted")

    return lr_model, dt_best, rf_best, gbm_best

def evaluate_models(
    y_test: np.ndarray,
    dt_best: DecisionTreeRegressor,
    rf_best: RandomForestRegressor,
    gbm_best: GradientBoostingRegressor,
    lm_model: LinearRegression,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all models and return metrics DataFrame."""
    # Make predictions
    dt_pred = dt_best.predict(X_test)
    rf_pred = rf_best.predict(X_test)
    gbm_pred = gbm_best.predict(X_test)
    lm_pred = lm_model.predict(X_test)

    # Evaluate each model and store in a dataframe
    dt_metrics = evaluate_model(y_test, dt_pred, "Decision Tree")
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
    gbm_metrics = evaluate_model(y_test, gbm_pred, "Gradient Boosting")
    lm_metrics = evaluate_model(y_test, lm_pred, "Linear Regression")

    metrics_df = pd.DataFrame([lm_metrics, dt_metrics, rf_metrics, gbm_metrics])
    logger.info(f"\nModel performance summary:\n{metrics_df.to_string(index=False)}")
    return metrics_df

def permutation_importance_plot(
    best_model, X_test: pd.DataFrame, y_test: np.ndarray, save_path: Optional[Path] = None
) -> Figure:
    """Plot permutation importance using matplotlib."""
    logger.info("Calculating permutation importance plot...")

    result: Any = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
    )

    importances = result.importances_mean
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance_df['Feature'],
        importance_df['Importance'],
        color='skyblue',
        edgecolor='black',
        alpha=0.8,
    )
    ax.set_xlabel('Permutation Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance for Gradient Boosting Model')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Permutation importance plot saved to: {save_path}")

    return fig


def partial_dependence_plot(
    best_model,
    X_test: pd.DataFrame,
    features: List[str],
    feature_titles: List[str],
    save_path: Optional[Path] = None,
) -> Figure:
    """Create partial dependence plots with centered and non-centered views."""
    logger.info("Creating partial dependence plots...")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    for i, feature in enumerate(features):
        for j, centered in enumerate([False, True]):
            PartialDependenceDisplay.from_estimator(
                best_model,
                X_test,
                [feature],
                kind="both",
                subsample=50,
                n_jobs=-1,
                random_state=42,
                centered=centered,
                ax=[axs[i, j]],
            )

            centered_text = " (Centered)" if centered else ""
            axs[i, j].set_title(f'PDP of {feature_titles[i]}{centered_text}')
            axs[i, j].set_ylabel('Partial Dependence')
            axs[i, j].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Partial Dependence Plots for Gradient Boosting Model', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Partial dependence plots saved to: {save_path}")

    return fig

def airfoil_self_noise():
    logger.info("Running Airfoil Self-Noise pipeline...")

    air_X_train, air_X_test, air_y_train, air_y_test, air_df_train, _ = fetch_split_ucirepo(
        291, "scaled-sound-pressure"
    )

    corr_fig = corr_calc(air_df_train, OUTPUT_DIR / 'airfoil_corr.png')
    plt.close(corr_fig)

    air_X_train = air_X_train.drop(columns='attack-angle')
    air_X_test = air_X_test.drop(columns='attack-angle')

    air_lr_best, air_dt_best, air_rf_best, air_gbm_best = init_models(air_X_train, air_y_train)

    metrics_df = evaluate_models(
        air_y_test, air_dt_best, air_rf_best, air_gbm_best, air_lr_best, air_X_test
    )
    metrics_df.to_csv(OUTPUT_DIR / 'airfoil_metrics.csv', index=False)

    pi_fig = permutation_importance_plot(
        air_gbm_best, air_X_test, air_y_test, OUTPUT_DIR / 'airfoil_perm_importance.png'
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        air_gbm_best,
        air_X_test,
        ['frequency', 'suction-side-displacement-thickness'],
        ['Frequency', 'Suction-side-displacement-thickness'],
        OUTPUT_DIR / 'airfoil_pdp.png',
    )
    plt.close(pdp_fig)

def concrete_data():
    logger.info("Running Concrete dataset pipeline...")

    (
        concrete_X_train,
        concrete_X_test,
        concrete_y_train,
        concrete_y_test,
        concrete_df_train,
        _,
    ) = fetch_split_ucirepo(165, "Concrete compressive strength")

    corr_fig = corr_calc(concrete_df_train, OUTPUT_DIR / 'concrete_corr.png')
    plt.close(corr_fig)

    (
        concrete_lr_model,
        concrete_dt_best,
        concrete_rf_best,
        concrete_gbm_best,
    ) = init_models(concrete_X_train, concrete_y_train)

    metrics_df = evaluate_models(
        concrete_y_test,
        concrete_dt_best,
        concrete_rf_best,
        concrete_gbm_best,
        concrete_lr_model,
        concrete_X_test,
    )
    metrics_df.to_csv(OUTPUT_DIR / 'concrete_metrics.csv', index=False)

    pi_fig = permutation_importance_plot(
        concrete_gbm_best,
        concrete_X_test,
        concrete_y_test,
        OUTPUT_DIR / 'concrete_perm_importance.png',
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        concrete_gbm_best,
        concrete_X_test,
        ['Age', 'Cement'],
        ['Age', 'Cement'],
        OUTPUT_DIR / 'concrete_pdp.png',
    )
    plt.close(pdp_fig)

def wine_data():
    logger.info("Running Wine Quality dataset pipeline...")

    wine_X_train, wine_X_test, wine_y_train, wine_y_test, wine_df_train, _ = fetch_split_ucirepo(
        186, "quality"
    )

    corr_fig = corr_calc(wine_df_train, OUTPUT_DIR / 'wine_corr.png')
    plt.close(corr_fig)

    wine_X_train = wine_X_train.drop(columns='free_sulfur_dioxide')
    wine_X_test = wine_X_test.drop(columns='free_sulfur_dioxide')

    wine_lr_model, wine_dt_best, wine_rf_best, wine_gbm_best = init_models(
        wine_X_train, wine_y_train
    )

    metrics_df = evaluate_models(
        wine_y_test,
        wine_dt_best,
        wine_rf_best,
        wine_gbm_best,
        wine_lr_model,
        wine_X_test,
    )
    metrics_df.to_csv(OUTPUT_DIR / 'wine_metrics.csv', index=False)

    pi_fig = permutation_importance_plot(
        wine_rf_best, wine_X_test, wine_y_test, OUTPUT_DIR / 'wine_perm_importance.png'
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        wine_rf_best,
        wine_X_test,
        ['alcohol', 'volatile_acidity'],
        ['Alcohol', 'Volatile Acidity'],
        OUTPUT_DIR / 'wine_pdp.png',
    )
    plt.close(pdp_fig)

def main():
    """Run all dataset pipelines sequentially."""
    logger.info("=" * 60)
    logger.info("Starting Application Pipelines")
    logger.info("=" * 60)

    airfoil_self_noise()
    concrete_data()
    wine_data()

    logger.info("=" * 60)
    logger.info("All pipelines completed. Outputs saved to output/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()