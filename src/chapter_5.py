import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
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
OUTPUT_DIR = Path(__file__).parent.parent / 'output/chapter_5'
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
) -> Dict[str, Union[float, str]]:
    """Compute evaluation metrics for a model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

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
    X_train: pd.DataFrame, y_train: np.ndarray, focus_model: Optional[str] = None
) -> Tuple[
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    List[Tuple[Dict[str, Any], BaseEstimator]]
]:
    """
    Train baseline and tuned models.
    
    If 'focus_model' is provided ('gbm' or 'rf'), it also extracts and returns the 
    top 5 hyperparameter configurations for that model type.
    
    Returns:
        lr_model, dt_best, rf_best, gbm_best, top_5_models
    """
    logger.info(f"Initializing and fitting models (LR, DT, RF, GBM). Focus: {focus_model}")

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
    
    top_5_models: List[Tuple[Dict[str, Any], BaseEstimator]] = []

    if focus_model:
        target_grid = None
        ModelClass = None
        
        if focus_model.lower() == 'gbm':
            target_grid = gbm_grid
            ModelClass = GradientBoostingRegressor
        elif focus_model.lower() == 'rf':
            target_grid = rf_grid
            ModelClass = RandomForestRegressor
            
        if target_grid and ModelClass:
            # cv_results_ is a dict, convert to df for easier sorting
            results_df = pd.DataFrame(target_grid.cv_results_)
            # Sort by rank_test_score (1 is best)
            top_5_results = results_df.nsmallest(5, 'rank_test_score')
            
            logger.info(f"Extracting top 5 configurations for {focus_model}...")
            
            for rank, (_, row) in enumerate(top_5_results.iterrows(), 1):
                params = row['params']
                logger.info(f"Rank {rank}: {params}")
                
                # Re-instantiate and fit on full train data
                model = ModelClass(**params, random_state=42)
                model.fit(X_train, y_train_1d)
                top_5_models.append((params, model))

    return lr_model, dt_best, rf_best, gbm_best, top_5_models

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

def evaluate_top_5(
    top_5_models: List[Tuple[Dict[str, Any], BaseEstimator]],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    base_name: str
) -> pd.DataFrame:
    """Evaluate list of top 5 models."""
    metrics_list = []
    for i, (params, model) in enumerate(top_5_models, 1):
        pred = model.predict(X_test)
        metrics = evaluate_model(y_test, pred, f"{base_name} Top {i}")
        # Add params as string for reference
        metrics['Params'] = str(params)
        metrics['Rank'] = i
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    return df

def get_top_5_importances(
    top_5_models: List[Tuple[Dict[str, Any], BaseEstimator]],
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract and aggregate feature importances from top 5 models."""
    df_combined = pd.DataFrame({'Feature': feature_names})
    
    for i, (_, model) in enumerate(top_5_models, 1):
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            df_combined[f'Top {i}'] = imps
    
    # Calculate Mean Importance for sorting (optional but good for display)
    df_combined['Mean'] = df_combined.filter(like='Top').mean(axis=1)
    df_combined = df_combined.sort_values('Mean', ascending=False)
    
    # Drop Mean column if you only want the raw model columns, but keeping it helps ordering
    # For the table, we might just show Top 1-5.
    df_combined = df_combined.drop(columns=['Mean'])
    
    return df_combined

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
    ax.set_title('Feature Importance')
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

    plt.suptitle('Partial Dependence Plots', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Partial dependence plots saved to: {save_path}")

    return fig

def export_results_latex(
    all_results: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """
    Export all results to a single LaTeX file.
    
    all_results structure:
    {
        'DatasetName': {
            'metrics': pd.DataFrame (top 5 metrics),
            'importances': pd.DataFrame (feature importances comparative)
        }
    }
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(r"\section{Model Analysis Results}" + "\n\n")
        
        for dataset_name, data in all_results.items():
            f.write(f"\\subsection{{{dataset_name}}}" + "\n")
            
            # Metrics Table
            metrics_df = data['metrics']
            f.write(r"\subsubsection{Top 5 Models Performance}" + "\n")
            f.write(r"\begin{table}[H]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\begin{tabular}{|c|l|c|c|c|}" + "\n")
            f.write(r"\hline" + "\n")
            f.write(r"Rank & Params & RMSE & MAE & $R^2$ \\" + "\n")
            f.write(r"\hline" + "\n")
            
            for _, row in metrics_df.iterrows():
                # Escape special chars in params string
                params_str = str(row['Params']).replace("_", r"\_").replace("'", "").replace("{", "").replace("}", "")
                
                f.write(f"{row['Rank']} & {params_str} & {row['RMSE']:.4f} & {row['MAE']:.4f} & {row['R²']:.4f} \\\\" + "\n")
            
            f.write(r"\hline" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(f"\\caption{{Performance metrics for top 5 models - {dataset_name}}}" + "\n")
            f.write(r"\label{tab:metrics_" + dataset_name.lower().replace(" ", "_") + r"}" + "\n")
            f.write(r"\end{table}" + "\n\n")
            
            # Comparison Importance Table
            imp_df = data['importances']
            if not imp_df.empty:
                f.write(r"\subsubsection{Feature Importance Comparison}" + "\n")
                f.write(r"\begin{table}[H]" + "\n")
                f.write(r"\centering" + "\n")
                # Define columns: Feature + 5 models => l|c|c|c|c|c|
                f.write(r"\begin{tabular}{|l|c|c|c|c|c|}" + "\n")
                f.write(r"\hline" + "\n")
                f.write(r"Feature & Top 1 & Top 2 & Top 3 & Top 4 & Top 5 \\" + "\n")
                f.write(r"\hline" + "\n")
                
                for _, row in imp_df.iterrows():
                    feat_name = str(row['Feature']).replace("_", r"\_")
                    f.write(f"{feat_name} & {row['Top 1']:.4f} & {row['Top 2']:.4f} & {row['Top 3']:.4f} & {row['Top 4']:.4f} & {row['Top 5']:.4f} \\\\" + "\n")
                    
                f.write(r"\hline" + "\n")
                f.write(r"\end{tabular}" + "\n")
                f.write(f"\\caption{{Feature importance comparison for top 5 models - {dataset_name}}}" + "\n")
                f.write(r"\label{tab:imp_" + dataset_name.lower().replace(" ", "_") + r"}" + "\n")
                f.write(r"\end{table}" + "\n\n")

def airfoil_self_noise() -> Dict[str, Any]:
    logger.info("Running Airfoil Self-Noise pipeline...")

    air_X_train, air_X_test, air_y_train, air_y_test, air_df_train, _ = fetch_split_ucirepo(
        291, "scaled-sound-pressure"
    )

    corr_fig = corr_calc(air_df_train, OUTPUT_DIR / 'airfoil_corr.pdf')
    plt.close(corr_fig)

    air_X_train = air_X_train.drop(columns='attack-angle')
    air_X_test = air_X_test.drop(columns='attack-angle')

    # Focus on GBM for Airfoil
    _, _, _, air_gbm_best, top_5 = init_models(air_X_train, air_y_train, focus_model='gbm')

    # Evaluate Top 5
    top_5_metrics = evaluate_top_5(top_5, air_X_test, air_y_test, "GBM")
    
    # Get importances comparison
    importances_df = get_top_5_importances(top_5, air_X_train.columns)

    # Standard existing plots for best model
    best_model_obj = top_5[0][1]
    
    pi_fig = permutation_importance_plot(
        best_model_obj, air_X_test, air_y_test, OUTPUT_DIR / 'airfoil_perm_importance.pdf'
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        best_model_obj,
        air_X_test,
        ['frequency', 'suction-side-displacement-thickness'],
        ['Frequency', 'Suction-side-displacement-thickness'],
        OUTPUT_DIR / 'airfoil_pdp.pdf',
    )
    plt.close(pdp_fig)
    
    return {
        'metrics': top_5_metrics,
        'importances': importances_df
    }

def concrete_data() -> Dict[str, Any]:
    logger.info("Running Concrete dataset pipeline...")

    (
        concrete_X_train,
        concrete_X_test,
        concrete_y_train,
        concrete_y_test,
        concrete_df_train,
        _,
    ) = fetch_split_ucirepo(165, "Concrete compressive strength")

    corr_fig = corr_calc(concrete_df_train, OUTPUT_DIR / 'concrete_corr.pdf')
    plt.close(corr_fig)

    # Focus on GBM for Concrete
    _, _, _, concrete_gbm_best, top_5 = init_models(concrete_X_train, concrete_y_train, focus_model='gbm')

    top_5_metrics = evaluate_top_5(top_5, concrete_X_test, concrete_y_test, "GBM")
    
    importances_df = get_top_5_importances(top_5, concrete_X_train.columns)
    best_model_obj = top_5[0][1]

    pi_fig = permutation_importance_plot(
        best_model_obj,
        concrete_X_test,
        concrete_y_test,
        OUTPUT_DIR / 'concrete_perm_importance.pdf',
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        best_model_obj,
        concrete_X_test,
        ['Age', 'Cement'],
        ['Age', 'Cement'],
        OUTPUT_DIR / 'concrete_pdp.pdf',
    )
    plt.close(pdp_fig)
    
    return {
        'metrics': top_5_metrics,
        'importances': importances_df
    }

def wine_data() -> Dict[str, Any]:
    logger.info("Running Wine Quality dataset pipeline...")

    wine_X_train, wine_X_test, wine_y_train, wine_y_test, wine_df_train, _ = fetch_split_ucirepo(
        186, "quality"
    )

    corr_fig = corr_calc(wine_df_train, OUTPUT_DIR / 'wine_corr.pdf')
    plt.close(corr_fig)

    wine_X_train = wine_X_train.drop(columns='free_sulfur_dioxide')
    wine_X_test = wine_X_test.drop(columns='free_sulfur_dioxide')

    # Focus on RF for Wine
    _, _, _, _, top_5 = init_models(wine_X_train, wine_y_train, focus_model='rf')

    top_5_metrics = evaluate_top_5(top_5, wine_X_test, wine_y_test, "Random Forest")
    
    importances_df = get_top_5_importances(top_5, wine_X_train.columns)
    best_model_obj = top_5[0][1]

    pi_fig = permutation_importance_plot(
        best_model_obj, wine_X_test, wine_y_test, OUTPUT_DIR / 'wine_perm_importance.pdf'
    )
    plt.close(pi_fig)

    pdp_fig = partial_dependence_plot(
        best_model_obj,
        wine_X_test,
        ['alcohol', 'volatile_acidity'],
        ['Alcohol', 'Volatile Acidity'],
        OUTPUT_DIR / 'wine_pdp.pdf',
    )
    plt.close(pdp_fig)
    
    return {
        'metrics': top_5_metrics,
        'importances': importances_df
    }

def main():
    """Run all dataset pipelines sequentially."""
    logger.info("=" * 60)
    logger.info("Starting Application Pipelines")
    logger.info("=" * 60)

    results = {}
    
    results['Airfoil Self Noise'] = airfoil_self_noise()
    results['Concrete Strength'] = concrete_data()
    results['Wine Quality'] = wine_data()
    
    latex_path = OUTPUT_DIR / 'analysis_results.tex'
    export_results_latex(results, latex_path)

    logger.info("=" * 60)
    logger.info(f"All pipelines completed. Analysis saved to {latex_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()