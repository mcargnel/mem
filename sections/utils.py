#| label: imports

import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.inspection import permutation_importance, PartialDependenceDisplay



#| label: fetch_split_ucirepo_funct
def fetch_split_ucirepo(id, dep_var, test_size = 0.2):

    dataset = fetch_ucirepo(id=id) 

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data.features, 
        dataset.data.targets, 
        test_size=test_size, 
        random_state=42
    )

    # Create complete dataframes for train and test sets
    df_train = X_train.copy()
    df_train[dep_var] = y_train

    df_test = X_test.copy()
    df_test[dep_var] = y_test

    return X_train, X_test, y_train, y_test, df_train, df_test


#| label: evaluate_model_funct

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Return metrics as a dictionary
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

#| label: param_grid

dt_param_grid = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_param_grid = {
    'n_estimators': [i for i in range(100, 2000, 100)]
}

gb_param_grid = {
    'n_estimators': [i for i in range(100, 2000, 100)],
    'learning_rate': [0.01, 0.1, 0.2]
}

def corr_calc(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 6))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlación de las variables')
    plt.tight_layout()
    plt.show()

#| label: concrete_init_models


def init_models(X_train, y_train):
    
    dt_model = DecisionTreeRegressor(random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    gbm_model = GradientBoostingRegressor(random_state=42)

    dt_grid = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    gbm_grid = GridSearchCV(gbm_model, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    
    lr_model = LinearRegression()    
    lr_model.fit(X_train, y_train)
    print("Linear Regression model fitted")

    dt_grid.fit(X_train, y_train)
    dt_best = dt_grid.best_estimator_
    print("Decision Tree model fitted")
    y_train_1d = y_train.values.ravel() if hasattr(y_train, 'values') else np.ravel(y_train)

    rf_grid.fit(X_train, y_train_1d)
    rf_best = rf_grid.best_estimator_
    print("Random Forest model fitted")

    gbm_grid.fit(X_train, y_train_1d)
    gbm_best = gbm_grid.best_estimator_
    print("Gradient Boosting model fitted")
    return lr_model, dt_best, rf_best, gbm_best

def evaluate_models(y_test, dt_best, rf_best, gbm_best, lm_model, X_test):
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

    # Create a dataframe with all metrics
    metrics_df = pd.DataFrame([lm_metrics, dt_metrics, rf_metrics, gbm_metrics])

    display(metrics_df.round(4).style.hide(axis='index'))



def permutation_importance_plot(best_model, X_test, y_test):
    # Perform permutation importance calculation
    result = permutation_importance(
        best_model, 
        X_test, 
        y_test, 
        n_repeats=10, 
        random_state=42
    )

    # Extract importance scores
    importances = result.importances_mean

    # Create a DataFrame to display the results
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)

    # Visualize the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], 
            color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance for Gradient Boosting Model')
    plt.tight_layout()
    plt.show()

    #| label: fig-concrete_partial_dependence
#| fig-cap: Dependencia parcial de las variables Age y Cement
#| include: True


def partial_dependence_plot(best_model, X_test, features, feature_titles):


    # Create the partial dependence plots in a 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # Loop through features and create plots
    for i, feature in enumerate(features):
        for j, centered in enumerate([False, True]):
            # Create partial dependence plot
            PartialDependenceDisplay.from_estimator(
                best_model,
                X_test,
                [feature],
                kind="both",
                subsample=50,
                n_jobs=-1,
                random_state=42,
                centered=centered,
                ax=[axs[i, j]]
            )
            
            # Customize plot
            centered_text = " (Centered)" if centered else ""
            axs[i, j].set_title(f'PDP of {feature_titles[i]}{centered_text}')
            axs[i, j].set_ylabel('Partial Dependence')
            axs[i, j].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Partial Dependence Plots for Gradient Boosting Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()