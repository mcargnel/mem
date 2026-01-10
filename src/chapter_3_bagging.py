import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from ucimlrepo import fetch_ucirepo

OUTPUT_DIR = Path(__file__).parent.parent / 'book/images/capitulo_3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_abalone_data():
    
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
    
    return X, y

def define_model(n_jobs=-1, n_estimators=20):
    bagg_model = BaggingRegressor(random_state=42, n_estimators=n_estimators, oob_score=True, bootstrap=True, n_jobs=n_jobs)
    return bagg_model

def get_oob_preds_error(bagg_model, X, y):
    bagg_model.fit(X, y)
    oob_preds = bagg_model.oob_prediction_
    return mean_squared_error(y, oob_preds)

def fit_loocv_model(bagg_model, X, y):
    scores = cross_val_score(bagg_model, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()

def run_simulations(X, y, n_estimators = range(10, 100, 10)):
    oob_results = np.array([])
    loocv_results = np.array([])
    for n_estimator in n_estimators:
        print(f"Running simulation for {n_estimator} estimators")
        bagg_model = define_model(n_estimators=n_estimator)
        oob_mse = get_oob_preds_error(bagg_model, X, y)
        loocv_mse = fit_loocv_model(bagg_model, X, y)
        oob_results = np.append(oob_results, oob_mse)
        loocv_results = np.append(loocv_results, loocv_mse)

    return oob_results, loocv_results

def plot_results(oob_results, loocv_results, n_estimators):
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, oob_results, label=r'OOB')
    plt.plot(n_estimators, loocv_results, label=r'LOOCV')
    
    plt.xlabel('Número de estimadores', fontsize=12)
    plt.ylabel('Error Cuadrático Medio (ECM)', fontsize=12)
    plt.title('Comparación de OOB y LOOCV', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(False) 
    
    save_path = OUTPUT_DIR / 'oob_vs_loocv.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def main():
    X, y = fetch_abalone_data()
    n_estimators = range(10, 100, 10)
    oob_results, loocv_results = run_simulations(X, y, n_estimators)
    plot_results(oob_results, loocv_results, n_estimators)

if __name__ == "__main__":
    main()