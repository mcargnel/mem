import numpy as np
import pandas as pd

np.random.seed(73815)
n = 1000

# Generate all X features at once
X = np.random.uniform(0, 1, (n, 10))
X_cols = [f'X_{i+1}' for i in range(10)]

# Calculate y using vectorized operations where possible
y = (X[:, 0]**2 + np.sin(X[:, 1]) - np.exp(X[:, 2]) + np.log1p(X[:, 3]) + 
     X[:, 4]*X[:, 5] + X[:, 6]*X[:, 7]*X[:, 8] - np.sqrt(X[:, 9]) + 
     X[:, 0]*X[:, 1] - X[:, 2]*X[:, 3] + X[:, 4]*X[:, 5]*X[:, 6] +
     np.random.normal(0, 0.1, n))

# Create DataFrame with all columns
df = pd.DataFrame(X, columns=X_cols)
df['y'] = y

#| label: arbol-decision

from sklearn.model_selection import train_test_split
X = df.drop(columns='y')

y = df['y']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

# Initialize and train the regression tree model
regression_tree = DecisionTreeRegressor(random_state=42, max_depth=3)
regression_tree.fit(X_train, y_train)

#| label: fig-arbol-decision
#| include: true

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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
plt.show()

#| label: calc-arbol-alpha

from sklearn.metrics import mean_squared_error

# Definir un rango de complejidades del árbol (número máximo de nodos hoja)
max_leaf_nodes_list = [i for i in range(2,250, 10)]

# Entrenar y evaluar modelos
test_mse_scores = []
for max_leaf_nodes in max_leaf_nodes_list:
    reg = DecisionTreeRegressor(random_state=42, max_leaf_nodes=max_leaf_nodes)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    test_mse_scores.append(mse)

# Convertir los resultados a un DataFrame para análisis
results_df = pd.DataFrame({
    'max_leaf_nodes': max_leaf_nodes_list,
    'Error Cuadrático Medio': test_mse_scores
})

#| label: fig-arbol-alpha
#| include: true

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x=max_leaf_nodes_list, y=test_mse_scores, marker='o')
plt.xlabel('Número Máximo de Nodos Hoja')
plt.ylabel('Error Cuadrático Medio')
plt.title('Error Cuadrático Medio vs. Complejidad del Árbol (Máx. Nodos Hoja)')
plt.show()
