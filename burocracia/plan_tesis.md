# Plan de Tesis de Maestría en Estadística

Directora: Daniela Rodriguez
Alumno: Martín Gabriel Cargnel

## a) Tema de investigación sobre el cual tratará el trabajo

La tesis consistirá en un estudio de técnicas de aprendizaje automático interpretable aplicadas a ensambles de árboles de decisión. En particular, se estudiarán métodos para cuantificar la importancia de las covariables y para estimar su efecto sobre la variable de respuesta en modelos de *Random Forest* y *Gradient Boosting Machines*, con el objetivo de mitigar el *trade-off* existente entre capacidad predictiva e interpretabilidad.

## b) Antecedentes sobre el tema

Los ensambles de árboles, como *Random Forest* (Breiman, 2001) y *Gradient Boosting Machines* (Friedman, 2001), son ampliamente utilizados en la práctica por su alta capacidad predictiva. Sin embargo, al agrupar múltiples árboles de decisión, estos modelos pierden la interpretabilidad que caracteriza a los árboles individuales, por lo que se los considera modelos de "caja negra".

Para abordar este problema, se desarrollaron diversas técnicas de interpretabilidad. *Split Gain Feature Importance* (Friedman, 2001) asigna importancias en base a la reducción del error en cada división del árbol, mientras que *Permutation Feature Importance* (Breiman, 2001; Fisher et al., 2019) mide el incremento del error al permutar los valores de cada covariable. No obstante, estas técnicas presentan inestabilidad: modelos con rendimiento predictivo similar pueden asignar importancias muy distintas a las variables, fenómeno conocido como el *Rashomon Effect* (Breiman, 2001). Fisher et al. (2019) proponen *Model Class Reliance* como solución, que computa el rango de importancias sobre todos los modelos con rendimiento similar.

En cuanto a la visualización del efecto de las variables, los *Partial Dependence Plots* (Friedman, 2001) permiten estimar el efecto marginal promedio de una covariable, y los *Individual Conditional Expectation* (Goldstein et al., 2015) extienden esta idea a nivel de observación individual, permitiendo detectar heterogeneidad e interacciones. Zhao y Hastie (2021) demostraron que, bajo ciertos supuestos, los *Partial Dependence Plots* admiten una interpretación causal a través de la conexión con el *backdoor adjustment* de Pearl (1993). Finalmente, Freiesleben et al. (2024) formalizan la relación entre estas técnicas y el verdadero proceso generador de datos, introduciendo los conceptos de DGP-PD y DGP-PFI como análogos poblacionales.

## c) Naturaleza del aporte proyectado

En primer lugar, el trabajo supondrá una exposición integral y autocontenida de las principales técnicas de interpretabilidad para ensambles de árboles, unificando en un mismo marco los métodos de importancia de variables, visualización de efectos y sus extensiones teóricas.

En segundo lugar, se presentarán aplicaciones prácticas en tres conjuntos de datos públicos (*Airfoil Self-Noise*, *Concrete Compressive Strength* y *Wine Quality*), comparando el rendimiento predictivo de regresión lineal, árboles de decisión, *Random Forest* y *Gradient Boosting Machines*. En cada caso, se aplicarán las técnicas de interpretabilidad estudiadas para extraer información sobre las variables más relevantes y su efecto sobre la variable dependiente, evaluando además la estabilidad de estas interpretaciones entre modelos con capacidad predictiva similar.

## d) Metodología tentativa a seguir para lograr los objetivos propuestos

El esquema de trabajo está planeado en las siguientes tareas:

1) Desarrollo teórico
    a) Se realizará una exposición teórica sobre los árboles de regresión y clasificación (CART), incluyendo la construcción del modelo, el algoritmo de poda, el manejo de *outliers* y las principales limitaciones que motivan el uso de técnicas de ensamble.
    b) Se estudiarán los ensambles de árboles, en particular *Bagging*, *Random Forest* y *Gradient Boosting Machines*, presentando sus fundamentos teóricos, las propiedades de reducción de varianza y las técnicas de regularización.
    c) Se presentarán las técnicas de interpretabilidad: *Split Gain Feature Importance*, *Permutation Feature Importance*, *Model Class Reliance*, *Partial Dependence Plots*, *Individual Conditional Expectation* y sus variantes (centradas y derivadas).
    d) Se discutirán las extensiones teóricas: la interpretación causal de los *Partial Dependence Plots* y la relación formal con el proceso generador de datos.

2) Aplicaciones prácticas
    a) Análisis exploratorio de los datos: descripción de los tres conjuntos de datos, con las visualizaciones y análisis de correlación correspondientes.
    b) Selección de covariables: se determinará qué variables incorporar a los modelos, basándose en el análisis de correlación realizado en el paso anterior.
    c) Búsqueda de hiperparámetros mediante *grid search* con validación cruzada para cada tipo de modelo.
    d) Evaluación y comparación de los modelos en conjuntos de prueba, reportando RMSE, MAE y R².
    e) Aplicación de las técnicas de interpretabilidad (*Permutation Feature Importance* y *Partial Dependence Plots*) a los mejores modelos, evaluando la estabilidad de los resultados entre modelos con rendimiento similar.
    f) Todo el análisis computacional se realizará haciendo uso de Python.
