#-----------------------------------------------------------------------------------------

# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SEABORN 0.9.0
#   SCIKIT-LEARN 0.20.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Uso del módulo Pandas
import pandas as pnd

#Carga de dataset
dataset = pnd.read_csv("datas/dataset.csv",delimiter='\t')

#Eliminación de los valores NA (columnas: Primer Pokemon, Segundo Pokemon)
dataset = dataset.dropna(axis=0, how='any')


#X = se toman todos los datos, pero solo de las columnas 4 a la 11
#    PUNTOS_ATAQUE;PUNTOS_DEFENSA;PUNTOS_ATAQUE_ESPECIAL;PUNTO_DEFENSA_ESPECIAL;PUNTOS_VELOCIDAD;CANTIDAD_GENERACIONES
X = dataset.iloc[:, 5:12].values

#y = solo se toma la columna PORCENTAJE_DE_VICTORIA (16º valor)
y = dataset.iloc[:, 17].values


#Construccón del conjunto de entrenamiento y del conjunto de prueba
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_APRENDIZAJE, X_VALIDACION, Y_APRENDIZAJE, Y_VALIDACION = train_test_split(X, y, test_size = 0.2, random_state = 0)





#---- ALGORITMO 1: REGRESION LINEAL -----
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

#Elección del algoritmo
algoritmo = LinearRegression()

#Aprendizaje con ayuda de la función fit
algoritmo.fit(X_APRENDIZAJE, Y_APRENDIZAJE)

#Realización de la predicción sobre el conjunto de prueba
predicciones = algoritmo.predict(X_VALIDACION)

#Cálculo de la precisión del aprendizaje con ayuda de la
#función r2_score
precision = r2_score(Y_VALIDACION, predicciones)


print(">> ----------- REGRESION LINEAL -----------")
print(">> Precisión = "+str(precision))
print("------------------------------------------")



#---- ALGORITMO 2: ARBOL DE DECISION APLICA A LA REGRESION-----


#Elección del algoritmo
from sklearn.tree import DecisionTreeRegressor
algoritmo = DecisionTreeRegressor()

#Aprendizaje con ayuda de la función fit
algoritmo.fit(X_APRENDIZAJE, Y_APRENDIZAJE)

#Realización de la predicción sobre el conjunto de prueba
predicciones = algoritmo.predict(X_VALIDACION)

#Cálculo de la precisión del aprendizaje con ayuda de la
#función r2_score
precision = r2_score(Y_VALIDACION, predicciones)


print(">> ----------- ARBOLES DE DECISION -----------")
print(">> Precisión = "+str(precision))
print("------------------------------------------")




#Elección del algoritmo
from sklearn.ensemble import RandomForestRegressor
algoritmo = RandomForestRegressor()

#Aprendizaje con la ayuda de la función fit
algoritmo.fit(X_APRENDIZAJE, Y_APRENDIZAJE)

#Realización de la predicción sobre el conjunto de prueba
predicciones = algoritmo.predict(X_VALIDACION)

#Cálculo de la precisión del aprendizaje con ayuda de la
#función r2_score
precision = r2_score(Y_VALIDACION, predicciones)


print(">> ----------- ARBOLES ALEATORIOS -----------")
print(">> Precisión = "+str(precision))
print("------------------------------------------")


#Guardar el algoritmo
from sklearn.externals import joblib
archivo = 'modelo/modelo_pokemon.mod'
joblib.dump(algoritmo, archivo)