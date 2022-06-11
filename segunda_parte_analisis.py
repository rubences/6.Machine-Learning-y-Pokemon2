#-----------------------------------------------------------------------------------------

# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SEABORN 0.9.0
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


#------------------------------------------
# IMPORTAR LOS MODULOS
#------------------------------------------

#Uso del módulo Pandas
import pandas as pnd

#Desactivación de la cantidad máxima de columnas del DataFrame a mostrar
pnd.set_option('display.max_columns',None)


#------------------------------------------
# ANALISIS DE LOS DATOS
#------------------------------------------

#Recuperación del código de la primera parte
nuestrosPokemon = pnd.read_csv("datas/pokedex.csv")
nuestrosPokemon['LEGENDARIO'] = (nuestrosPokemon['LEGENDARIO']=='VERDADERO').astype(int)
#nuestrosPokemon['NOMBRE'][62] = "Primeape"
combates = pnd.read_csv("datas/combates.csv")
nVecesPrimeraPosicion = combates.groupby('Primer_Pokemon').count()
nVecesSegundaPosicion = combates.groupby('Segundo_Pokemon').count()
cantidadTotalDeCombates = nVecesPrimeraPosicion + nVecesSegundaPosicion
cantidadDeVictorias = combates.groupby('Pokemon_Ganador').count()
listaAAgregar = combates.groupby('Pokemon_Ganador').count()
listaAAgregar.sort_index()
listaAAgregar['N_COMBATES'] = nVecesPrimeraPosicion.Pokemon_Ganador + nVecesSegundaPosicion.Pokemon_Ganador
listaAAgregar['N_VICTORIAS'] = cantidadDeVictorias.Primer_Pokemon
listaAAgregar['PORCENTAJE_DE_VICTORIAS']= cantidadDeVictorias.Primer_Pokemon/(nVecesPrimeraPosicion.Pokemon_Ganador + nVecesSegundaPosicion.Pokemon_Ganador)
nuevoPokedex = nuestrosPokemon.merge(listaAAgregar, left_on='NUMERO', right_index = True, how='left')

#Segunda parte

import matplotlib.pyplot as plt
import seaborn as sns

#Visualizacion de los Pokemon de tipo 1
axe_X = sns.countplot(x="TIPO_1", hue="LEGENDARIO", data=nuevoPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TIPO_1')
plt.ylabel('Total ')
plt.title("POKEMON DE TIPO_1")
plt.show()

#Visualización de los Pokemon de tipo 2
axe_X = sns.countplot(x="TIPO_2", hue="LEGENDARIO", data=nuevoPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TIPO_2')
plt.ylabel('Total')
plt.title("POKEMON DE TIPO_2")
plt.show()

#Búsqueda de correlación
print(nuevoPokedex.groupby('TIPO_1').agg({"PORCENTAJE_DE_VICTORIAS": "mean"}).sort_values(by = "PORCENTAJE_DE_VICTORIAS"))
corr = nuevoPokedex.loc[:,['TIPO_1','PUNTOS_DE_VIDA','PUNTOS_ATAQUE','PUNTOS_DEFENSA','PUNTOS_ATAQUE_ESPECIAL','PUNTO_DEFENSA_ESPECIAL','PUNTOS_VELOCIDAD','LEGENDARIO','PORCENTAJE_DE_VICTORIAS']].corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()

#Guardar Pokedex
dataset = nuevoPokedex
dataset.to_csv("datas/dataset.csv", sep='\t')

#X = se toman todos los datos, pero solo las características de 4 a 11
#PUNTOS_ATAQUE;PUNTOS_DEFENSA;PUNTOS_ATAQUE_ESPECIAL;PUNTO_DEFENSA_ESPECIAL;PUNTOS_VELOCIDAD;CANTIDAD_GENERACIONES
x = dataset.iloc[:, 4:11].values

#y = solo se toma la columna PORCENTAJE_DE_VICTORIA (característica 16) los : significan "Para todas las observaciones"
y = dataset.iloc[:, 16].values

#Distribución en conjunto de aprendizaje y conjunto de prueba
from sklearn.model_selection import train_test_split
X_aprendizaje, X_verificacion, y_aprendizaje, y_verificacion = train_test_split(x, y, test_size = 0.2, random_state = 0)
