#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


#Módulo de lectura de archivos CSV
import csv

#Module de chargement du modèle d'apprentissage
from sklearn.externals import joblib

def busquedaInformacionPokemon(numPokemon,Pokedex):
    infosPokemon = []
    for pokemon in Pokedex:
        if (int(pokemon[0])==numPokemon):
            infosPokemon = [pokemon[0],pokemon[1],pokemon[4],pokemon[5],pokemon[6],pokemon[7],pokemon[8],pokemon[9],pokemon[10]]
            break
    return infosPokemon


def prediccion (numeroPokemon1, numeroPokemon2,Pokedex):
    pokemon1 = busquedaInformacionPokemon(numeroPokemon1, Pokedex)
    pokemon2 = busquedaInformacionPokemon(numeroPokemon2, Pokedex)
    modelo_prediccion = joblib.load('modelo/modelo_pokemon.mod')
    prediccion_Pokemon_1 = modelo_prediccion.predict([[pokemon1[2],pokemon1[3],pokemon1[4],pokemon1[5],pokemon1[6],pokemon1[7],pokemon1[8]]])
    prediccion_Pokemon_2 = modelo_prediccion.predict([[pokemon2[2], pokemon2[3], pokemon2[4], pokemon2[5], pokemon2[6], pokemon2[7], pokemon2[8]]])
    print ("COMBATE QUE ENFRENTA: ("+str(numeroPokemon1)+") "+pokemon1[1]+" a ("+str(numeroPokemon2)+") "+pokemon2[1])
    print ("   "+pokemon1[1]+": "+str(prediccion_Pokemon_1[0]))
    print("   " +pokemon2[1] + ": " + str(prediccion_Pokemon_2[0]))
    print ("")
    if prediccion_Pokemon_1>prediccion_Pokemon_2:
        print(pokemon1[1].upper()+" ES EL GANADOR")
    else:
        print(pokemon2[1].upper() + " ES EL GANADOR")



#Carga de Pokedex e inicio de un combate
with open("datas/pokedex.csv", newline='') as csvfile:
    pokedex = csv.reader(csvfile)
    next(pokedex)
    prediccion(368,598,pokedex)

