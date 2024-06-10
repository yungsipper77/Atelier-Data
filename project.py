import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import requests
import random
from io import StringIO

df=pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/78348f03-a11c-4a6b-b8db-2acf4fee81b1", sep="|",nrows=50)
print(df.shape)
#j'ai enlevé toutes les colonnes contanant que des NaN car elles ne servent pratiquement à rien.
df.dropna(axis=1 , how= 'all' )



# %% fonctions à appeler ultérieurement 


def replaceStr (string,old,new) :
    #Prend en entrée un string qui va être modifié, le caractère à remplacer et le nouveau caractère.
    #Renvoie un nouveau string avec les caractères remplacés"""
    result = ''
    if type(string) == str :
        for o in string :
            if o == old :
                result += new
            else :
                result += o

        return result
    else :
        print(string, " est de type ",type(string), " pas str.")



def dfStrToFloat(df,columns=[]) :
  #Renvoie le dataframe avec les valeurs de la colonnes en float."""

    if columns == []: #Si aucune colonnes n'est rentrée en paramètre on traite toute la dataframe
        columns = df.columns

    
    for o in columns : #On déroule sur toutes les colonnes
        newC = []   #Initialisation de la nouvelle colonne
        for string in df[o] :
            if type(string) == str :
                string = replaceStr (string,",",".") #S'il y a des virgules on les transforme en .
                string = float(string) #On transforme la chaîne en flottant

            newC.append(string) #On ajoute le flottant à la nouvelle ligne

        df[o] = newC #On met à jour la ligne

    return df #On renvoie la dataframe mise à jour


def skip_function(x):
    '''Fonction pour voir si un x est dans les lignes generes, sample_indices contient les indices aleatoire'''
    if x == 0:
        return False
    return x not in sample_indices

# %% main
url = "https://www.data.gouv.fr/fr/datasets/r/78348f03-a11c-4a6b-b8db-2acf4fee81b1"
response = requests.get(url)
data = response.text

#choisir 20 lignes aleatoirement
num_lines = sum(1 for line in data.split('\n')) - 1
sample_size = 20
sample_indices = sorted(random.sample(range(1, num_lines + 1), sample_size))
data_io = StringIO(data)


df = pd.read_csv(data_io, sep="|", usecols=['No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere',
                                            'No voie', 'B/T/Q', 'Type de voie', 'Code voie', 
                                            'Voie', 'Code postal','Commune', 'Code departement', 
                                            'Code commune', 'Section', 'No plan','1er lot', '2eme lot', 
                                            'Nombre de lots', 'Code type local','Type local', 'Surface reelle bati',
                                            'Nombre pieces principales','Nature culture', 'Surface terrain'],
                                            skiprows=skip_function)
df.dropna()


#suppression dépendance car généralement piece vide 
df = df[df['Code type local'] != 3]



#transformation attribut
df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')


#transofrmation des NAN en 0 
df_fill = df.fillna(0)
print(df_fill)

# REMPLACER Maison et appartement par 1 et 2 pour faciliter le tri 
new_df = df_fill.replace(["maison", "appartement"], [1, 2])

# suppression lignes avec NAN
df.dropna(subset=['Valeur fonciere', 'Surface reelle bati'], inplace=True)
# change data type
df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.').astype(float)
df['Surface reelle bati'] = df['Surface reelle bati'].astype(float)

# calcul euro par m2
df['Euro per m^2'] = df['Valeur fonciere'] / df['Surface reelle bati']

# tri par m2
df.sort_values('Euro per m^2', inplace=True)

# draw scatter
plt.figure(figsize=(10, 6))
plt.plot(df['Euro per m^2'], df['Surface reelle bati'], marker='o')
plt.title('Progression de la valeur fonciere par m^2 en fonction de surface')
plt.ylabel('Surface reelle bati')
plt.xlabel('Euro per m^2')
plt.yticks(np.arange(0, df['Surface reelle bati'].max() + 100, 100))
plt.xticks(np.arange(0, df['Euro per m^2'].max() + 1000, 1000))

plt.grid(True)
plt.show()
