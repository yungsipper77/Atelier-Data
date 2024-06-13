import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import random
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
from matplotlib.dates import date2num,num2date
from sklearn.preprocessing import StandardScaler 
import folium



# %% Fonctions auxiliaires utilisées dans le code

#  Fonctions auxiliaires


def replaceStr (string,old,new) :
    """Fonction similaire à la méthode str.replace.
    Prend en entrée un string qui va être modifié, le caractère à remplacer et le nouveau caractère.
    Renvoie un nouveau string avec les caractères remplacés"""
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
    """Fonction en place
    Prends en paramètres une dataframe et optionnellement une liste de colonnes dont les valeurs doivent être transformées en float. Si columns n'est pas précisé, la fonction va s'appliquer sur toute la dataframe.
    Utilise la fonction auxiliaire : replaceStr
    Renvoie le dataframe avec les valeurs de la colonnes en float."""

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

def formeAdresse(num,typevoie,voie,commune,codepostal) :
    """Fonction qui renvoie l'entièreté de l'adresse à partir de plusieurs fragments.
    Prend en entré différentes str représentant le numéro de voie, le type de voie, le nom de la voie, la commune ainsi que le code postal.
    Renvoie l'adresse reformée en une unique str."""
    return str(num)[:-2] + " " + str(typevoie) + " " + str(voie) + " " + str(commune) + ", " + str(codepostal)[:-2]

# enlever les mauvais donnees en faisant plusieurs fois de kmeans
def kmeans_remove_outliers(data, k, max_iters=10, outlier_threshold=1.5):
    for i in range(max_iters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distances = np.min(kmeans.transform(data), axis=1)
        
        # calcule la limite de la distance
        threshold = np.percentile(distances, 100 * (1 - outlier_threshold / 100))
        
        # trouver les mauvais donnees
        outliers = distances > threshold
        if not np.any(outliers):
            break  # sinon arrete
        
        data = data[~outliers]  # enlever les mauvais donnes
        print(f"Iteration {i+1}: removed {np.sum(outliers)} outliers")
    
    return data, kmeans
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


# %% formatter les donnees
#suppression dépendance
df = df[df['Code type local'] != 3]



#transformation attribut
df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
df.dropna(subset=['Valeur fonciere', 'Nature mutation'], inplace=True)
df.dropna(subset=['Date mutation', 'Nature mutation'], inplace=True)
df.dropna(subset=['Valeur fonciere', 'Code departement'], inplace=True)
df.dropna(subset=['Valeur fonciere', 'Surface reelle bati', 'Commune'], inplace=True)



#transofrmation des NAN en 0 pour pouvoir calculer la moyenne de 0 pour chaque ligne et supprimer les lignes avec plus de 50% de 0 
#df_fill = df.fillna(0)
#print(df_fill)

# REMPLACER Maison et appartement par 1 et 2 pour faciliter le tri 
df["Type local"] = [1 if df.iloc[i]["Type local"] == "Maison" else 2 if "Appartement" else 3 for i in range(len(df))]

# Enlever les NaN
df.dropna(subset=['Valeur fonciere' ,"Nombre pieces principales",'Nombre de lots','Surface terrain','Type local' ,"No voie","Type de voie","Voie","Code postal","Commune"], inplace=True)


# Changement de type de str
#df['Valeur fonciere'] = df['Valeur fonciere'].str.replace(',', '.').astype(float)
#df['Surface reelle bati'] = df['Surface reelle bati'].astype(float)
df = dfStrToFloat(df,['Surface reelle bati','Valeur fonciere'])

# Ajout d'adresse
df["Adresse"] = [formeAdresse(df.iloc[obj]["No voie"], df.iloc[obj]["Type de voie"], df.iloc[obj]["Voie"], df.iloc[obj]["Commune"], df.iloc[obj]["Code postal"]) for obj in range (len(df))]

# Algo IQR
Q1 = df[['Valeur fonciere', 'Surface reelle bati']].quantile(0.25)
Q3 = df[['Valeur fonciere', 'Surface reelle bati']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Valeur fonciere', 'Surface reelle bati']] < (Q1 - 1.5 * IQR)) | (df[['Valeur fonciere', 'Surface reelle bati']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Enlever les NaN
df.dropna(subset=['Valeur fonciere', 'Surface reelle bati'], inplace=True)

# Calcule le nouveau attribut
df['Euro per m^2'] = df['Valeur fonciere'] / df['Surface reelle bati']

# Enlever NaN et inf
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(subset=['Euro per m^2'], inplace=True)


# IQR pour nettoyer Euro per m^2
Q1_euro = df['Euro per m^2'].quantile(0.25)
Q3_euro = df['Euro per m^2'].quantile(0.75)
IQR_euro = Q3_euro - Q1_euro
df = df[(df['Euro per m^2'] >= (Q1_euro - 1.5 * IQR_euro)) & (df['Euro per m^2'] <= (Q3_euro + 1.5 * IQR_euro))]

#   Data geographique
df_points = df[["Valeur fonciere","Nombre pieces principales",'Nombre de lots','Surface terrain','Type local']]

geolocator = Nominatim(user_agent="str")
Long = []
Lat = []
for ad in df["Adresse"] : 
    
    try:
        location = geolocator.geocode(ad)
        if location:
            Lat.append(location.latitude)
            Long.append(location.longitude)
        else:
            Lat.append(None)
            Long.append(None)
    except Exception as e:
        print(f"Erreur pour adresse : {ad}")
        Lat.append(None)
        Long.append(None)
        
df_points["Longitude"] = Long
df_points["Latitude"] = Lat

# %% progression de prix en moyen de tous les departements
# Calule le moyen Euro per m^2
mean_euro_per_m2 = df.groupby('Code departement')['Euro per m^2'].mean()
mean_euro_per_m2 = mean_euro_per_m2.sort_values(ascending=False)

plt.figure(figsize=(14, 7))
mean_euro_per_m2.plot(kind='bar')
plt.title('Average Euro per m^2 per Code Departement')
plt.xlabel('Code Departement')
plt.ylabel('Average Euro per m^2')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

 # %% zone des graphes

# Nature mutation/ Valeur moyenne
average_values = df.groupby('Nature mutation')['Valeur fonciere'].mean().reset_index()
average_values.rename(columns={'Valeur fonciere': 'Valeur Moyenne'}, inplace=True)
plt.figure(figsize=(12, 8))
plt.bar(average_values['Nature mutation'], average_values['Valeur Moyenne'], color='skyblue')
plt.ylabel('Valeur Moyenne')
plt.xlabel('Type mutation')
plt.title('Average Valeur fonciere by Type mutation')
plt.xticks(rotation=45, ha='right')  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Date / Mutation
grouped = df.groupby('Nature mutation')


colors = plt.cm.tab20(np.linspace(0, 1, len(grouped)))


plt.figure(figsize=(14, 10))

for (name, group), color in zip(grouped, colors):
    plt.hist(group['Date mutation'], bins=30, alpha=0.6, label=name, color=color)

plt.xlabel('Date')
plt.ylabel('Frequency')
plt.title('Distribution of Dates by Nature Mutation')
plt.legend(title='Nature mutation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Departement / Valeur moyenne (groupby)
average_values = df.groupby('Code departement')['Valeur fonciere'].mean().reset_index()
average_values.rename(columns={'Valeur fonciere': 'Valeur Moyenne'}, inplace=True)


plt.figure(figsize=(14, 10))
plt.bar(average_values['Code departement'], average_values['Valeur Moyenne'], color='skyblue')
plt.ylabel('Valeur Moyenne')
plt.xlabel('Code departement')
plt.title('Average Valeur fonciere by Code departement')
plt.xticks(rotation=90, ha='right')  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#  Commune / Euro per m^2
average_prices = df.groupby('Commune')['Euro per m^2'].mean().reset_index()


plt.figure(figsize=(14, 10))
plt.bar(average_prices['Commune'], average_prices['Euro per m^2'], color='blue')
plt.ylabel('Euro per m^2')
plt.xlabel('Commune')
plt.title('Average Price per Square Meter by Commune')
plt.xticks(rotation=90, ha='right')  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calcul de la moyenne du prix par mettre carré pour chaque type local  

prix_par_m2_median = df.groupby('Type local')['Euro per m^2'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(prix_par_m2_median['Type local'], prix_par_m2_median['Euro per m^2'], color='green')
plt.xlabel('Type local')
plt.ylabel('Prix par mètre carré médian')
plt.title('Prix moyen par mètre carré pour chaque type local (Maison, Appartement)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#  modèle knn 

df_points.dropna(subset=['Longitude','Latitude',"Nombre pieces principales",'Nombre de lots','Surface terrain','Type local'], inplace=True)

X = df_points.drop('Valeur fonciere', axis = 1)
Y = df_points['Valeur fonciere']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

model = LinearRegression()

model.fit(X_train,y_train)

mse = mean_squared_error(y_test,model.predict(X_test))
rmse = np.sqrt(mse)

print(mse,rmse)


#  Calcul kmeans

#kmeans prix par m2 en fonction de la surface reele batie :
filtered_df = df.dropna(subset=['Prix par m2'])
scaler = StandardScaler()
filtered_df[['Prix par m2']] = scaler.fit_transform(filtered_df[['Prix par m2']])
kmeans = KMeans(n_clusters=3, random_state=0).fit(filtered_df[['Prix par m2']])
filtered_df['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
for cluster in range(kmeans.n_clusters):
    cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
    plt.plot(cluster_data['Surface reelle bati'], cluster_data['Prix par m2'], 'o', label=f'Cluster {cluster}')

plt.xlabel('Surface réelle bâtie')
plt.ylabel('Prix par m2 (standardisé)')
plt.title('Clusters de Prix par mètre carré')
plt.legend()
plt.show()
filtered_df['Prix par m2'] = scaler.inverse_transform(filtered_df[['Prix par m2']])
for cluster in range(kmeans.n_clusters):
    cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(cluster_data[['Type local', 'Prix par m2']].groupby('Type local').mean())


#  kmeans

#dessiner scatter pour un code departement aleatoire (100 lignes de data)
# prix en fonciton de surface
# Choisir code departement
random_departements = df['Code departement'].unique()
selected_departement = random.choice(random_departements)

# Choisir 100 lignes de data
df_selected = df[df['Code departement'] == selected_departement].sample(n=100, random_state=42)

# Transferer date en num
df['Date mutation num'] = date2num(df['Date mutation'])

# Clustering sur Euro per m^2 en fonction de la Surface reelle bati 
X_selected = df_selected[['Euro per m^2', 'Surface reelle bati']].values
filtered_X_selected, final_kmeans_selected = kmeans_remove_outliers(X_selected, 3)

# Euro per m^2 vs. Surface reelle bati
plt.figure(figsize=(10, 6))
plt.scatter(filtered_X_selected[:, 0], filtered_X_selected[:, 1], alpha=0.6)
plt.scatter(final_kmeans_selected.cluster_centers_[:, 0], final_kmeans_selected.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title(f'Euro per m^2 en fonction de Surface reelle bati dans Code Departement {selected_departement}')
plt.xlabel('Euro per m^2')
plt.ylabel('Surface reelle bati (m^2)')
plt.grid(True)
plt.legend()
plt.annotate(f'Code Departement: {selected_departement}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
plt.show()

# Clustering sur Euro per m^2 en fonction de la Date mutation num 
X_date_selected = df_selected[['Euro per m^2', 'Date mutation num']].values
filtered_X_date_selected, final_kmeans_date_selected = kmeans_remove_outliers(X_date_selected, 3)

#  Euro per m^2 vs. Date mutation
plt.figure(figsize=(10, 6))
plt.scatter(df_selected['Euro per m^2'], df_selected['Date mutation'], alpha=0.6)
plt.scatter(final_kmeans_date_selected.cluster_centers_[:, 0], num2date(final_kmeans_date_selected.cluster_centers_[:, 1]), c='red', marker='x', s=200, label='Centroids')
plt.title(f'Distribution of Euro per m^2 vs. Date mutation in Code Departement {selected_departement}')
plt.xlabel('Euro per m^2')
plt.ylabel('Date mutation')
plt.grid(True)
plt.legend()
plt.annotate(f'Code Departement: {selected_departement}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
plt.show()







# maping

map_center = [46.603354, 1.888334]  # Latitude et longitude approximatives du centre de la France
zoom_level =  7  # Niveau de zoom pour afficher toute la France

my_map = folium.Map(location=map_center, zoom_start=zoom_level)

df_points["Couleurs"] = ["green" if df.iloc[i]['Valeur fonciere'] < 100000 else "blue" if df.iloc[i]['Valeur fonciere'] >= 100000 and df.iloc[i]['Valeur fonciere'] < 200000 else "purple" if df.iloc[i]['Valeur fonciere'] >= 200000 and df.iloc[i]['Valeur fonciere'] < 300000 else "orange" if df.iloc[i]['Valeur fonciere'] >= 300000 and df.iloc[i]['Valeur fonciere'] < 400000 else "red" if df.iloc[i]['Valeur fonciere'] < 1000000 else "yellow" for i in range(len(df))]

for i in range(len(df_points)):
    folium.CircleMarker(
        location=[df_points.iloc[i]["Latitude"],df_points.iloc[i]["Longitude"]],
        radius=0.1,
        color=df_points.iloc[i]["Couleurs"],
        fill=True,
        fill_color=df_points.iloc[i]["Couleurs"]
    ).add_to(my_map)
    
my_map.save("Map_ValeurFoncière.html")

