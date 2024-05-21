# Données d'entraînement
data = [
   ["ou est situé l'iut de douala ?", "l'iut est situé dans le quartier ndongbon", "localisation"],
    ["qui est le directeur de l'iut ?", " l'actuel directeur de l'iut est le Pr Jacques ETAME","directeur"],
    ["Quels sont les différent département de l’iut de douala?", "les différents département de l'iut sont :  département du génie electrique et Informatique Industriel(GEII), département du génie industriel et Maintenance (GIM), département du génie mécanique et Productique(GMP), département du génie thermique et Energie,département du génie logistique et transport, département du génie informatique,département du génie Gestion des Entreprises et des Administrations","departements"]

   ]

# Séparation des caractéristiques (questions) et des étiquettes (information1)
questions = [row[0] for row in data]    # Caractéristiques : Questions
reponses = [row[1] for row in data]     # Étiquettes : Réponses
information1 = [row[2] for row in data] # Étiquettes : information1

# utilisation de labelEncoder

from sklearn.preprocessing import LabelEncoder

# Données d'exemple
information1 = ['localisation', 'directeur', 'departements', 'localisation']

# Création d'une instance de LabelEncoder
label_encoder = LabelEncoder()

# Encodage des variables catégorielles
information1_encoded = label_encoder.fit_transform(information1)

print(information1_encoded)
# gestion des valeurs manquante avec SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer

# Données d'exemple
data = {
    'Nom': ['ornella', 'yoan', 'vedrine', 'william'],
    'information1': ['localisation', 'directeur', 'departements', 'localisation'],
    'Âge': [20, 21, None, 25],
    'Email': ['ornella@gmail.com', 'yoan@gmail.com', None, 'william@gmail.com']
}

# Création du DataFrame à partir des données
df = pd.DataFrame(data)

# Affichage du DataFrame avant le traitement des valeurs manquantes
print(df)

# Suppression des enregistrements contenant des valeurs manquantes
df = df.dropna()

# Affichage du DataFrame après le traitement des valeurs manquantes
print(df)

# division des données grâce a la fonction train_test_split

from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df[['Nom', 'Départements', 'Âge']], df['Email'], test_size=0.2, random_state=42)

# Afficher les dimensions des ensembles d'entraînement et de test
print("Dimensions de l'ensemble d'entraînement :", X_train.shape, y_train.shape)
print("Dimensions de l'ensemble de test :", X_test.shape, y_test.shape)