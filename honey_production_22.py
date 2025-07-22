import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

import pandas as pd

# 1. Chargement des données
honey_data = pd.read_csv('data/honeyproduction.csv')
df = pd.DataFrame(honey_data)

# 2. Affichage des premières lignes pour comprendre la structure
print("📊 Structure du DataFrame :\n", df.head())

# 3. Groupement par année
prod_per_year = df.groupby('year')

# 4. Moyenne de la production totale par an
print("\n📈 Moyenne de la production de miel par an :")
mean_prod_per_year = prod_per_year['totalprod'].mean().reset_index()
print(mean_prod_per_year)

# 5. Somme de la production totale par an
print("\n📊 Somme de la production de miel par an :")
sum_prod_per_year = prod_per_year['totalprod'].sum().reset_index()
print(sum_prod_per_year)



#------------>>>>3

X = mean_prod_per_year['year']
X = X.values.reshape(-1,1)

#------------>>>>4
y = mean_prod_per_year['totalprod']


#------------>>>>5
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='orange', label='Données réelles')
plt.title('Production moyenne de miel par année')
plt.xlabel('Année')
plt.ylabel('Production moyenne (totalprod)')
plt.grid(True)
plt.legend()
plt.show()


print("\n\n")



"""Create and Fit a Linear Regression Model"""
#------------>>>>6
'''1- model'''
regr = LinearRegression()

#------------>>>>7
'''2- entrainement du model '''
regr.fit(X,y)
#------------>>>>8
'''3- affichage de regr.coef_ (la pente 'm') et l'ordonnee a l'origine regr.intercept_'''
print("La pente est 'm' est : ", regr.coef_)
print("`l'ordonnee a l'origine est : ", regr.intercept_)
#------------>>>>9

print("prediction : ")
y_predict = regr.predict(X)

#------------>>>>10
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='orange', label='Données réelles')
plt.plot(X, y_predict, color='blue', label='Régression linéaire')
plt.title('Régression linéaire de la production de miel')
plt.xlabel('Année')
plt.ylabel('Production moyenne (totalprod)')
plt.grid(True)
plt.legend()
plt.show()

print("\n\n")




"""Predict the Honey Decline"""



