#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importation des données:
train=pd.read_csv('train_data.csv')


# In[3]:


#Faire une copie de ces données pour faciliter l'execution:
df1=train.copy()


# In[4]:


df1.head(10)


# In[6]:


#Affichage des données:
len(df1)


# In[7]:


# Maitenant on va aplliquer ça à toutes les colonnes catégorielles
# Ici on récupère toutes les colonnes catégorielles
cat_cols = df1.select_dtypes(include=['object']).columns


# In[8]:


# On applique une boucle for
for col1 in cat_cols:
    freq = df1[col1].value_counts()
    df1[col1] = df1[col1].map(freq)


# In[10]:


num_cols = df1.select_dtypes(include=['float', 'int']).columns
for col2 in num_cols:
    mean = df1[col2].mean()
    df1[col2] = df1[col2].fillna(mean)


# In[13]:


df1.head()


# In[14]:


# on sépare nos données pour prédire la cible , ici on l'appelle target
y = df1['target']
X = df1.drop('target', axis=1)


# In[16]:


# Maintenant on peut séparer nos données en entrainement puis en données test 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Choix du modèle:
model = RandomForestClassifier(random_state=0)


# In[18]:


#Entrainement du modèle sur nos données d'entrainement:
model.fit(X_train,y_train)


# In[19]:


#Vérification du score de ce modèle:
val_accuracy = model.score(X_val, y_val)
print(f'le score sur l\'ensemble de validation : {val_accuracy:.2f}')


# In[20]:


# Pour calculer l'aire sous la courbe
y_pp = model.predict_proba(X_val)


# In[21]:


y_proba_pos = y_pp[:, 1]


# In[22]:


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_val, y_proba_pos)
print(f'le score de l\'air sous la courbe : {roc_auc:.2f}')


# In[23]:


y_predi=model.predict(X_val)


# In[24]:


# tracer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_val, y_proba_pos)


# In[25]:


plt.plot(fpr, tpr, 'b-', label='Modèle')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Courbe ROC')
plt.legend(loc='best')
plt.show()


# In[26]:


# Utilisation de la matrice de confusion pour évaluer le modèle
from sklearn.metrics import confusion_matrix
# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class = (y_predi > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_val, y_pred_class)
print(confusion_mat)


# #### Application du modèle sur les données test : 

# In[27]:


#Importation des données test : 
df2=pd.read_csv('test_data.csv')


# In[28]:


# On refait la meme transformation pour les données test
cat_cols2 = df2.select_dtypes(include=['object']).columns


# In[29]:


# On applique une boucle for pour parcourir toutes les variables qualitatives:
for colt in cat_cols2:
    freq = df2[colt].value_counts()
    df2[colt] = df2[colt].map(freq)


# In[30]:


# Essayons maintenant de gérer les valeurs manquantes 
num_cols1 = df2.select_dtypes(include=['float', 'int']).columns
for colt1 in num_cols1:
    mean = df2[colt1].mean()
    df2[colt1] = df2[colt1].fillna(mean)


# In[31]:


#Pour la prédiction finale:
y_pred = model.predict(df2)


# In[32]:


y_pred=pd.DataFrame(y_pred)


# In[33]:


y_pred=y_pred.rename(columns={0: 'Classes prédites'})


# In[34]:


print(y_pred)


# In[39]:


#Calcule des probabiltés qu'un client fera defaut ou pas :
proba=model.predict_proba(df2)


# In[40]:


proba=pd.DataFrame(proba)


# In[41]:


proba=proba.rename(columns={0: 'Classe 0 ',1 : 'classe 1'})


# In[42]:


print(proba)


# In[43]:


#Concatener les deux dataframes: 
con=pd.concat([proba,y_pred],axis=1)
con=con.to_numpy()


# In[44]:


#récuperer les probas où les classes égaux à 0:
C00=con[con[:,2] == 0]


# In[45]:


C11=con[con[:,2] == 1]


# #### Modèle 3 : Regression logistique

# In[46]:


from sklearn.linear_model import  LogisticRegression


# In[47]:


#Entrainemment du modèle 
reg_log=LogisticRegression(penalty='none',solver='newton-cg')


# In[48]:


reg_log.fit(X_train,y_train)


# In[49]:


#Effectuer la prédiction sur les données test:
predict=reg_log.predict(X_val)


# In[50]:


# Évaluation du modèle
val_accur= reg_log.score(X_val, y_val)
print(f'Accuracy: {val_accur:.2f}')


# In[51]:


#On effectue la prédiction sur nos données test:
p=reg_log.predict(df2)


# In[52]:


p=pd.DataFrame(p)
p=p.rename(columns={0: 'Classe prédites'})


# In[53]:


print(p)


# In[54]:


prob=reg_log.predict_proba(df2)


# In[55]:


prob=pd.DataFrame(prob)
prob=prob.rename(columns={0:'Classe 0', 1: 'Classe 1'})


# In[56]:


print(prob)


# In[57]:


# Utilisation de la matrice de confusion pour évaluer le modèle
from sklearn.metrics import confusion_matrix
# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class = (predict > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_val, y_pred_class)
print(confusion_mat)


# In[58]:


#Concatiner les deux dataframes 
hh=pd.concat([prob,p],axis=1)


# In[59]:


#transformer le nouveau dataframe en matrice
hh=hh.to_numpy()
#Selectionner seulement les lignes où la 3 ème colonne =1
C1 = hh[hh[:,2] == 1]


# In[60]:


#Selectionner seulement les lignes où la 3 ème colonne =0
C0=hh[hh[:,2] == 0]


# In[61]:


# Pour calculer l'aire sous la courbe
y_pr = reg_log.predict_proba(X_val)


# In[62]:


y_prob = y_pr[:, 1]


# In[63]:


roc_auc = roc_auc_score(y_val, y_prob)
print(f'le score de l\'air sous la courbe : {roc_auc:.2f}')


# In[64]:


# tracer la courbe ROC
fpr2, tpr2, thresholds2 = roc_curve(y_val, y_prob)


# In[65]:


plt.plot(fpr2, tpr2, 'b-', label='Modèle', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Courbe ROC')
plt.legend(loc='best')
plt.show()


# In[66]:


# Créer la figure
fig, ax = plt.subplots(figsize=(15, 6))

# Tracez la première courbe ROC
ax.plot(fpr2, tpr2, '--', label='Reg-logistique', color='black')
# Tracez la troisième courbe ROC
ax.plot(fpr, tpr, 'b-', label='Random Forest')

# Tracez la ligne en pointillé pour la ligne de hasard
ax.plot([0, 1], [0, 1], 'k-', label='Hasard', color='green')

# Ajouter les labels, le titre et la légende
ax.set_xlabel('Taux de faux positifs (FPR)')
ax.set_ylabel('Taux de vrais positifs (TPR)')
ax.set_title('Courbe ROC')
ax.legend(loc='best')

# Afficher la figure
plt.show()


# In[67]:


# Définir la taille de la figure
plt.figure(figsize=(20, 3))
# Tracer les densités de probabilité pour chaque colonne
sns.kdeplot(C00[:, 0], shade=True, label='Classe 0')
sns.kdeplot(C11[:, 1], shade=True, label='Classe 1')

# Ajouter les labels et le titre du graphique
plt.xlabel('probabilités prédites')
plt.ylabel('Densités de probabilités')
plt.title('Densités de probabilité prédites pour le Random Forest')
plt.legend(loc='upper center')

# Afficher le graphique
plt.show()


# In[68]:


# Définir la taille de la figure
plt.figure(figsize=(20, 3))
# Tracer les densités de probabilité pour chaque colonne
sns.kdeplot(C0[:, 0], shade=True, label='Classe 0')
sns.kdeplot(C1[:, 1], shade=True, label='Classe 1')

# Ajouter les labels et le titre du graphique
plt.xlabel('probabilités prédites')
plt.ylabel('Densités de probabilités')
plt.title('Densités de probabilité prédites pour la Reg-logistique')
plt.legend(loc='upper center')

# Afficher le graphique
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')


# In[52]:


#Importation des données:
train=pd.read_csv('train_data.csv')


# In[53]:


#Faire une copie de ces données pour faciliter l'execution:
df1=train.copy()


# In[54]:


df1.head(10)


# In[55]:


#Récuprer les variables catégorielles:
cat_features=df1.select_dtypes(include=['object']).columns
#Récuprer les colonnes des variables numériques :
num_features= df1.select_dtypes(include=['int64','float64']).columns


# In[56]:


#On va importer la fonction qui fait l'encodage des variables catégorielles: 
from sklearn.preprocessing import LabelEncoder
scale=LabelEncoder()
for i in cat_features:
    df1[i]=scale.fit_transform(df1[i])


# In[57]:


#Remplacer les valeurs manquantes par la médiane: 
for i in num_features:
    df1[i].fillna(df1[i].median(),inplace=True)


# Encodage de label : L'encodage de label est une méthode qui remplace chaque catégorie par un nombre entier unique. Cette méthode est souvent utilisée pour les variables catégorielles avec un petit nombre de catégories uniques.

# 
# L'encodage de label consiste à remplacer chaque valeur unique d'une variable catégorielle par une étiquette numérique unique, qui peut être utilisée comme entrée dans un modèle de machine learning. Les étiquettes sont attribuées en fonction de l'ordre d'apparition des valeurs uniques dans la variable catégorielle, sans tenir compte de l'ordre ou de la signification intrinsèque de ces valeurs. Par conséquent, l'encodage de label peut être utilisé pour toutes les variables catégorielles, ordonnées ou non ordonnées.

# In[58]:


print(df1.head(10))


# In[59]:


# on sépare nos données pour prédire la cible , ici on l'appelle target
y = df1['target']
X = df1.drop('target', axis=1)


# In[60]:


# Maintenant on peut séparer nos données en entrainement puis en données test 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[61]:


# Choix du modèle:
model = RandomForestClassifier(random_state=42)


# In[62]:


#Entrainement du modèle sur nos données d'entrainement:
model.fit(X_train,y_train)


# In[63]:


#Vérification du score de ce modèle:
val_accuracy = model.score(X_val, y_val)
print(f'le score sur l\'ensemble de validation : {val_accuracy:.2f}')


# In[64]:


# Pour calculer l'aire sous la courbe
y_p = model.predict_proba(X_val)
print(y_p)


# In[65]:


y_proba_pos = y_p[:, 1]


# In[66]:


print(y_proba_pos)


# In[67]:


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_val, y_proba_pos)
print(f'le score de l\'air sous la courbe : {roc_auc:.2f}')


# In[68]:


y_predi=model.predict(X_val)
print(y_predi)


# In[69]:


from sklearn.metrics import mean_squared_error, r2_score
erreur=mean_squared_error(y_val,y_predi)
print(erreur)


# In[70]:


# tracer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_val, y_proba_pos)


# In[71]:


plt.plot(fpr, tpr, 'b-', label='Modèle')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Courbe ROC')
plt.legend(loc='best')
plt.show()


# #### Application du modèle sur les données test : 

# In[72]:


#Importation des données test : 
df2=pd.read_csv('test_data.csv')


# In[73]:


#Récuprer les variables catégorielles:
cat_features=df2.select_dtypes(include=['object']).columns
#Récuprer les colonnes des variables numériques :
num_features= df2.select_dtypes(include=['int64','float64']).columns


# In[74]:


#On va importer la fonction qui fait l'encodage des variables catégorielles: 
from sklearn.preprocessing import LabelEncoder
scale=LabelEncoder()
for i in cat_features:
    df2[i]=scale.fit_transform(df2[i])


# In[75]:


#Remplacer les valeurs manquantes par la médiane: 
for i in num_features:
    df2[i].fillna(df2[i].median(),inplace=True)


# In[76]:


print(df2.head())


# In[155]:


#Pour la prédiction finale:
y_pred= model.predict(df2)


# In[156]:


y_pred=pd.DataFrame(y_pred)


# In[157]:


y_pred=y_pred.rename(columns={0: 'Classes prédites'})


# In[158]:


print(y_pred)


# In[159]:


#Calcule des probabiltés qu'un client aura l'approbation ou pas  :
proba=model.predict_proba(df2)


# In[160]:


proba=pd.DataFrame(proba)


# In[161]:


proba=proba.rename(columns={0: 'Classe 0 ',1 : 'classe 1'})


# In[162]:


print(proba)


# In[165]:


#Concatener les deux dataframes: 
con=pd.concat([proba,y_pred],axis=1)
con=con.to_numpy()


# In[168]:


#récuperer les probas où les classes égaux à 0:
C00=con[con[:,2] == 0]


# In[169]:


C11=con[con[:,2] == 1]


# In[86]:


# Utilisation de la matrice de confusion pour évaluer le modèle
from sklearn.metrics import confusion_matrix
# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class = (y_predi > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_val, y_pred_class)
print(confusion_mat)


# ###### Modèle 2: Utiliser LASSO 

# utiliser la régression lasso et voir ce qu'il se passe 

# In[87]:


from sklearn.linear_model import Lasso


# In[88]:


lasso = Lasso(alpha=0.03)
lasso.fit(X_train, y_train)
coef = lasso.coef_


# In[89]:


#selection des caractérisiques non nuls : 
selected_features = [X_train.columns[i] for i in range(len(coef)) if coef[i] != 0]
#model de random forest sur cette selection 
rf = RandomForestClassifier(n_estimators=200, max_depth=100)


# In[90]:


#Entrainement du modèle sur cette classification:
rf.fit(X_train[selected_features], y_train)


# In[91]:


# resultats de la prédiction: 
y_pred = rf.predict(X_val[selected_features])
# score de cette préidction: 
accuracy = rf.score(X_val[selected_features], y_val)
print(f"Accuracy: {accuracy:.2f}")


# In[92]:


# Pour calculer l'aire sous la courbe
y_prob = rf.predict_proba(X_val[selected_features])


# In[93]:


y_proba = y_prob[:, 1]


# In[94]:


roc_auc = roc_auc_score(y_val, y_proba)
print(f'le score de l\'air sous la courbe : {roc_auc:.2f}')


# In[95]:


# tracer la courbe ROC
fpr1, tpr1, thresholds1 = roc_curve(y_val, y_proba)


# In[96]:


plt.plot(fpr1, tpr1, 'b-', label='Modèle avec Lasso ', color="red")
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Courbe ROC')
plt.legend(loc='best')
plt.show()


# In[97]:


y_pred1=rf.predict(df2[selected_features])


# In[98]:


y_pred1=pd.DataFrame(y_pred1)


# In[99]:


y_pred1=y_pred1.rename(columns={0: 'Classes prédites'})


# In[100]:


print(y_pred1)


# In[101]:


# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class1 = (y_pred > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat1 = confusion_matrix(y_val, y_pred_class1)
print(confusion_mat)


# In[102]:


y_chap=rf.predict_proba(df2[selected_features])


# In[103]:


y_chap=pd.DataFrame(y_chap)
y_chap=y_chap.rename(columns={0: 'Classe 0', 1 :'Classe 1'})


# In[104]:


print(y_chap)


# In[177]:


#Concatener les deux dataframes: 
conca=pd.concat([y_chap,y_pred1],axis=1)
conca=conca.to_numpy()


# In[179]:


#récuperer les proba des clients où la classe égale à 0
C01=conca[conca[:,2] == 0]
#récuperer les proba des clients où la classe égale à 0
C10=conca[conca[:,2] == 1]


# #### Modèle 3 : Regression logistique

# In[105]:


from sklearn.linear_model import  LogisticRegression


# In[106]:


#Entrainemment du modèle 
reg_log=LogisticRegression(penalty='none',solver='newton-cg')


# In[107]:


reg_log.fit(X_train,y_train)


# In[108]:


#Effectuer la prédiction sur les données test:
predict=reg_log.predict(X_val)


# In[109]:


# Évaluation du modèle
val_accur= reg_log.score(X_val, y_val)
print(f'Accuracy: {val_accur:.2f}')


# In[110]:


#On effectue la prédiction sur nos données test:
p=reg_log.predict(df2)


# In[111]:


p=pd.DataFrame(p)
p=p.rename(columns={0: 'Classe prédites'})


# In[112]:


print(p)


# In[113]:


prob=reg_log.predict_proba(df2)


# In[114]:


prob=pd.DataFrame(prob)
prob=prob.rename(columns={0:'Classe 0', 1: 'Classe 1'})


# In[131]:


print(prob)

# In[149]:


#Concatiner les deux dataframes 
hh=pd.concat([prob,p],axis=1)


# In[150]:


#transformer le nouveau dataframe en matrice
hh=hh.to_numpy()
#Selectionner seulement les lignes où la 3 ème colonne =1
C1 = hh[hh[:,2] == 1]
C1


# In[144]:


#Selectionner seulement les lignes où la 3 ème colonne =0
C0=hh[hh[:,2] == 0]


# In[116]:


# Pour calculer l'aire sous la courbe
y_pr = reg_log.predict_proba(X_val)


# In[117]:


y_prob = y_pr[:, 1]


# In[118]:


print(y_prob)


# In[119]:


roc_auc = roc_auc_score(y_val, y_prob)
print(f'le score de l\'air sous la courbe : {roc_auc:.2f}')


# In[120]:


# tracer la courbe ROC
fpr2, tpr2, thresholds2 = roc_curve(y_val, y_prob)


# In[121]:


plt.plot(fpr2, tpr2, 'b-', label='Modèle', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Courbe ROC')
plt.legend(loc='best')
plt.show()


# In[122]:


# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class2 = (predict > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat2 = confusion_matrix(y_val, y_pred_class2)
print(confusion_mat)


# ### Supperposition des graphes ROC pour chaque modèle "Lasso", "Random Forest","Reg-logistique"

# In[123]:


# Créer la figure
fig, ax = plt.subplots(figsize=(15, 6))

# Tracez la première courbe ROC
ax.plot(fpr2, tpr2, '--', label='Reg-logistique', color='black')
# Tracez la troisième courbe ROC
ax.plot(fpr, tpr, 'b-', label='Random Forest')

# Tracez la deuxième courbe ROC
ax.plot(fpr1, tpr1, 'b-', label='Lasso ', color="yellow")

# Tracez la ligne en pointillé pour la ligne de hasard
ax.plot([0, 1], [0, 1], 'k-', label='Hasard', color='green')

# Ajouter les labels, le titre et la légende
ax.set_xlabel('Taux de faux positifs (FPR)')
ax.set_ylabel('Taux de vrais positifs (TPR)')
ax.set_title('Courbe ROC')
ax.legend(loc='best')

# Afficher la figure
plt.show()


# In[173]:


# Définir la taille de la figure
plt.figure(figsize=(20, 3))
# Tracer les densités de probabilité pour chaque colonne
sns.kdeplot(C00[:, 0], shade=True, label='Classe 0')
sns.kdeplot(C11[:, 1], shade=True, label='Classe 1')

# Ajouter les labels et le titre du graphique
plt.xlabel('probabilités prédites')
plt.ylabel('Densités de probabilités')
plt.title('Densités de probabilité prédites pour le Random Forest')
plt.legend(loc='upper center')

# Afficher le graphique
plt.show()


# In[181]:


# Définir la taille de la figure
plt.figure(figsize=(20, 3))
# Tracer les densités de probabilité pour chaque colonne
sns.kdeplot(C01[:, 0], shade=True, label='Classe 0')
sns.kdeplot(C10[:, 1], shade=True, label='Classe 1')

# Ajouter les labels et le titre du graphique
plt.xlabel('probabilités prédites')
plt.ylabel('Densités de probabilités')
plt.title('Densités de probabilité prédites pour le Random forest avec Lasso')
plt.legend(loc='upper center')

# Afficher le graphique
plt.show()


# In[174]:


# Définir la taille de la figure
plt.figure(figsize=(20, 3))
# Tracer les densités de probabilité pour chaque colonne
sns.kdeplot(C0[:, 0], shade=True, label='Classe 0')
sns.kdeplot(C1[:, 1], shade=True, label='Classe 1')

# Ajouter les labels et le titre du graphique
plt.xlabel('probabilités prédites')
plt.ylabel('Densités de probabilités')
plt.title('Densités de probabilité prédites pour la Reg-logistique')
plt.legend(loc='upper center')

# Afficher le graphique
plt.show()


# In[ ]:




# In[ ]:
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pandas.plotting import scatter_matrix


# In[ ]:


# Séparation des données en ensembles d'entraînement et de validation
X = df1.drop('target', axis=1)
y = df1['target']


# In[ ]:


# Normaliser les variables numériques
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_val)


# In[ ]:


#Initialisation du modèle
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compiler le modèle
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train_norm, y_train, epochs=50, batch_size=50)


# In[ ]:


# Entraînement du modèle
model.fit(X_train_norm, y_train, epochs=50, batch_size=50)


# In[ ]:


# Obtenir les probabilités prédites pour les données de test
y_pred = model.predict(X_test_norm)

# Calculer les taux de faux positifs et les taux de vrais positifs
fpr, tpr, seuils = roc_curve(y_val, y_pred_proba)

# Calculer l'aire sous la courbe ROC (AUC)
auc_score = auc(fpr, tpr)

# Calculer la courbe ROC
fpr, tpr, seuils = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.plot(fpr, tpr, color='blue', label='Courbe ROC (AUC = %0.2f)' % roc_auc)
# Tracer la prédiction du hasard
plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Prédiction du hasard')
# Ajouter la légende
plt.legend(loc="lower right")
plt.title("Courbe ROC")
plt.grid(True)
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test_norm)

# Convertir les prédictions en étiquettes de classe en utilisant une seuil de 0,5
y_pred_class = (y_pred > 0.5).astype(int)

# Calculer la matrice de confusion
confusion_mat = confusion_matrix(y_val, y_pred_class)
print(confusion_mat)

# In[ ]:


#Pour la prédiction final
y_pred = model.predict(df2)


# In[ ]:


#mettre sous forme de Dataframe
df_pred = pd.DataFrame(y_pred, columns =['Result'],dtype=int)

