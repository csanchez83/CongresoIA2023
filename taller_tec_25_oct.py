# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 01:58:32 2023

@author: LOBO_AZUL
"""
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

os.chdir("D:\\Google Drive\\_Posdoc 2022\\taller_tec_NM")

#%%
mi_data_pixeles = pd.read_csv("mnist_pixeles.csv",header=None)
mi_data_clases = pd.read_csv("mnist_clases.csv",header=None)

#%%

mi_data_clases.shape
mi_data_pixeles.shape

#%%

ejemplo_num = mi_data_pixeles.iloc[0]
array_ejemplo = ejemplo_num.to_numpy()
plt.imshow(ejemplo_num.to_numpy().reshape(28,28), cmap="Greys")

mi_data_clases.iloc[0]

#%%
'''Analizar balanceo de datos'''
mi_data_clases.value_counts()*100/mi_data_clases.shape[0]

#%%

from sklearn.decomposition import PCA
pca = PCA(0.8)

mnist_pca = pca.fit_transform(mi_data_pixeles)
mnist_pca.shape

#%%

from scipy.stats import randint as sp_randint

clf = KNeighborsClassifier()

busqueda_dist_parametros = {
    "n_neighbors": sp_randint(2,10),
    "p": sp_randint(1,3),
    "weights": ["uniform", "distance"]
}


#%%
from sklearn.model_selection import RandomizedSearchCV

busqueda = RandomizedSearchCV(estimator=clf,
                             param_distributions=busqueda_dist_parametros,
                             n_iter=3,
                             cv=3,
                             n_jobs=-1,
                             scoring="f1_micro")

busqueda.fit(X=mnist_pca, y=mi_data_clases.values.ravel())

busqueda.best_score_
busqueda.best_params_

#%%

mejores_params = {'n_neighbors': 3, 'p': 2, 'weights': 'distance'}

mejor_knn = KNeighborsClassifier(**mejores_params)
mejor_knn.fit(mnist_pca, mi_data_clases.values.ravel())


#%%

mi_numero = pd.read_csv("mi_numero.csv",header = None)
mi_numero.iloc[0].to_numpy()
plt.imshow(mi_numero.iloc[0].to_numpy().reshape(28,28), cmap="Greys")

#%%
nuevos_pca = pca.transform(mi_numero)
nuevos_pca
mejor_knn.predict(nuevos_pca)

