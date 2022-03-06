#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 2 19:01:27 2022
@author: alivaliyev
"""

#Implementation of SDV to Iris dataset.
import numpy as np 
import pandas as pd 
import warnings

#importing libraries
from sdv.tabular import GaussianCopula
from sdv.tabular import TVAE


#reading dataset
data= pd.read_csv("Iris.csv")

model_Gaussian = GaussianCopula(primary_key='Id')
model_TVAE = TVAE(primary_key='Id')

# fitting the models
model_Gaussian.fit(data)
model_TVAE.fit(data)

# creating synthetic data with each fitted model
new_data_model_Gaussian = model_Gaussian.sample(500)
new_data_model_TVAE = model_TVAE.sample(500)

new_data_model_Gaussian.head()
new_data_model_TVAE.head()

#saving synthetic datas
da = pd.DataFrame(new_data_model_Gaussian)
da.to_csv('model_GaussianforIris.csv', index=False)
dd = pd.DataFrame(new_data_model_TVAE)
dd.to_csv('model_TVAEforIris.csv', index=False)