#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 01:01:53 2022

@author: alivaliyev
"""

#Implementation of SDV to Titanic dataset.
import numpy as np 
import pandas as pd 

#importing libraries
from sdv.tabular import GaussianCopula
from sdv.tabular import TVAE

#reading dataset
data= pd.read_csv("titanic.csv")

model_Gaussian = GaussianCopula(primary_key='PassengerId')
model_TVAE = TVAE(primary_key='PassengerId')

# fit the models
model_Gaussian.fit(data)
model_TVAE.fit(data)

# create synthetic data with each fitted model
new_data_model_Gaussian = model_Gaussian.sample(1000)
new_data_model_TVAE = model_TVAE.sample(1000)

new_data_model_Gaussian.head()
new_data_model_TVAE.head()

da = pd.DataFrame(new_data_model_Gaussian)
da.to_csv('model_Gaussian_for_Titanic.csv', index=False)
dd = pd.DataFrame(new_data_model_TVAE)
dd.to_csv('model_TVAE_for_Titanic.csv', index=False)