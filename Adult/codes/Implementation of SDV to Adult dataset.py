#Implementation of SDV to Titanic dataset.
import numpy as np 
import pandas as pd 

#importing libraries
from sdv.tabular import GaussianCopula
from sdv.tabular import TVAE

#reading dataset
data= pd.read_csv("adult.csv")

model_Gaussian = GaussianCopula()
model_TVAE = TVAE()

# fit the models
model_Gaussian.fit(data)
model_TVAE.fit(data)

# create synthetic data with each fitted model
new_data_model_Gaussian = model_Gaussian.sample(500)
new_data_model_TVAE = model_TVAE.sample(500)

new_data_model_Gaussian.head()
new_data_model_TVAE.head()

da = pd.DataFrame(new_data_model_Gaussian)
da.to_csv('model_Gaussian_for_adult.csv', index=False)
dd = pd.DataFrame(new_data_model_TVAE)
dd.to_csv('model_TVAE_for_adult.csv', index=False)
