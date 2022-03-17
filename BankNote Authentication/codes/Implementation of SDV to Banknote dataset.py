#Implementation of SDV to Titanic dataset.
import numpy as np 
import pandas as pd 

#importing libraries
from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import TVAE

#reading dataset
data= pd.read_csv("BankNote_Authentication.csv")

model_Gaussian = GaussianCopula()
model_CTGAN = CTGAN()
model_TVAE = TVAE()

# fit the models
model_Gaussian.fit(data)
model_CTGAN.fit(data)
model_TVAE.fit(data)

# create synthetic data with each fitted model
new_data_model_Gaussian = model_Gaussian.sample(1000)
new_data_model_CTGAN = model_CTGAN.sample(1000)
new_data_model_TVAE = model_TVAE.sample(1000)

new_data_model_Gaussian.head()
new_data_model_CTGAN.head()
new_data_model_TVAE.head()

da = pd.DataFrame(new_data_model_Gaussian)
da.to_csv('model_Gaussian_for_banknote.csv', index=False)
db = pd.DataFrame(new_data_model_CTGAN)
db.to_csv('model_CTGAN_for_banknote.csv', index=False)
dd = pd.DataFrame(new_data_model_TVAE)
dd.to_csv('model_TVAE_for_banknote.csv', index=False)
