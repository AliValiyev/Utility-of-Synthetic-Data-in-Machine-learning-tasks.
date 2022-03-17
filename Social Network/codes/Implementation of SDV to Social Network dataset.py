#Implementation of SDV to Social_Network_Ads dataset.
import numpy as np 
import pandas as pd 

#importing libraries
from sdv.tabular import GaussianCopula
from sdv.tabular import TVAE

#reading dataset
data= pd.read_csv("Social_Network_Ads.csv")

model_Gaussian = GaussianCopula(primary_key='User ID')
model_TVAE = TVAE(primary_key='User ID')

# fit the models
model_Gaussian.fit(data)
model_TVAE.fit(data)

# create synthetic data with each fitted model
new_data_model_Gaussian = model_Gaussian.sample(1000)
new_data_model_TVAE = model_TVAE.sample(1000)

new_data_model_Gaussian.head()
new_data_model_TVAE.head()

#saving synthetic datas
da = pd.DataFrame(new_data_model_Gaussian)
da.to_csv('model_Gaussian_for_Social_Network_Ads.csv', index=False)
dd = pd.DataFrame(new_data_model_TVAE)
dd.to_csv('model_TVAE_for_Social_Network_Ads.csv', index=False)
