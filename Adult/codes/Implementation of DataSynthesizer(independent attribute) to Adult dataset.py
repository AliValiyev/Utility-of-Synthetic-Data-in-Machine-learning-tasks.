from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd

# input dataset
input_data = 'adult.csv'
# location of two output files
mode = 'independent_attribute_mode'
description_file = 'description_for_adult_dataset.json'
synthetic_data = 'synthetic_data_for_adult_dataset_independent_mode.csv'
# An attribute is categorical if its domain size is less than this threshold.
threshold_value = 20

# specify categorical attributes
categorical_attributes = {'income': True}

# specify which attributes are candidate keys of input dataset.
candidate_keys = {'education': False}

# Number of tuples generated in synthetic dataset.
num_tuples_to_generate = 500

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data,
                                                         attribute_to_is_categorical=categorical_attributes,
                                                         attribute_to_is_candidate_key=candidate_keys)
describer.save_dataset_description_to_file(description_file)
generator = DataGenerator()
generator.generate_dataset_in_independent_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)
# Read both datasets using Pandas.
input_df = pd.read_csv(input_data, skipinitialspace=True)
synthetic_df = pd.read_csv(synthetic_data)
# Read attribute description from the dataset description file.
attribute_description = read_json_file(description_file)['attribute_description']

inspector = ModelInspector(input_df, synthetic_df, attribute_description)
for attribute in synthetic_df.columns:
    inspector.compare_histograms(attribute)
inspector.mutual_information_heatmap()