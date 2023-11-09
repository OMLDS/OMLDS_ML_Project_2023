# Kaggle Credit Card Fraud
# Source: https://examples.yourdictionary.com/well-written-examples-of-learning-objectives.html
# Data source: https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pprint import pprint as pp

import warnings
warnings.filterwarnings('ignore')

# Read and transform test dataset

from utils import read_dataset

# read dataset
test = read_dataset(file = '../../data/fraudTest.csv')
test

test.info()

# verify duplicated rows
test.duplicated().sum()

# descriptive statistics
test.describe().T

# plot histograms
test.hist()

# unique values per column/variable
test.nunique()

# verify missing values
test.isnull().sum().sum()

# drop constant columns
from utils import drop_constant_column

test = drop_constant_column(test)

# count `is_fraud` (target)
test['is_fraud'].value_counts(normalize=True)

# feature engineering (create new features)
from utils import feature_eng

test = feature_eng(test)

# create feature distance
from utils import calc_distance, create_distance_feature

test = create_distance_feature(test)

test.info()

test.nunique()

# additional features
test['log_city_pop'] = np.log1p(test['city_pop'])

# `age_bin`
cut_bins = [12, 30, 46, 62, 79, 100]
test['age_bin'] = pd.cut(test['age'], 
                              bins=cut_bins, 
                              labels=['12-30', '31-46', '47-62', '63-79', '80+'])

# correlation heatmap (Pearson)
test['trans_hour'] = test['trans_hour'].astype(int)
test['age_bin_cat'] = test['age_bin'].astype('category')
test['age_bin_cat'] = test['age_bin_cat'].cat.codes

# correlation heatmap (Pearson)
sns.heatmap(test.corr(), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')

# `amt` data biinning by quartiles
test['amt_bin'] = pd.qcut(test['amt'], q=4)

# convert `amt_bin` to ordinal category
test['amt_bin_cat'] = test['amt_bin'].astype('category')
test['amt_bin_cat'] = test['amt_bin_cat'].cat.codes

# save test to pickle
test.to_pickle('../../data/credit_card_fraud_test.pkl')
