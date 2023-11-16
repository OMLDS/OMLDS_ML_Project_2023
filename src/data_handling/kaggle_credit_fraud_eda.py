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

# from pandas_profiling import ProfileReport
# from dataprep.eda import create_report

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 16)

from utils import read_dataset

# read dataset
# df_raw = pd.read_csv('./fraudTrain.csv', index_col=[0], parse_dates=['trans_date_trans_time', 'dob'])
df_raw = read_dataset(file = '../../data/fraudTrain.csv')
df_raw

df_raw.info()

# verify duplicated rows
df_raw.duplicated().sum()

# Comment: There are 0 duplicated rows.

# show duplicate rows
# df_raw[df_raw.duplicated()]

# descriptive statistics
df_raw.describe().T

# plot histograms
df_raw.hist()

# unique values per column/variable
df_raw.nunique()

# unique values for object type variables
df_raw.select_dtypes(include = 'object').nunique()

# plot unique values (bar chart)
df_raw.select_dtypes(include = 'object').nunique().plot.bar()
plt.xlabel('Variables')
plt.ylabel('Number of unique values')
plt.title('Cardinality check')
plt.show()

# df_raw['category'].value_counts(ascending=True).plot(kind='barh')

# sns.countplot(data=df_raw, x='category', order=df_raw['category'].value_counts().index)

from utils import drop_constant_column

df_raw = drop_constant_column(df_raw)

# verify missing values
df_raw.isnull().sum().sum()

# Comment: No missing values.

# count `is_fraud` (target)
df_raw['is_fraud'].value_counts(normalize=True)

# Comment: Target `is_fraud` is highly imbalanced.

# create a stratified sampling from 30% of total `df_raw` rows
# seed = 2019 (default)

from utils import sample_strata_df

df_sample = sample_strata_df(df_raw, frac = 0.5)
df_sample

df_sample['is_fraud'].value_counts(normalize=True)

# EDA - Pandas Profiling
# [profile = ProfileReport(df_sample, title="Kaggle Credit Card Fraud Detection Dataset Profile Report")

# profile.to_file('./kaggle_credit_card_fraud_report.html')]

# EDA - dataprep
# run create report
# report = create_report(df_sample, title = 'Kaggle Credit Card Fraud Detection Dataset EDA')

# report.save('kaggle_credit_card_fraud_dataprep_report.html')

# report.show_browser()

# report.show()

# plot `is_fraud` (target) count
sns.countplot(x='is_fraud', data=df_sample)

# plot `target` count by percent
# df_sample['is_fraud'] = df_sample['is_fraud'].map({'No': 0, 'Yes': 1})
ax = sns.barplot(x = 'is_fraud', y = 'is_fraud', data = df_sample, estimator = lambda x: len(x) / len(df_sample) * 100)
ax.set(ylabel = 'Percentage (%)')
plt.show()

# Comment: `is_fraud` (target) feature is higly imbalanced.

# subset features with less than 15 unique values
# cols_binary = df_clean.loc[:, df_clean.nunique() == 2].columns.to_list()

# remove selected features
# del_features = {'is_fraud', 'Sex', 'PhysicalHealth_bin', 'MentalHealth_bin'}

# cols_binary = [ele for ele in cols_binary if ele not in del_features]
# cols_binary

# convert selected features to binary
# df_clean[cols_binary] = df_clean[cols_binary].apply(lambda x: np.where(x == 'No', 0, 1))


# Bivariate Analysis (Continous - Continuous)
# Source: https://python-charts.com/correlation/scatter-plot-regression-line-seaborn/

# `amt` by `age`

# is 'age' a significant factor for 'is_fraud'
# df_sample['trans_date'] = df_sample['trans_date_trans_time'].dt.date
# df_sample['trans_date'] = pd.to_datetime(df_sample['trans_date'])

# calculate `age` by subtracting `trans_date` from `dob`
# df_sample['age'] = ((df_sample['trans_date'].min() - df_sample['dob']) / pd.Timedelta(days = 365)).astype(int)

from utils import feature_eng

# Feature engineering (create new features: city_state, age (customer), log_amt, 
# trans_month, trans_day, trans_wday, is_weekend, trans_hour, shop_net)

df_sample = feature_eng(df_sample)

# create feature distance
from utils import calc_distance, create_distance_feature

df_sample = create_distance_feature(df_sample)

df_sample.info()

df_sample.nunique()

# subset features with less than 15 unique values
cols_unique_less_15 = df_sample.loc[:, df_sample.nunique() < 16].columns.to_list()
cols_unique_less_15

# using seaborn
# sns.color_palette('colorblind')
f, axes = plt.subplots(6, 2, figsize=(15, 25), sharex=False)
for ax, feature in zip(axes.flat, df_sample[cols_unique_less_15]):
    sns.countplot(df_sample[feature], 
                  order =  df_sample[feature].value_counts().index, 
                  palette='colorblind', 
                  ax=ax)
    ax.tick_params(axis = 'x', rotation = 45)

# `age` histogram
df_sample['age'].hist()

sns.scatterplot(data = df_sample, x = 'age', y = 'amt')

# with regression line
sns.regplot(data = df_sample, x = 'age', y = 'amt', 
            scatter_kws = {"color": "black", "alpha": 0.5},
            line_kws = {"color": "red"})

# `lat` by `merch_lat`
sns.scatterplot(data = df_sample, x = 'lat', y = 'merch_lat')

# `amt` by `age` with hue = `is_fraud`
sns.lmplot(data = df_sample, x = 'age', y = 'amt', 
           col = 'is_fraud', 
           hue = 'is_fraud', markers = ["o", "x"], 
           scatter_kws = {"alpha": 0.5}, 
           line_kws = {"color": "red"})
plt.yscale('log')


# Bivariate continuous-categorical visualization
from utils import Bivariate_cont_cat

# is log(`city_pop`) a significant factor for `is_fraud`
# df_sample['log_city_pop'] = np.log1p(df_sample['city_pop'])
sns.boxplot(x = 'is_fraud', y = 'log_city_pop', data=df_sample)
Bivariate_cont_cat(df_sample, 'log_city_pop', 'is_fraud', 1)

# is log1p('amt') a significant factor for 'is_fraud'
# df_sample['log_amt'] = np.log1p(df_sample['amt'])
sns.boxplot(x = 'is_fraud', y = 'log_amt', data=df_sample)
Bivariate_cont_cat(df_sample, 'log_amt', 'is_fraud', 1)

# `age` boxplot
df_sample['age'].plot(kind='box')
sns.boxplot(x = 'age', data = df_sample)

# is `age` a significant factor for `is_fraud`
sns.boxplot(x = 'is_fraud', y = 'age', data = df_sample)
Bivariate_cont_cat(df_sample, 'age', 'is_fraud', 1)

# is `distance_home_buyer_city_state` a significant factor for `is_fraud`
sns.boxplot(x = 'is_fraud', y = 'distance_home_buyer_city_state', data = df_sample)
Bivariate_cont_cat(df_sample, 'distance_home_buyer_city_state', 'is_fraud', 1)

# Bivariate Analysis (Categorical - Categorical)
from utils import BVA_categorical_plot, stacked_bar_chart_with_ttest, px_stacked_bar_chart_with_ttest

# is 'gender' a significant factor for 'is_fraud'
stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'gender')
px_stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'gender')
BVA_categorical_plot(df_sample, 'is_fraud', 'gender')

# is 'category' a significant factor for 'is_fraud'
df_sample.groupby('category')['is_fraud'] \
     .value_counts(normalize=True).unstack() \
     .plot(kind='bar', stacked='True')

stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'category')
px_stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'category')
BVA_categorical_plot(df_sample, 'is_fraud', 'category')

# `age_bin`
# cut_bins = [12, 30, 46, 62, 79, 100]
# df_sample['age_bin'] = pd.cut(df_sample['age'], 
#                               bins=cut_bins, 
#                               labels=['12-30', '31-46', '47-62', '63-79', '80+'])

BVA_categorical_plot(df_sample, 'is_fraud', 'age_bin')

# is the transaction 'month' a significant factor for 'is_fraud'
# df_sample['trans_month'] = df_sample['trans_date_trans_time'].dt.month.astype('object')
BVA_categorical_plot(df_sample, 'is_fraud', 'trans_month')

# is the transaction 'day' a significant factor for 'is_fraud'
# df_sample['trans_day'] = df_sample['trans_date_trans_time'].dt.day.astype('object')
BVA_categorical_plot(df_sample, 'is_fraud', 'trans_day')

# is the transaction 'weekday' a significant factor for 'is_fraud'
# df_sample['trans_wday'] = df_sample['trans_date_trans_time'].dt.day_name()
BVA_categorical_plot(df_sample, 'is_fraud', 'trans_wday')

# is the transaction 'is_weekend' a significant factor for 'is_fraud'
# df_sample['is_weekend'] = df_sample['trans_date'].dt.day_of_week > 4
BVA_categorical_plot(df_sample, 'is_fraud', 'is_weekend')

# is the transaction 'hour' a significant factor for 'is_fraud'
# df_sample['trans_hour'] = df_sample['trans_date_trans_time'].dt.hour.astype('object')
BVA_categorical_plot(df_sample, 'is_fraud', 'trans_hour')

# create `shop_net` flag for `category` with `_net`
# df_sample['shop_net'] = df_sample['category'].str.contains('_net$', case=False)
stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'shop_net')
px_stacked_bar_chart_with_ttest(df_sample, 'is_fraud', 'shop_net')
BVA_categorical_plot(df_sample, 'is_fraud', 'shop_net')

# correlation heatmap (Pearson)
# df_sample['trans_hour'] = df_sample['trans_hour'].astype(int)
# df_sample['age_bin_cat'] = df_sample['age_bin'].astype('category')
# df_sample['age_bin_cat'] = df_sample['age_bin_cat'].cat.codes

# correlation heatmap (Pearson)
sns.heatmap(df_sample.corr(), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')

# correlation heatmap (Spearman)
sns.heatmap(df_sample.corr(method = 'spearman'), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')

# correlation heatmap (Kendall)
sns.heatmap(df_sample.corr(method = 'kendall'), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')

# `cc_num`
df_sample['cc_num'].value_counts()

df_sample.groupby(['cc_num', 'is_fraud'])['amt'].agg(['mean', 'median', 'std', 'size'])

# sample 15 `cc_num` values
seed = 4256
cc_num_sample_list = df_sample['cc_num'].sample(n = 15, random_state = seed).tolist()
cc_num_sample_list

df_cc_num_sample = df_sample[df_sample['cc_num'].isin(cc_num_sample_list)].reset_index(drop=True)
df_cc_num_sample

df_cc_num_sample.groupby(['cc_num', 'is_fraud'])['amt'].agg(['mean', 'median', 'std', 'size'])

# boxplot `amt` by `cc_num` and `is_fraud`
sns.boxplot(data=df_cc_num_sample, x = 'cc_num', y = 'amt', hue='is_fraud')
plt.yscale('log')
plt.xticks(rotation=45)

# `amt` data biinning by quartiles
# df_sample['amt_bin'] = pd.qcut(df_sample['amt'], q=4)
df_sample.groupby('amt_bin')['is_fraud'] \
     .value_counts(normalize=True) \
     .unstack() \
     .plot(kind='bar', stacked='True')

BVA_categorical_plot(df_sample, 'is_fraud', 'amt_bin')

# convert `amt_bin` to ordinal category
# df_sample['amt_bin_cat'] = df_sample['amt_bin'].astype('category')
# df_sample['amt_bin_cat'] = df_sample['amt_bin_cat'].cat.codes

# save df_sample as csv
# df_sample.to_csv('../../data/credit_card_fraud_sample.csv', index = False)

# save df_sample to pickle
# df_sample.to_pickle('../../data/credit_card_fraud_sample.pkl')

# read df_sample.csv
df_sample = pd.read_pickle('../../data/credit_card_fraud_sample.pkl')
df_sample

df_sample.info()

# statistical tests
import pingouin as pg

# pairwise t-test
# `is_fraud` by `amt`
# parametric test (assumes data has a Gaussian/normal distribution)
pg.pairwise_ttests(data = df_sample, dv = 'log_amt', between = 'is_fraud').round(3)

# non-parametrics (does not assume a particular distribution for the data)
pg.pairwise_ttests(data = df_sample, dv = 'log_amt', between = 'is_fraud', parametric = False) \
     .round(3)

# Cohen's d (effect size)
pg.compute_effsize(x = df_sample['log_amt'], y = df_sample['is_fraud'], eftype = 'cohen')

# Hedges g (effect size)
pg.compute_effsize(x = df_sample['log_amt'], y = df_sample['is_fraud'], eftype = 'hedges')

# `is_fraud` by `city_pop`
pg.pairwise_ttests(data = df_sample, dv = 'city_pop', between = 'is_fraud').round(3)

pg.pairwise_ttests(data = df_sample, dv = 'city_pop', between = 'is_fraud', parametric = False) \
     .round(3)

# Cohen's d (effect size)
pg.compute_effsize(x = df_sample['city_pop'], y = df_sample['is_fraud'], eftype = 'cohen')

# `is_fraud` by `age`
pg.pairwise_ttests(data = df_sample, dv = 'age', between = 'is_fraud').round(3)

pg.pairwise_ttests(data = df_sample, dv = 'age', between = 'is_fraud', parametric = False) \
     .round(3)

# Cohen's d (effect size)
pg.compute_effsize(x = df_sample['age'], y = df_sample['is_fraud'], eftype = 'cohen')

# chi2 test for independence between two categorical variables
# `is_fraud` by `gender`
pg.chi2_independence(data = df_sample, x = 'gender', y = 'is_fraud')

# `is_fraud` by `trans_hour`
pg.chi2_independence(data = df_sample, x = 'trans_hour', y = 'is_fraud')

# `is_fraud` by `amt_bin`
pg.chi2_independence(data = df_sample, x = 'amt_bin', y = 'is_fraud')

# city,state geo coordinates
# geo_coordinates = pd.read_csv('../../data/new_home_coords.csv')
# geo_coordinates

# grouped correlation
df_sample.groupby('is_fraud').apply(lambda df: df['lat'].corr(df['merch_lat']))

df_sample.groupby('is_fraud').apply(lambda df: df['long'].corr(df['merch_long']))
