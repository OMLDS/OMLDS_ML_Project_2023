import pandas as pd
import numpy as np

# read Kaggle credit card fraud transaction dataset
def read_dataset(file):
    """
    Function to read dataset and coerce type of selected columns

    Args:
        file (str): String for path and dataset filename (csv)

    Returns:
        _type_: Pandas dataframe
    """
    df = pd.read_csv(file, index_col=[0], parse_dates=['trans_date_trans_time', 'dob'])

    # coerce `cc_num` and `zip` to object type
    df['cc_num'] = df['cc_num'].astype('object')
    df['zip'] = df['zip'].astype('object')

    # combine `city` and `state`
    df['city_state'] = df['city'].astype(str) + ', ' + df['state']
    df['city_state'] = df['city_state'].str.strip()

    # convert 'gender' to binary feature
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

    return df

# drop constant value columns
def drop_constant_column(df):
    """
    Drops constant value columns of pandas dataframe (df).
    """
    return df.loc[:, (df != df.iloc[0]).any()]

# create a stratified sampling from precent of total dataframe rows
def sample_strata_df(data, frac, strata = 'is_fraud', seed = 2019):
    """
    Function to create a stratified sample from the dataset

    Args:
        data (dataframe): Pandas dataframe
        frac (float): Sampling percent from 0 to 1
        strata (str, optional): Dataframe column to stratify sample. Defaults to 'is_fraud'.
        seed (int, optional): Random seed number. Defaults to 2019.

    Returns:
        _type_: Pandas dataframe
    """
    seed = seed
    df = data.groupby(strata, group_keys=False) \
     .apply(lambda x: x.sample(frac=frac, random_state=seed)) \
     .reset_index(drop = 'True')
    
    return df