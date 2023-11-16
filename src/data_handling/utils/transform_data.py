import pandas as pd
import numpy as np
from haversine import haversine, Unit
import category_encoders as ce

def feature_eng(data):
    """
    Function to create new features from dataset

    Args:
        data (dataframe): Pandas dataframe

    Returns:
        _type_: Pandas dataframe
    """
    # calculate `age` feature
    data['trans_date'] = data['trans_date_trans_time'].dt.date
    data['trans_date'] = pd.to_datetime(data['trans_date'])
    data['age'] = ((data['trans_date'].min() - data['dob']) / pd.Timedelta(days = 365)).astype(int)

    # create `log_amt`
    data['log_amt'] = np.log1p(data['amt'])

    # create `age_bin`
    cut_bins = [12, 30, 46, 62, 79, 100]
    data['age_bin'] = pd.cut(data['age'], 
                             bins=cut_bins, 
                             labels=['12-30', '31-46', '47-62', '63-79', '80+'])

    # create date features
    data['trans_month'] = data['trans_date_trans_time'].dt.month.astype('object')
    data['trans_day'] = data['trans_date_trans_time'].dt.day.astype('object')
    data['trans_wday'] = data['trans_date_trans_time'].dt.day_name()
    data['is_weekend'] = data['trans_date'].dt.day_of_week > 4
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour.astype('object')

    # create `shop_net` flag
    data['shop_net'] = data['category'].str.contains('_net$', case=False)

    # create 'log_city_pop'
    data['log_city_pop'] = np.log1p(data['city_pop'])

    # coerce selected categorical columns to numeric for correlation analysis
    data['trans_hour'] = data['trans_hour'].astype(int)
    data['age_bin_cat'] = data['age_bin'].astype('category')
    data['age_bin_cat'] = data['age_bin_cat'].cat.codes

    # `amt` data biinning by quartiles
    amt_cut_bins = [0, 9.64, 47.5, 83.06, 30000]
    data['amt_bin'] = pd.cut(data['amt'], 
                             bins = amt_cut_bins, 
                             labels = ['0 - 9.64', 
                                       '9.65 - 47.50', 
                                       '47.51 - 83.06', 
                                       '83.07 - 30,000']
                        )

    # convert `amt_bin` to ordinal category
    # data['amt_bin_cat'] = data['amt_bin'].astype('category')
    # data['amt_bin_cat'] = data['amt_bin_cat'].cat.codes

    return data

def calc_distance(loc1_col, loc2_col):
    """Calculates the haversine distance between two geopoints 
    (tuples of latitude, longitude) in miles, rounded to 2 decimals.
    """
    return round((haversine(loc1_col, loc2_col, unit=Unit.MILES)), 2)

def create_distance_feature(data):
    """Create a distance feature from lat and long features with 
    geocoordinates from city_state feature using the `haversine` library

    Args:
        data (dataframe): Pandas dataframe
    """

    # retrieve geocode by city state
    df_buyer_city_state = pd.read_csv('../../data/new_home_coords.csv')

    df_buyer_city_state = df_buyer_city_state.rename(
        columns={
            "address": "city_state", 
            "lat_home":"buyer_city_state_lat", 
            "long_home":"buyer_city_state_long"
            }
        )
    
    # create tuple for distance calc
    df_buyer_city_state["buyer_city_state_location"] = df_buyer_city_state[
        ["buyer_city_state_lat", "buyer_city_state_long"]].apply(tuple, axis=1)
    
    # Merge in the city_state based location data
    data = pd.merge(
        left=data, 
        right=df_buyer_city_state,
        how="left",
        left_on="city_state",
        right_on="city_state")

    # convert lat and long to tuple
    data["home_location"] = data[["lat", "long"]].apply(tuple, axis=1)

    # distance between buyer's city-state loc and merch loc
    data["distance_home_buyer_city_state"] = data.apply(
        lambda x: calc_distance(
            x["home_location"], 
            x["buyer_city_state_location"]
        ), 
        axis=1
    )

    # drop selected columns
    data = data.drop(['first', 'last', 'street', 'merch_lat', 'merch_long', 'buyer_city_state_lat', 
                      'buyer_city_state_long', 'buyer_city_state_location', 
                      'home_location'], axis = 1)
    
    return data

# function to consolidate categories less than threshold_percent
def category_merge(data, column, threshold_percent):
    '''
    Purpose: Consolidate/merge miscellaneous categories less than threshold_percent,
    Parameters
    ----------
        df : Pandes dataframe
            Dataframe to get the columns from
        column
            Column with categories to be consolidated/merge
        threshold_percent
            Number as percent to start consolidating/merging miscellaneous categories
    '''
    series = pd.value_counts(data[column])
    mask = (series / series.sum() * 100).lt(threshold_percent)
    data[column+'_updated'] = np.where(data[column].isin(series[mask].index),'Other', data[column])

    return data

def feature_selection_prep(data, one_hot = False, drop_first = False):
    """
    Function to prep dataset for feature selection methods

    Args:
        data (dataframe): Pandas dataframe
        one_hot (bool, optional): Activate one-hot encoding preprocess. Defaults to False.
        drop_first (bool, optional): Activate drop_first argument to drop first column in one-hot encoding. 
        Defaults to False.
    """

    # drop selected columns
    cols_drop = ['trans_date_trans_time', 'merchant', 'dob', 'trans_num', 'unix_time', 
                 'trans_date', 'age_bin', 'state']
    data = data.drop(columns = cols_drop, axis = 1)

    # merge less frequent categories in 'city_state', 'city', 'zip', 'job', and 'cc_num'
    # from utils import category_merge
    category_merge(data, 'city_state', 0.01)
    category_merge(data, 'city', 0.01)
    category_merge(data, 'zip', 0.01)
    category_merge(data, 'job', 0.01)
    category_merge(data, 'cc_num', 0.01)

    # drop ['cc_unm', 'city', 'city_state', 'zip']
    data = data.drop(['cc_num', 'job', 'city', 'city_state', 'zip'], axis = 1)

    # convert `trans_wday` to numeric values
    data['trans_wday'] = data['trans_wday'].map({'Sunday': 0, 'Monday': 1, 
                                                 'Tuesday': 2, 'Wednesday': 3, 
                                                 'Thursday': 4, 'Friday': 5, 
                                                 'Saturday': 6})
    
    # coerce `trans_month`, `trans_day` to int
    cols_trans = ['trans_month', 'trans_day']
    data[cols_trans] = data[cols_trans].astype(int)

    # convert boolean columns to integer
    data['is_weekend'] = data['is_weekend'].replace({False: 0, True: 1})
    data['shop_net'] = data['shop_net'].replace({False: 0, True: 1})

    if one_hot:

        data = pd.get_dummies(data, columns=['category', 'amt_bin'], 
                              drop_first = drop_first)
    
    return data


def woe_category_encoding(train_data, test_data):

    # select categorical columns
    train_cols_cat = train_data.select_dtypes(include = 'object').columns.to_list()
    test_cols_cat = test_data.select_dtypes(include = 'object').columns.to_list()

    # drop 'category', 'state' and 'amt_bin' from cols_cat
    remove_cat = {'category', 'amt_bin'}
    train_cols_cat = [item for item in train_cols_cat if item not in remove_cat]
    test_cols_cat = [item for item in test_cols_cat if item not in remove_cat]

    # create X, y datasets
    X_train_cat = train_data[train_cols_cat]
    y_train = train_data['is_fraud']

    X_test_cat = test_data[test_cols_cat]
    y_test = test_data['is_fraud']

    # transform X_test_cat using WOE encoder
    WOE_fit = ce.WOEEncoder().fit(X_train_cat, y_train)
    X_train_trans = WOE_fit.transform(X_train_cat, y_train)
    X_test_trans = WOE_fit.transform(X_test_cat, y_test)

    # replace test categorical values with X_tes_trans encoded column values
    train_data[test_cols_cat] = X_train_trans[test_cols_cat]
    test_data[test_cols_cat] = X_test_trans[test_cols_cat]

    return train_data, test_data