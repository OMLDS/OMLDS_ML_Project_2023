import pandas as pd
import numpy as np
from haversine import haversine, Unit

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

