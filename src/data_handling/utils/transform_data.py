import pandas as pd
import numpy as np

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