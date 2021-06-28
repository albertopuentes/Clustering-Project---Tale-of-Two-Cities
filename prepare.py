import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

    
def handle_missing_values(df, col_perc, row_perc):
    ''' 
        take in a dataframe and percent null parameters, returns df eliminating nulls by inputed threshold
    '''
    col_thresh = int(round(col_perc*df.shape[1],0)) # calc column threshold
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    row_thresh = int(round(row_perc*df.shape[0],0))  # calc row threshhold
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df    
    

def prep_zillow(df):
    
    ''' Single Unit Properties excluding shared structures'''
    # Restrict propertylandusedesc to those of single unit
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
            (df.propertylandusedesc == 'Mobile Home') |
          (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes')]
    
    # remove outliers in bed count, bath count, and area to better target single unit properties
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])
    
    # dropping cols/rows where more than half of the values are null
    df = handle_missing_values(df, col_perc = .5, row_perc = .5)
    
    # dropping the columns with 17K missing values; too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt', 'heatingorsystemdesc'])
    
    ##### transactiondate outlier #### 
    
    # address date outlier by replacing with mode
    df['transactiondate'] = df.transactiondate.replace('2018-05-25', '2017-06-30')
    
    # Convert transactiondate to datetime
    df['transactiondate'] = pd.to_datetime(df.transactiondate)
    
    # drop assessment year as it is one single year
    df = df.drop(columns=(['assessmentyear']))
    
    #dropping the id columns, as parcelid is a unique identifier
    df= df.drop(columns=('id'))
    
    # Convert logerror to absolute value
    df['logerror'] = df.logerror.abs()
    
    return df
    
    ### FIPS conversion ####
    
def fips_county(df):
    '''referencing FIPS data to create a county feature'''
    # convert fips to integer to drop decimals before converting to string
    df['fips'] = df['fips'].astype(int)
    # convert fips data to string so that we can split
    df['fips'] = df['fips'].astype('str')
    # Create County column by stripping state reference location(last three digits)
    df['county'] = df['fips'].str[1:]
    # Replace numerics with labels
    df['county'] = df['county'].replace({'037':'Los Angeles', '059':'Orange', '111':'Ventura'})
    
    return df
    
    
    ### tax_rate conversion ###
def taxrate_calc(df)    
    '''caclulates tax rate column based on tax amount and totol value'''
    df['tax_rate'] = round((df.taxamount / df.taxvaluedollarcnt), 3)
    # drop the rows with tax rates in excess of 0.2
    df = df.drop(df[df['tax_rate']>0.2].index)
    
    return df
    
    ### Split Date ### 

    
### split continuous data ###
def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test

def data_split_impute(df)
    '''organize features and split data into train, validate and test and impute missing observations'''
    
    # Organize Discrete & Continous features
    discrete_col = ['calculatedbathnbr', 'fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock']
    cont_col = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
    
    # Split data
    train, validate, test = split_continuous(df)
    
    # impute mode on discrete columns in split data
    imputer = SimpleImputer(strategy='most_frequent')
    train[discrete_col] = imputer.fit_transform(train[discrete_col])
    validate[discrete_col] = imputer.transform(validate[discrete_col])
    test[discrete_col] = imputer.transform(test[discrete_col])
    
    # impute mean on continuous columns in split data
    imputer = SimpleImputer(strategy='mean')
    train[cont_col] = imputer.fit_transform(train[cont_col])
    validate[cont_col] = imputer.transform(validate[cont_col])
    test[cont_col] = imputer.transform(test[cont_col])
    
    return train, validate, test


def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df


def min_max_scale(train, validate, test, cols_to_scale):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(train[cols_to_scale])

    # scale train, validate, test using the mins and maxes stored in the scaler derived from train.
    
    train_scaled_array = scaler.transform(train[cols_to_scale])
    validate_scaled_array = scaler.transform(validate[cols_to_scale])
    test_scaled_array = scaler.transform(test[cols_to_scale])

    # convert arrays to dataframes
    train_scaled = pd.DataFrame(train_scaled_array, columns=cols_to_scale).set_index([train.index.values])

    validate_scaled = pd.DataFrame(validate_scaled_array, columns=cols_to_scale).set_index([validate.index.values])

    test_scaled = pd.DataFrame(test_scaled_array, columns=cols_to_scale).set_index([test.index.values])

    return train_scaled, validate_scaled, test_scaled