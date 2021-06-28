import pandas as pd
import numpy as np
import os
from env import host, username, password


# Acquire

# Get Connection
def get_connection(db, username=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup SQL db.
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

# Get Zillow Data
def zillow_data():
    '''
    This function reads the telco_churn data from the Codeup db into a df
    '''
    
    # Create SQL query
    sql_query = '''SELECT *                    
    FROM predictions_2017
    JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE latitude IS NOT Null
        AND longitude IS NOT Null'''

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))

    return df

# Cache'ing Zillow Data
def cache_zillow_data(cached=False):
    '''
    This function reads in Zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_df.csv') == False:

        # Read fresh data from db into a DataFrame
        df = zillow_data()

        # Cache data
        df.to_csv('zillow_data.csv')

    else:

        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        


