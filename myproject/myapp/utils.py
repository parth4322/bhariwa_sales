import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from django.db import connection
import logging
import numpy as np
from scipy.stats import boxcox

logging.basicConfig(level=logging.DEBUG)

def get_database_connection():
    return connection

def fetch_data(item_code):
    with connection.cursor() as cursor:
        query = """
        SELECT
            tmain.trndate AS date,
            tpro.MCODE as item_code,
            SUM(tpro.quantity) AS sales_qty
        FROM
            TRNPROD tpro
        JOIN
            TRNMAIN tmain ON tmain.vchrno = tpro.vchrno
        WHERE
            tpro.MCODE = %s and tpro.quantity >= 0
        GROUP BY
            tmain.trndate,tpro.MCODE
        ORDER BY
            tmain.trndate;
        """
        cursor.execute(query, [item_code])
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['date', 'item_code', 'sales_qty'])
    df['date'] = pd.to_datetime(df['date'])
    df['sales_qty'] = df['sales_qty'].astype(float)

    # Resample to daily frequency and fill missing days with 0 sales
    df.set_index('date', inplace=True)
    df = df.resample('D').sum().fillna(0).reset_index()

    current_year = datetime.now().year
    cutoff_month = 4

    # Filter out data for the current year beyond April
    df = df[(df['date'].dt.year < current_year) | 
            
            ((df['date'].dt.year == current_year) & (df['date'].dt.month <= cutoff_month))]
    #print(df)
    return df

def check_stationarity(data,column='sales_qty'):
    result = adfuller(data[column].dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
    return result[1] < .05    
        
def check_seasonality(data, period=365):
    decomposition = seasonal_decompose(data['sales_qty'], model='additive', period=period)
    seasonal = decomposition.seasonal
    autocorr = acf(data['sales_qty'], nlags=period*2)
    
    # Quantifying seasonality strength
    seasonal_strength = (seasonal.var() / (seasonal.var() + decomposition.resid.var()))
    
    print("seasonal,autocorr,seasonal_strength",seasonal, autocorr, seasonal_strength)
    return seasonal_strength > 0.7


def preprocess_data(data):
    print("in preprocess_data")
    data.set_index('date', inplace=True)
    data = data.asfreq('D').fillna(0)
    print("data-----", data)

    # Initial check for stationarity
    if not check_stationarity(data):
        # First differencing
        data['sales_qty_diff'] = data['sales_qty'].diff()
        if not check_stationarity(data, 'sales_qty_diff'):
            # Log transformation
            data['sales_qty_log'] = np.log(data['sales_qty'] + 1)
            if not check_stationarity(data, 'sales_qty_log'):
                # Log differencing
                data['sales_qty_log_diff'] = data['sales_qty_log'].diff()
                if not check_stationarity(data, 'sales_qty_log_diff'):
                    # Box-Cox transformation
                    data['sales_qty_boxcox'], _ = boxcox(data['sales_qty'] + 1)
                    if not check_stationarity(data, 'sales_qty_boxcox'):
                        # Box-Cox differencing
                        data['sales_qty_boxcox_diff'] = pd.Series(data['sales_qty_boxcox']).diff()
                        if not check_stationarity(data, 'sales_qty_boxcox_diff'):
                            return None, "Data is not stationary after multiple preprocessing steps"
                        else:
                            data['sales_qty'] = data['sales_qty_boxcox_diff']
                    else:
                        data['sales_qty'] = data['sales_qty_boxcox']
                else:
                    data['sales_qty'] = data['sales_qty_log_diff']
            else:
                data['sales_qty'] = data['sales_qty_log']
        else:
            data['sales_qty'] = data['sales_qty_diff']
    
    data = data.dropna(subset=['sales_qty'])  # Drop NaNs that may have been introduced by differencing
    print("data[['sales_qty']]------", data[['sales_qty']])
    return data[['sales_qty']], None
def train_models(data):
    results = {}

    # complete_df, error_message = preprocess_data(data)
    # if error_message:
    #     return None, error_message
    complete_df = data[['date', 'sales_qty']].copy()

    complete_df.set_index('date', inplace=True)
    complete_df = complete_df.asfreq('D')
    print(len(data))
    # Train Exponential Smoothing model
    exponential_smoothing = ExponentialSmoothing(complete_df['sales_qty'], seasonal='additive', seasonal_periods=30).fit()

    # Train SARIMAX model
    sarimax_model = SARIMAX(complete_df['sales_qty'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30)).fit(disp=False)

    results['exponential_smoothing'] = exponential_smoothing
    results['sarimax'] = sarimax_model

    return results

def get_data_prev_month(from_date,to_date,data):
    print(from_date[4:])

    starting_year = data['date'].iloc[0].year
    end_year = data['date'].iloc[-1].year
    
    print(starting_year,end_year,'years-----')
    yearly_data = []
    while end_year>=starting_year:
        newFromDate = str(starting_year) + from_date[4:]
        newToDate = str(starting_year) + to_date[4:]
        newData = data.loc[(data['date'] >= newFromDate) & (data['date'] <= newToDate)]
        yearly_data.append(newData['sales_qty'].sum())
        # yearly_data.append(newData['sales_qty'])
        starting_year+=1
    print(yearly_data)
        
    return yearly_data

def forecast(item_code, models, model_type, from_date, to_date, no_of_days):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y-%m-%d")
    if isinstance(to_date, str):
        to_date = datetime.strptime(to_date, "%Y-%m-%d")

    exponential_smoothing_model = models['exponential_smoothing']
    sarimax_model = models['sarimax']

    future_dates = pd.date_range(start=from_date, periods=no_of_days, freq='D')

    if model_type == 'exponential_smoothing':
        exp_sm_forecast = exponential_smoothing_model.forecast(steps=no_of_days)
        exp_sm_forecast_df = pd.DataFrame({'date': future_dates, 'sales_qty': exp_sm_forecast})
        # print(exp_sm_forecast_df.to_dict(orient='records'))

        return exp_sm_forecast_df.to_dict(orient='records')
        # exp_sm_forecast_mean = exp_sm_forecast_df['sales_qty'].mean()
        # return {'forecasted_sales_qty_by_exponential_smoothing': exp_sm_forecast_mean}
    
    if model_type == 'sarimax':
        sarimax_forecast = sarimax_model.get_forecast(steps=no_of_days)
        sarimax_forecast_df = sarimax_forecast.summary_frame()
        sarimax_forecast_df['date'] = future_dates
        sarimax_forecast_df = sarimax_forecast_df[['date', 'mean']].rename(columns={'mean': 'sales_qty'})
        # print(sarimax_forecast_df.to_dict(orient='records'))
        return sarimax_forecast_df.to_dict(orient='records')
        # sarimax_mean = sarimax_forecast_df['mean'].mean()
        # return {'forecasted_sales_qty_by_sarimax': sarimax_mean}

    return {}

def compare_forecasts(item_code, models, from_date, to_date, no_of_days):
    exp_sm_forecast = forecast(item_code, models, 'exponential_smoothing', from_date, to_date, no_of_days)
    sarimax_forecast = forecast(item_code, models, 'sarimax', from_date, to_date, no_of_days)

    comparison = {
        'exponential_smoothing': {
            'sales_qty': exp_sm_forecast,
        },
        'sarimax': {
            'sales_qty': sarimax_forecast,
        }
    }

    return comparison


# def get_prev_data(data,from_date,to_date):
#     data = data.loc[()]
