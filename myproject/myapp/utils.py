import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from django.db import connection
import logging

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
            tpro.MCODE = %s  
        GROUP BY
            tmain.trndate, tpro.MCODE
        ORDER BY
            tmain.trndate
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
    print(df)
    return df

def check_stationarity(data):
    result = adfuller(data['sales_qty'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')

def check_seasonality(data, period=365):
    decomposition = seasonal_decompose(data['sales_qty'], model='additive', period=period)
    seasonal = decomposition.seasonal
    autocorr = acf(data['sales_qty'], nlags=period*2)
    
    # Quantifying seasonality strength
    seasonal_strength = (seasonal.var() / (seasonal.var() + decomposition.resid.var()))
    
    return seasonal, autocorr, seasonal_strength

def train_models(data):
    results = {}

    complete_df = data[['date', 'sales_qty']].copy()
    complete_df.set_index('date', inplace=True)

    # Train Exponential Smoothing model
    exponential_smoothing = ExponentialSmoothing(complete_df['sales_qty'], seasonal='additive', seasonal_periods=30).fit()

    # Train SARIMAX model
    sarimax_model = SARIMAX(complete_df['sales_qty'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30)).fit(disp=False)

    results['exponential_smoothing'] = exponential_smoothing
    results['sarimax'] = sarimax_model

    return results

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
