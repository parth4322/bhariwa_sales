# myapp/utils.py
import pandas as pd
#import pyodbc
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import matplotlib.pyplot as plt
from datetime import datetime
from django.db import connection


def get_database_connection():
    return connection


    # plt.figure(figsize=(12, 6))
    # plt.plot(data['sales_qty'], marker='o', linestyle='-', color='b')
    # plt.title('Sales Quantity Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Sales Quantity')
    # plt.grid(True)
    # plt.show()
    
def check_stationarity(data):
    result = adfuller(data['sales_qty'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')   
        
def check_seasonality(data, period=12):
    decomposition = seasonal_decompose(data['sales_qty'], model='additive', period=period)
    seasonal = decomposition.seasonal
    autocorr = acf(data['sales_qty'], nlags=period*2)
    
    # Quantifying seasonality strength
    seasonal_strength = (seasonal.var() / (seasonal.var() + decomposition.resid.var()))
    
    return seasonal, autocorr, seasonal_strength         
   


def fetch_data(item_code):
    with connection.cursor() as cursor:
        #print("in utils-",item_code)

        query = """
        SELECT
            YEAR(tmain.trndate) AS year,
            MONTH(tmain.trndate) AS month,
            tpro.MCODE as item_code,
            SUM(tpro.quantity) AS sales_qty
        FROM
            TRNPROD tpro
        JOIN
            TRNMAIN tmain ON tmain.vchrno = tpro.vchrno
        WHERE
                tpro.MCODE = %s  
        GROUP BY
            YEAR(tmain.trndate), MONTH(tmain.trndate), tpro.MCODE
        ORDER BY
            tpro.MCODE, YEAR(tmain.trndate), MONTH(tmain.trndate)
        """
        cursor.execute(query,[item_code])
        rows = cursor.fetchall()
        #print(rows)
        
    

    df = pd.DataFrame(rows, columns=['year', 'month', 'item_code', 'sales_qty'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['sales_qty'] = df['sales_qty'].astype(float)
    #print(df)
    # return df
    #connection.close()
    # print(df)
    # return df

    current_year = datetime.now().year
    cutoff_month = 4

    # Generate a DataFrame with all months for each year in the data
    all_months = pd.DataFrame({
        'month': range(1, 13)
    })

    all_years = pd.DataFrame({
        'year': df['year'].unique()
    })
    
    all_item_codes = pd.DataFrame({
        'item_code': [item_code]
    })


    # Create a Cartesian product of all_years and all_months
    all_combinations = all_years.merge(all_months, how='cross').merge(all_item_codes, how='cross')

    # Merge with the original data to fill missing months with 0 sales_qty
    complete_df = all_combinations.merge(df, on=['year', 'month', 'item_code'], how='left').fillna(0)

    # Convert sales_qty back to float (merge may convert it to object)
    complete_df['sales_qty'] = complete_df['sales_qty'].astype(float)

    # Filter out data for the current year beyond April
    complete_df = complete_df[(complete_df['year'] < current_year) | ((complete_df['year'] == current_year) & (complete_df['month'] <= cutoff_month))]

    # Ensure the order is correct
    complete_df = complete_df.sort_values(by=['item_code', 'year', 'month'])

    print(complete_df)
    return complete_df

def check_stationarity(data):
    result = adfuller(data['sales_qty'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
        
def check_seasonality(data, period=12):
    decomposition = seasonal_decompose(data['sales_qty'], model='additive', period=period)
    seasonal = decomposition.seasonal
    autocorr = acf(data['sales_qty'], nlags=period*2)
    
    # Quantifying seasonality strength
    seasonal_strength = (seasonal.var() / (seasonal.var() + decomposition.resid.var()))
    
    return seasonal, autocorr, seasonal_strength        


def get_data_prev_month(item_code,from_date,to_date,data):
    try:
        start_month =datetime.strptime(from_date, '%Y-%m-%d').month
        #end_month=datetime.month(to_date)
        
        prevData = data.loc[(data['month'] == start_month)]
        
        return prevData
    except Exception as e:
        print(e)
        return {'statusCode':-1,'message':'something went wrong'}
    
def train_models(data):
    results = {}

    for item_code, group in data.groupby('item_code'):
        complete_df = group[['year', 'month', 'sales_qty']].copy()
        complete_df['date'] = pd.to_datetime(complete_df[['year', 'month']].assign(day=1))
        complete_df = complete_df[['date', 'sales_qty']]
        
        complete_df.set_index('date', inplace=True)

        
        exponential_smoothing = ExponentialSmoothing(complete_df['sales_qty'], seasonal='additive', seasonal_periods=12).fit()
        # print(df)
        # df.columns = ['ds', 'y']
        # print("Training data for item_code:", item_code)
        # print(df)

        # # Train Prophet model
        # prophet_model = Prophet(
        #             changepoint_prior_scale=0.08, 
        #             seasonality_prior_scale=12.0,
        #             seasonality_mode='multiplicative',  # Adjust seasonality mode if needed
        #             yearly_seasonality=True
        #             )

        # prophet_model.fit(df)
        
        

        # # Train SARIMAX model
        # df.set_index('ds', inplace=True)

        # sarimax_model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        sarimax_model = SARIMAX(complete_df['sales_qty'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)

        results[item_code] = {
            'exponential_smoothing': exponential_smoothing,
            'sarimax': sarimax_model
        }

    return results

def forecast(item_code, models,model_type, from_date, to_date,  no_of_days):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y-%m-%d")
    if isinstance(to_date, str):
        to_date = datetime.strptime(to_date, "%Y-%m-%d")
    # if isinstance(next_date, str):
    #     next_date = datetime.strptime(next_date, "%Y-%m-%d")

    exponential_smoothing_model = models[item_code]['exponential_smoothing']
    sarimax_model = models[item_code]['sarimax']

    future_dates = pd.date_range(start=from_date, periods=no_of_days, freq='D')
    # future_df = pd.DataFrame({'date': future_dates})
    
    # future_dates = pd.date_range(start=from_date, end=to_date, freq='MS')

    # future_df['ds'] = future_df['ds'] + pd.offsets.MonthBegin(1)

    # Make forecasts with the trained Prophet model
    # if model_type == 'prophet':
    #     prophet_forecast = prophet_model.predict(future_df)
    #     prophet_forecast_dict = prophet_forecast[['ds', 'yhat']].to_dict(orient='records')
    #     prophet_mean = prophet_forecast['yhat'].mean()

    #     return {'forecasted sales quantity by prophet': int(prophet_mean)}
    if model_type == 'exponential_smoothing':
        exp_sm_forecast = exponential_smoothing_model.forecast(steps=no_of_days)
       # print("exp_sm_forecast------",exp_sm_forecast)
        exp_sm_forecast_df = pd.DataFrame({'date': future_dates, 'sales_qty': exp_sm_forecast})
        exp_sm_forecast_mean = exp_sm_forecast_df['sales_qty'].mean()
        return {'forecasted_sales_qty_by_exponential_smoothing': (exp_sm_forecast_mean)}
    
        # exp_sm_forecast = exponential_smoothing_model.forecast(steps=len(future_dates))
        # exp_sm_forecast_df = pd.DataFrame({'date': future_dates, 'sales_qty': exp_sm_forecast})
        # exp_sm_forecast_df['year'] = exp_sm_forecast_df['date'].dt.year
        # exp_sm_forecast_df['month'] = exp_sm_forecast_df['date'].dt.month
        # return exp_sm_forecast_df[['year', 'month', 'sales_qty']].to_dict(orient='records')
        
    if model_type == 'sarimax':
        
    # Make forecasts with the trained SARIMAX model
        sarimax_forecast = sarimax_model.get_forecast(steps=no_of_days)
        print("sarimax_forecast-------",sarimax_forecast)
        sarimax_forecast_df = sarimax_forecast.summary_frame()
        print(sarimax_forecast_df)
        sarimax_forecast_df['date'] = future_dates
        sarimax_forecast_dict = sarimax_forecast_df[['date', 'mean']].rename(columns={'mean': 'sales_qty'}).to_dict(orient='records')
        sarimax_mean = sarimax_forecast_df['mean'].mean()

        
            
        return {' Forecasted sales_qty by sarimax':(sarimax_mean)}
        # sarimax_forecast = sarimax_model.get_forecast(steps=len(future_dates))
        # sarimax_forecast_df = sarimax_forecast.summary_frame()
        # sarimax_forecast_df['date'] = future_dates
        # sarimax_forecast_df['year'] = sarimax_forecast_df['date'].dt.year
        # sarimax_forecast_df['month'] = sarimax_forecast_df['date'].dt.month
        # return sarimax_forecast_df[['year', 'month', 'mean']].rename(columns={'mean': 'sales_qty'}).to_dict(orient='records')

    return{}


    # prophet_forecast_dict = prophet_forecast[['ds', 'yhat']].to_dict(orient='records')
    # sarimax_forecast_dict = sarimax_forecast_df[['ds', 'mean']].rename(columns={'mean': 'yhat'}).to_dict(orient='records')


    # # Combine the forecasts into a dictionary
    # forecast_dict = {
    #     'prophet': prophet_forecast_dict,
    #     'sarimax': sarimax_forecast_dict
    # }
    # # Combine the forecasts into a dictionary
    # # forecast_dict = {
    # #     'prophet': prophet_forecast[['ds', 'yhat']],
    # #     'sarimax': sarimax_forecast[['yhat']]
    # # }
    # print(forecast_dict)
    # return forecast_dict
    
def compare_forecasts(item_code, models, from_date, to_date, no_of_days):
    exp_sm_forecast = forecast(item_code, models, 'exponential_smoothing', from_date, to_date, no_of_days)
    sarimax_forecast = forecast(item_code, models, 'sarimax', from_date, to_date, no_of_days)

    comparison = {
        'exponential_smoothing': {
            'sales_qty':  exp_sm_forecast,
            # 'forecasted_values': exp_sm_forecast['forecasted_values']
        },
        'sarimax': {
            'sales_qty': sarimax_forecast,
            # 'forecasted_values': sarimax_forecast['forecasted_values']
        },
        #'difference': abs(exp_sm_forecast['mean_sales_qty'] - sarimax_forecast['mean_sales_qty'])
    }

    return comparison
