import pandas as pd
import numpy as np
from django.db import connection
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import traceback
from statsmodels.tsa.arima.model import ARIMA


def fetch_data(item_code):
    print("item_code---",item_code)
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
        try:
            cursor.execute(query, [item_code])
            rows = cursor.fetchall()
            # print(rows)
        except Exception as e:
            print(e)    
    if not rows:
        print("No data fetched from database.")
        return pd.DataFrame(columns=['date', 'item_code', 'sales_qty'])    

    df = pd.DataFrame(rows, columns=['date', 'item_code', 'sales_qty'])
    # print("Initial fetched data:")
    # print(df.head())

    df['date'] = pd.to_datetime(df['date'])
    
    df['sales_qty'] = df['sales_qty'].astype(float)

    # Resample to monthly frequency and fill missing months with 0 sales
    df.set_index('date', inplace=True)
    df = df.resample('ME').sum().reset_index()
    # print(df)

    current_year = datetime.now().year
    cutoff_month = 5

    # Filter out data for the current year beyond April
    df = df[(df['date'].dt.year < current_year) | 
            ((df['date'].dt.year == current_year) & (df['date'].dt.month <= cutoff_month))]
    # print(df.shape)
    
    # if 'item_code' in df.columns:
    #         df.drop(columns=['item_code'], inplace=True)
    #     # Check DataFrame contents
        
    # print("Data types:")
    # print(df.dtypes)
        
    # df.set_index('date', inplace=True)
    df.fillna(0, inplace=True)
    excel_filename = f'filtered_data_{item_code}.xlsx'
    df.to_excel(excel_filename, index=False)


    return df


def fit_models(df):
    try:
        print(df.dtypes)   
        if 'item_code' in df.columns:
            df.drop(columns=['item_code'], inplace=True)
        
        df['sales_diff'] = df['sales_qty'].diff().dropna()
        
        df.set_index('date', inplace=True)
        
        # print("df in fit models---",df)

  


        
        # Fit SARIMAX model
        sarimax_model = SARIMAX(df['sales_diff'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),freq = 'M')
        sarimax_results = sarimax_model.fit()

        # Fit Exponential Smoothing model
        # exp_smoothing_model = ExponentialSmoothing(df['sales_diff'], trend='add', seasonal = None)
        # exp_smoothing_results = exp_smoothing_model.fit()
        
        # Fit ARIMA model
        arima_model = ARIMA(df['sales_diff'], order=(1, 1, 1))
        arima_results = arima_model.fit()

        return sarimax_results,arima_results
    
    except Exception as e:
        print("Error fitting models:", e)
        traceback.print_exc()  # Print full traceback
        return None, None
def generate_forecasts(df,sarimax_results, arima_results, periods=12):
# def generate_forecasts(df,sarimax_results,periods=12):
    # Generate forecasts
    sarimax_forecast = sarimax_results.get_forecast(steps=periods)
    arima_forecast = arima_results.get_forecast(steps=periods)

    # exp_smoothing_forecast = exp_smoothing_results.forecast(steps=periods)

    # Convert forecasts to DataFrame
    sarimax_forecast_df = sarimax_forecast.conf_int()
    sarimax_forecast_df['predicted_mean'] = sarimax_forecast.predicted_mean

    forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
    sarimax_forecast_df.index = forecast_index
    arima_forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
    arima_forecast = pd.Series(arima_forecast.predicted_mean, index=arima_forecast_index)


    # exp_smoothing_forecast.index = forecast_index

    return sarimax_forecast_df, arima_forecast
    # return sarimax_forecast_df


def plot_forecasts(df, sarimax_forecast_df, arima_forecast  ):
    
    # Ensure 'date' column is present
    if df.index.name == 'date':
        df.reset_index(inplace=True)
# def plot_forecasts(df, sarimax_forecast_df):  

    print("df in plot forecast--",df)
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(df['date'], df['sales_qty'], label='Historical Data', marker='o')

    # Plot SARIMAX forecast
    plt.plot(sarimax_forecast_df.index, sarimax_forecast_df['predicted_mean'], label='SARIMAX Forecast', marker='o', markersize=5)
    plt.fill_between(sarimax_forecast_df.index, sarimax_forecast_df.iloc[:, 0], sarimax_forecast_df.iloc[:, 1], color='k', alpha=0.1)

    # Plot Exponential Smoothing forecast
    # plt.plot(exp_smoothing_forecast.index, exp_smoothing_forecast, label='Exponential Smoothing Forecast', marker='o', markersize=5)
    
    plt.plot(arima_forecast.index, arima_forecast, label='ARIMA Forecast', marker='o', markersize=5)


    plt.legend()
    plt.title('Sales Quantity Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.grid(True)
    plt.show()


def get_data_prev_month(from_date,to_date,data):
    print(from_date[4:])

    starting_year = data['date'].iloc[0].year
    end_year = data['date'].iloc[-1].year
    
    from_month = from_date[5:7]
    to_month = to_date[5:7]
    
    print(from_month,to_month,"------from_month,to_month")
    
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
