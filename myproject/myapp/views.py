from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from rest_framework.decorators import api_view
from django.http import HttpResponse 
import random
import requests
import seaborn as sns
from django.http import JsonResponse
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from statsmodels.tsa.stattools import adfuller


from .utils import fetch_data,fit_models,generate_forecasts,plot_forecasts,get_data_prev_month,sarimax_grid_search

@api_view(['GET'])
def plot_sales_qty(request, item_code):
    from_date = request.GET.get('from_date')
    to_date = request.GET.get('to_date')
    no_of_days = int(request.GET.get('no_of_days', 28))

    # Fetch and process data
    df = fetch_data(item_code)

    # Fit models
    sarimax_results, arima_results = fit_models(df)
    # sarimax_results = fit_models(df)

    # Generate forecasts
    sarimax_forecast_df, arima_forecast_df = generate_forecasts(df,sarimax_results, arima_results, periods=no_of_days)
    # sarimax_forecast_df = generate_forecasts(df,sarimax_results,periods=no_of_days)
    # Plot forecasts
    plot_forecasts(df, sarimax_forecast_df, arima_forecast_df)
    # plot_forecasts(df, sarimax_forecast_df)
    
    
    sarimax_forecast_dict = sarimax_forecast_df.to_dict()
    print("sarimax_forecast_dict----",sarimax_forecast_dict)
          
    arima_forecast_dict = arima_forecast_df.to_dict()
    print("arima_forecast_dict-----",arima_forecast_dict)
    
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)

    
    # filtered_sarima_forecast = {str(k): v for k, v in sarimax_forecast_dict.items() if from_date <= pd.to_datetime(k) <= to_date}
    filtered_arima_forecast = {str(k): v for k, v in arima_forecast_dict.items() if from_date <= pd.to_datetime(k) <= to_date}

    
    # from_date = pd.to_datetime(from_date)
    # to_date = pd.to_datetime(to_date)
    
    
    # filtered_sarima_forecast = {str(k): v for k, v in sarimax_forecast_dict.items() if from_date <= k <= to_date}
    # filtered_sarima_forecast = {k: v for k, v in sarimax_forecast_dict.items()}

    
    
    # filtered_arima_forecast = {str(k): v for k, v in arima_forecast_dict.items() if from_date <= k <= to_date}
    filtered_arima_forecast = {k: v for k, v in filtered_arima_forecast.items()}
    
    
    



    return HttpResponse({'arima_forecast':filtered_arima_forecast})


@api_view(['GET'])
def predict_sales_qty(request, item_code):
    from_date = request.GET.get('from_date')
    to_date = request.GET.get('to_date')
    no_of_days = int(request.GET.get('no_of_days', 28))

    # Fetch and process data
    df = fetch_data(item_code)
    
    # print("df-----",df)
    
    # Extract month from from_date and to_date
    from_month = pd.to_datetime(from_date).month
    to_month = pd.to_datetime(to_date).month
    
    prevsale = get_data_prev_month(from_date,to_date,df)
    
    print("prevsale----",prevsale)
    print("data type of prev sale-----", type(prevsale))
    increase_trend = 0
    decrease_trend = 0
    adjusted_forecast =0 
    if prevsale[-2] >= 0 and prevsale[-1] >= 0:
        if prevsale[-1] > prevsale[-2]:
            increase_trend = (prevsale[-1]  - prevsale[-2] ) / 2
        else:
        # trend = 0
            decrease_trend = (prevsale[-2] - prevsale[-1])/2

    # Seasonal naive forecast for May 2025
    # if prevsale[-1] > 0:
    #     seasonal_forecast = prevsale[-1]
    # else:
    #     seasonal_forecast = 0

    # Adjust forecast considering the trend
    if increase_trend:
        adjusted_forecast = prevsale[-1] + increase_trend
    elif decrease_trend :
        if  decrease_trend > 0:
            adjusted_forecast = (prevsale[-1] - decrease_trend)
            print("in elif adjusted_forecast----",adjusted_forecast)
            if adjusted_forecast > 0:
                adjusted_forecast = adjusted_forecast
            else:
                print("in else----")
                if prevsale[-1] == 0:
                    adjusted_forecast = prevsale[-2]
                else:
                    adjusted_forecast = prevsale[-1]
                        
                    
        
        
        
    
    # Ensure forecast is non-negative
    
    
    ## Sarima model
    
    # print("Infinite values check:")
    # print(df.isin([np.inf, -np.inf]).sum())
    
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # df.dropna(inplace=True)


    
    result = adfuller(df['sales_qty'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    
    # Parameters for SARIMAX model
    
    df['sales_diff'] = df['sales_qty'].diff().dropna()
    df.set_index('date', inplace=True)
    # print("df ----",df)
    
    # result = adfuller(df['sales_diff'])
    # print('ADF Statistic for differenced:', result[0])
    # print('p-value for differenced:', result[1])



    p, d, q = 1,1,1
    P, D, Q, s = 1, 1, 1, 12

    # Fit the SARIMA model
    model = SARIMAX(df['sales_diff'], 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s),
                    trend='c')
                    # enforce_stationarity=False,
                    # enforce_invertibility=False)
    sarima_results = model.fit(disp=False)
    
    residuals = sarima_results.resid
    
    # constant_term = sarima_results.params['const']
    
    # print("constant_term-----",constant_term)

    
    print(sarima_results.summary())
    
    intercept = sarima_results.params.get('intercept', 0)
    phi_1 = sarima_results.params.get('ar.L1', 0)
    theta_1 = sarima_results.params.get('ma.L1', 0)
    gamma = sarima_results.params.get('ar.S.L12', 0)
    Theta_1 = sarima_results.params.get('ma.S.L12', 0)
    
    delta_y_t1 = df['sales_diff'][-1]  # Δy_{t-1}
    delta_y_ts = df['sales_diff'][-s]  # Δy_{t-s}
    
    epsilon_t1 = residuals.iloc[-1]
    epsilon_ts = residuals.iloc[-s]
    # Calculate the forecasted differenced value for the to_date
    delta_y_forecast = (intercept +
                        (phi_1 * delta_y_t1) +
                        (theta_1 * 0) +  # ε_{t-1} is assumed to be 0 for direct calculation
                        (gamma * delta_y_ts) +
                        (Theta_1 * 0))  # ε_{t-s} is assumed to be 0 for direct calculation
    
    print("delta_y_forecast------",delta_y_forecast)
    
    # Last observed sales quantity
    last_sales = df['sales_qty'][-1]
    
    # Forecasted sales for the to_date
    forecast_value = 0
    # forecast_value = last_sales + delta_y_forecast
    forecast_value = delta_y_forecast
    # forecast_value = delta_y_forecast
    
    # Ensure non-negative forecast
    forecast_value = max(forecast_value, 0)
    
    print("intercept:", intercept)
    print("phi_1:", phi_1)
    print("theta_1:", theta_1)
    print("gamma:", gamma)
    print("Theta_1:", Theta_1)
    print("delta_y_t1:", delta_y_t1)
    print("delta_y_ts:", delta_y_ts)
    print("epsilon_t1:", epsilon_t1)
    print("epsilon_ts:", epsilon_ts)
    print("delta_y_forecast:", delta_y_forecast)
    print("Forecasted sales quantity for May 2025:", forecast_value)
    


    # Forecasting for the next year (e.g., May 2025)
    # forecast_steps = 12
    # forecast = sarima_results.get_forecast(steps=forecast_steps)
    # forecast_df = forecast.conf_int()
    # forecast_df['predicted_mean'] = forecast.predicted_mean
    # print("forecast_df------",forecast_df)

    # forecast_df['predicted_mean'] = forecast_df['predicted_mean'].apply(lambda x: max(x, 0))

    # # print("forecast_df------",forecast_df['forecast'])
    
    # # print("df.index------",df.index)
    # last_date = pd.to_datetime(df.index[-1])
    
    # forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
    # forecast_df.index = forecast_index
    # # print("forecast_index-----",forecast_index)


    # # Create a date range for the forecast steps
    

    # # Extract the forecast for the desired month (e.g., May 2025)
    # # forecast_date = pd.to_datetime(from_date)
    # # print("forecast_date--------",forecast_date)
    
    # if to_date in forecast_df.index:
    #     forecast_value = forecast_df.loc[to_date]['predicted_mean']
    #     forecast_value = max(forecast_value, 0)
        
    # else:
    #     forecast_value = 0  # Default to 0 if the forecast date is not available
    
    # p = d = q = range(0, 3)
    # pdq = list(itertools.product(p, d, q))
    # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    # param_grid = list(itertools.product(pdq, seasonal_pdq))

# Example usage
    # best_params = sarimax_grid_search(df['sales_diff'], param_grid)
    # best_order, best_seasonal_order = best_params
    # print("best_params-----",best_params)
    # sarimax_model = SARIMAX(df['sales_diff'], order=[1,1,1], seasonal_order=[1,1,1,12])
    # sarimax_fit = sarimax_model.fit(disp=False)
    
    # from prophet import Prophet

    # # prophet_data = df.rename(columns={'date': 'ds', 'sales_diff': 'y'})
    # prophet_data = df.reset_index().rename(columns={'date': 'ds', 'sales_diff': 'y'})

    # prophet_model = Prophet()
    # prophet_model.fit(prophet_data)
    
    # # Generate future dates
    # # future_dates = prophet_model.make_future_dataframe(periods=12, freq='M')
    # latest_date_in_dataset = df.index.max() 
    # next_month = latest_date_in_dataset + pd.DateOffset(months=1)  # Add one month to the latest date

    # # Get the latest date in your dataset
    # future_dates = pd.date_range(start=next_month, periods=12, freq='M').to_frame(index=False)
    # future_dates.columns = ['ds']

    # print("future_dates-------",future_dates)
    # # Get Prophet forecast
    # prophet_forecast = prophet_model.predict(future_dates)
    # prophet_forecast = prophet_forecast[['ds', 'yhat']].set_index('ds')
    
    # print("prophet_forecast-----",prophet_forecast)

    # # Get SARIMAX forecast
    # sarimax_forecast = sarimax_fit.get_forecast(steps=12).predicted_mean
    # print("sarimax_forecast-------",sarimax_forecast)
    # latest_date_in_dataset = df.index.max()  # Get the latest date in your dataset
    # # future_dates = pd.date_range(start=latest_date_in_dataset, periods=12, freq='M')
    # future_dates_flat = np.array(future_dates[-12:]).flatten()

    # sarimax_forecast = pd.DataFrame(sarimax_forecast, index=future_dates_flat[-12:], columns=['yhat'])
    # # print("sarimax forecast after combining--",sarimax_forecast)

    # # Combine forecasts
    # combined_forecast = (prophet_forecast + sarimax_forecast) / 2
    # print("combined_forecast before-----",combined_forecast)

    # Ensure no negative or zero values (except for items with last two years of zero sales)
    # for index, value in combined_forecast.iterrows():
    #     if value['yhat'] <= 0 and (df['sales_diff'][-24:].sum() != 0):
    #         combined_forecast.at[index, 'yhat'] = df['sales_diff'].mean()  # or another replacement strategy

    # print(combined_forecast)
    
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['date'], df['sales_qty'], label='Historical Data')
    # plt.plot(combined_forecast.index, combined_forecast['yhat'], label='Combined Forecast', color='orange')
    # plt.fill_between(combined_forecast.index, combined_forecast['yhat'] - 1.96*combined_forecast['yhat'].std(), 
    #                 combined_forecast['yhat'] + 1.96*combined_forecast['yhat'].std(), color='orange', alpha=0.3)
    # plt.xlabel('Date')
    # plt.ylabel('Sales Quantity')
    # plt.title('Combined Forecast from SARIMAX and Prophet')
    # plt.legend()
    # plt.show()




        
        # Ensure the forecast is non-negative
    # Return the result
    # response_data = {
    #     "adjusted_forecast": forecast_value
    # }
    
    # return JsonResponse(response_data) 


    # Return the result
    
    # print("adjusted_forecast------",adjusted_forecast)
    # return JsonResponse({"forecast_seasonal_naive": adjusted_forecast,"sarimax_forecast":forecast_value}) 
    return JsonResponse({"forecast_seasonal_naive": adjusted_forecast,"sarimax_forecast":forecast_value}) 

    # Calculate the seasonal index for the specified month
    # monthly_sales = df[df['date'].dt.month == to_month]['sales_qty']
    # print("monthly_sales-------",monthly_sales)
    # seasonal_index = monthly_sales.mean()
    # print("seasonal_index-----",seasonal_index)

    # # Adjusted forecast for the specified month
    # adjusted_forecast_naive = seasonal_index + prevsale[-1]
    # print("adjusted_forecast_naive-----",adjusted_forecast_naive)
    
    # ## moving average
    # df = df[(df['date'] >= df['date'].max() - pd.DateOffset(years=2))]

    # # Calculate the seasonal index for May
    # # may_sales = df[df['date'].dt.month == to_month]['sales_qty']
    # # prevsalemomth = df[df['date'].dt.month == to_month - 1]['sales_qty']
    # prevsale1 = prevsale[-1]
    # backprevsale = prevsale[-2]
    # trend_adjustment = (prevsale1 - backprevsale) / 2  # Calculate the trend adjustment
    
    # # Adjusted forecast for the specified month considering trend
    # if trend_adjustment > 0:  # If trend is increasing
    #     adjusted_forecast_avg = prevsale[-1] + trend_adjustment
    #     print("adjusted_forecast_avg--------",adjusted_forecast_avg)
    # else:  # If trend is decreasing
    #     adjusted_forecast_avg =  prevsale[-1] - trend_adjustment
    #     print("adjusted_forecast_avg in else-----",adjusted_forecast_avg)


    # Return the result
    # return HttpResponse({"adjusted_forecast_naive": adjusted_forecast_naive,"adjusted_forecast_avg":adjusted_forecast_avg})


