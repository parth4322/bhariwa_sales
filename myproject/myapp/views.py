from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from rest_framework.decorators import api_view
from django.http import HttpResponse 
import random
import requests
import seaborn as sns
from django.http import JsonResponse
from .utils import fetch_data,fit_models,generate_forecasts,plot_forecasts,get_data_prev_month

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
    
    # Extract month from from_date and to_date
    from_month = pd.to_datetime(from_date).month
    to_month = pd.to_datetime(to_date).month
    
    prevsale = get_data_prev_month(from_date,to_date,df)
    
    print("prevsale----",prevsale)
    print("data type of prev sale-----", type(prevsale))
    
    if prevsale[-2] > 0 and prevsale[-1] > 0:
        trend = (prevsale[-1]  - prevsale[-2] ) / 2
    else:
        trend = 0

    # Seasonal naive forecast for May 2025
    if prevsale[-1] > 0:
        seasonal_forecast = prevsale[-1]
    else:
        seasonal_forecast = 0

    # Adjust forecast considering the trend
    adjusted_forecast = seasonal_forecast + trend
    adjusted_forecast = max(adjusted_forecast, 0) 
    
    # Ensure forecast is non-negative

    # Return the result
    
    print("adjusted_forecast------",adjusted_forecast)
    return JsonResponse({"adjusted_forecast": adjusted_forecast}) 

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


