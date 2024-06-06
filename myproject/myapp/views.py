from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from rest_framework.decorators import api_view
from django.http import HttpResponse 
import random
import requests
import seaborn as sns
from django.http import JsonResponse
from .utils import fetch_data,fit_models,generate_forecasts,plot_forecasts

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
    
    # from_date = pd.to_datetime(from_date)
    # to_date = pd.to_datetime(to_date)
    
    
    # filtered_sarima_forecast = {str(k): v for k, v in sarimax_forecast_dict.items() if from_date <= k <= to_date}
    # filtered_sarima_forecast = {k.strftime('%Y-%m-%d'): v for k, v in sarimax_forecast_dict.items()}

    
    
    # filtered_arima_forecast = {str(k): v for k, v in arima_forecast_dict.items() if from_date <= k <= to_date}
    # filtered_arima_forecast = {k.strftime('%Y-%m-%d'): v for k, v in filtered_arima_forecast.items()}
    
    
    



    return HttpResponse({"sarimax_forecast":sarimax_forecast_dict,'arima_forecast':arima_forecast_dict})


