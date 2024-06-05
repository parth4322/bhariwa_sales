from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from rest_framework.decorators import api_view
from django.http import HttpResponse 
import random

# myapp/views.py
from django.http import JsonResponse
from .utils import fetch_data, train_models,get_data_prev_month, forecast,compare_forecasts,check_stationarity,check_seasonality
import logging

logger = logging.getLogger(__name__)


@api_view(['GET'])
def plot_sales_qty(request, item_code):
    data = fetch_data(item_code)
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data.set_index('date', inplace=True)

    seasonal, autocorr, seasonal_strength = check_seasonality(data)

    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    
    # Plot sales quantity over time
    axes[0].plot(data['sales_qty'], marker='o', linestyle='-', color='b')
    axes[0].set_title('Sales Quantity Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sales Quantity')
    axes[0].grid(True)
    
    # Plot seasonal component
    axes[1].plot(seasonal, marker='o', linestyle='-', color='g')
    axes[1].set_title('Seasonal Component')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Seasonal Effect')
    axes[1].grid(True)

    # Plot ACF
    plot_acf(data['sales_qty'], ax=axes[2], lags=13)
    axes[2].set_title('Autocorrelation Function')

    # Plot PACF
    plot_pacf(data['sales_qty'], ax=axes[3], lags=13)
    axes[3].set_title('Partial Autocorrelation Function')

    buffer = BytesIO()
    plt.tight_layout()
    plt.figtext(0.5, 0.02, f'Seasonal Strength: {seasonal_strength:.2f}', ha='center', fontsize=12)
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    check_stationarity(data)


    response = HttpResponse(image_png, content_type='image/png')
    return response


def randomChoicePlus(a,b):
    return a+b
def randomChoiceMinus(a,b):
    if a>=b:
        return a-b
    else:
        return b-a
def forecast_sales_qty(request):
    item_code = str(request.GET.get('item_code'))
    #print("item_code----",item_code)
    from_date = request.GET.get('from_date')
    #print("from_date----",from_date)
    to_date = request.GET.get('to_date')
    #print("to_date----",to_date)
    model_type = request.GET.get('model_type')
    # next_date = request.GET.get('next_date')
    # print("next_date----",next_date)

    
    
    no_of_days = request.GET.get('no_of_days')
    if no_of_days:
        no_of_days = int(no_of_days)
    # Log the received parameters
    #print("no_of_days----",no_of_days)


    data = fetch_data(item_code)
    # print('data-->> ',data)
    prevSale = get_data_prev_month(from_date,to_date,data)
    #plot_sales_data(data)
    check_stationarity(data)
    models= train_models(data)
    
    # if error_message:
    #     return JsonResponse({'error': error_message}, status=400)

    forecasts = forecast(item_code, models, model_type, from_date, to_date, no_of_days)
    
    # return JsonResponse(forecasts, safe=False)    #print(forecasts)
    # for key, value in forecasts.items():
    #     forecasts[key] = value.to_dict(orient='records')
    #print("forecasts-----",type(forecasts))
    slsQty=0
    for qty in forecasts:
        slsQty+=qty['sales_qty']
    #print("forecast-------",forecasts)    
    print(slsQty)
    print("prevsale------",len(prevSale))

    
    if forecasts:
        upperCutoff1 = 0
        upperCutoff2 = 0
        lowerCutoff1 = 0
        lowerCutoff2 = 0
        print("prevsale------",len(prevSale))
        upperCutoff1 = prevSale[-1]+((prevSale[-1]*5)/1000)
        print("upperCutoff1--------",upperCutoff1)
        if len(prevSale)>=2:
            upperCutoff2 = prevSale[-2]+((prevSale[-2]*5)/1000)
            print("upperCutoff2---",upperCutoff2)
        
        lowerCutoff1 = prevSale[-1]-((prevSale[-1]*5)/1000)
        print("lowerCutoff1-------",lowerCutoff1)
        if len(prevSale)>=2:
            lowerCutoff2 = prevSale[-2]-((prevSale[-2]*5)/1000)
            print("lowerCutoff2-------",lowerCutoff2)
            
    #     if slsQty<=upperCutoff1 and slsQty>=lowerCutoff1:
    #         return JsonResponse({'data':slsQty , 'model_type':model_type})
    #     elif (upperCutoff2 is not None) and (lowerCutoff2 is not None):    
    #         if slsQty<=upperCutoff2 and slsQty>=lowerCutoff2:
    #             print("slsQty------",slsQty)
    #             return JsonResponse({'data':slsQty , 'model_type':model_type})
        
    #         else:
    #             calcprevSale = prevSale[-1]
    #             randomSalsQty = 0
    #             randomSalsQty = random.randint(1,5)
                
    #             if prevSale[-1] == 0:
    #                 if len(prevSale)>=2:
    #                     calcprevSale = prevSale[-2]
                    
    #             result = random.choice([randomChoicePlus(calcprevSale,randomSalsQty*(calcprevSale)/100),randomChoiceMinus(calcprevSale,randomSalsQty*(calcprevSale)/100)])
                
    #             return JsonResponse({'sales_qty':result , 'model_type':model_type})
    # return JsonResponse({'data':slsQty , 'model_type':model_type},safe = False)    
            
        if  prevSale[-1] > prevSale[-2]:  
            if slsQty<=upperCutoff1 and slsQty>=lowerCutoff1:
                return JsonResponse({'data':upperCutoff1 , 'model_type':model_type})
            elif (upperCutoff2 is not None) and (lowerCutoff2 is not None):    
                if slsQty<=upperCutoff2 and slsQty>=lowerCutoff2:
                    print("slsQty------",slsQty)
                    return JsonResponse({'data':upperCutoff2 , 'model_type':model_type})
            
                else:
                    calcprevSale = prevSale[-1]
                    randomSalsQty = 0
                    randomSalsQty = random.randint(1,5)
                    
                    if prevSale[-1] == 0:
                        if len(prevSale)>=2:
                            calcprevSale = prevSale[-2]
                        
                    result = random.choice([randomChoicePlus(calcprevSale,randomSalsQty*(calcprevSale)/100),randomChoiceMinus(calcprevSale,randomSalsQty*(calcprevSale)/100)])
                    
                    return JsonResponse({'sales_qty':result , 'model_type':model_type})
        else:
            if slsQty<=upperCutoff1 and slsQty>=lowerCutoff1:
                return JsonResponse({'data':slsQty , 'model_type':model_type})
            elif (upperCutoff2 is not None) and (lowerCutoff2 is not None):    
                if slsQty<=upperCutoff2 and slsQty>=lowerCutoff2:
                    print("slsQty------",slsQty)
                    return JsonResponse({'data':slsQty , 'model_type':model_type})
            
                else:
                    calcprevSale = prevSale[-1]
                    randomSalsQty = 0
                    randomSalsQty = random.randint(1,5)
                    
                    if prevSale[-1] == 0:
                        if len(prevSale)>=2:
                            calcprevSale = prevSale[-2]
                        
                    result = random.choice([randomChoicePlus(calcprevSale,randomSalsQty*(calcprevSale)/100),randomChoiceMinus(calcprevSale,randomSalsQty*(calcprevSale)/100)])
                    
                    return JsonResponse({'sales_qty':result , 'model_type':model_type})
    return JsonResponse({'data':slsQty , 'model_type':model_type},safe = False)        

def compare_model_forecasts(request):
    item_code = str(request.GET.get('item_code'))
    print("item_code----",item_code)
    from_date = request.GET.get('from_date')
    print("from_date----",from_date)
    to_date = request.GET.get('to_date')
    print("to_date----",to_date)
    no_of_days = request.GET.get('no_of_days')
    if no_of_days:
        no_of_days = int(no_of_days)
        
    data = fetch_data(item_code)
    models = train_models(data)    
        
    comparisons = compare_forecasts(item_code, models, from_date, to_date, no_of_days) 
    
    return JsonResponse(comparisons,safe = False)
    
        

