# myapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # path('forecast_sales_qty/', views_06062024.forecast_sales_qty, name='forecast_sales_qty'),
    # path('compare_model_forecasts/', views_06062024.compare_model_forecasts, name='compare_model_forecasts'),
    # path('plot_sales_qty/<str:item_code>', views_06062024.plot_sales_qty, name='plot_sales_qty'),

    # path('plott_sales_qty/', views_06062024.plott_sales_qty, name='plott_sales_qty')
    
    path('plot_sales_qty/<str:item_code>', views.plot_sales_qty, name='plot_sales_qty')
    

]
