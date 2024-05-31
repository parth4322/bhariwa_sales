
# Create your models here.

# myapp/models.py
from django.db import models

class SalesData(models.Model):
    item_code = models.CharField(max_length=255)
    year = models.IntegerField()
    month = models.IntegerField()
    sales_qty = models.IntegerField()

