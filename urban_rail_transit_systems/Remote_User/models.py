from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class rail_delay_prediction_model(models.Model):
    names=models.CharField(max_length=300)
    rail_name=models.CharField(max_length=300)
    rail_type=models.CharField(max_length=300)
    departure_place=models.CharField(max_length=300)
    destination=models.CharField(max_length=300)
    departure_date=models.CharField(max_length=300)
    departure_time=models.CharField(max_length=300)
    arrival_date=models.CharField(max_length=300)
    arrival_time=models.CharField(max_length=300)
    distruption_place_name=models.CharField(max_length=300)
    distruption_reason=models.CharField(max_length=300)
    distruption_time=models.CharField(max_length=300)
    actual_arrival_time=models.CharField(max_length=300)
    impact=models.CharField(max_length=300)


class rail_delay_model(models.Model):
    names=models.CharField(max_length=300)
    rail_name=models.CharField(max_length=300)
    rail_type=models.CharField(max_length=300)
    departure_place=models.CharField(max_length=300)
    destination=models.CharField(max_length=300)
    departure_date=models.CharField(max_length=300)
    departure_time= models.CharField(max_length=300)
    arrival_date=models.CharField(max_length=300)
    arrival_time= models.CharField(max_length=300)
    distruption_place_name=models.CharField(max_length=300)
    distruption_reason=models.CharField(max_length=300)
    distruption_time=models.CharField(max_length=300)
    actual_arrival_time=models.CharField(max_length=300)

class impact_ratio_model(models.Model):
    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


