from django.urls import path
from .views import ModelPredictionView

urlpatterns = [
    path('predict/', ModelPredictionView.as_view(), name='predict-sales'),
]
