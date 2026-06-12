from django.urls import path
from . import views

urlpatterns = [
    path("", views.classify_complaint, name="home"),
]
