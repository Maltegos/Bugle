from django.urls import path
from . import views

# URLConf
urlpatterns = [
    path('', views.upload, name='upload'),
    path('create', views.create, name='create'),

]
