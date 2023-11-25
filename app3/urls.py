"""app3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import multinomial, gaussian_naive_bayes, ridge,main_page,about,contact,howtouse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',main_page,name='main_page'),
    path('multinomial', multinomial, name='multinomial'),
    path('gaussian_naive_bayes', gaussian_naive_bayes, name='gaussian_naive_bayes'),
    path('ridge', ridge, name='ridge'),
    path('about',about,name='about'),
    path('contact',contact,name='contact'),
    path('howtouse',howtouse,name='howtouse')
]
