from django.shortcuts import render, redirect
from .form import CarPredictionForm
from .Machine_Learning_Model import Model
import numpy as np

def main_page(request):

    # Validate the form data
    if request.method == "POST":  # Check if the request method is POST
        choice = request.POST.get('choice')  # Getting the choice value
        
        if choice == 'multinomial':
            return multinomial(request)
        elif choice == 'gaussian_naive_bayes':
            return gaussian_naive_bayes(request)
        elif choice == 'ridge':
            return ridge(request)

    return render(request, 'main_new.html')


def multinomial(request):
    prediction = ''
    if request.method == 'POST':
        form = CarPredictionForm(request.POST)
        if form.is_valid():
            mileage = form.cleaned_data['mileage']
            max_power = form.cleaned_data['max_power']
            year = form.cleaned_data['year']
            input_data = {'mileage': mileage, 'max_power': max_power, 'year': year}
            model = Model(input_data, 'app3/model.pkl')
            prediction = model.model_predict()
    else:
        form = CarPredictionForm()
    return render(request, 'multinomial.html', {'form': form, 'prediction': prediction})




def gaussian_naive_bayes(request):
    prediction = ''
    if request.method == 'POST':
        form = CarPredictionForm(request.POST)
        if form.is_valid():
            mileage = form.cleaned_data['mileage']
            max_power = form.cleaned_data['max_power']
            year = form.cleaned_data['year']
            input_data = {'mileage': mileage, 'max_power': max_power, 'year': year}
            model = Model(input_data, 'app3/GNB.pkl')
            prediction = model.model_predict()
    else:
        form = CarPredictionForm()
    return render(request, 'gaussian_naive_bayes.html', {'form': form, 'prediction': prediction})



def ridge(request):
    prediction = ''
    if request.method == 'POST':
        form = CarPredictionForm(request.POST)
        if form.is_valid():
            mileage = form.cleaned_data['mileage']
            max_power = form.cleaned_data['max_power']
            year = form.cleaned_data['year']
            input_data = {'mileage': mileage, 'max_power': max_power, 'year': year}
            model = Model(input_data, 'app3/ridge.pkl')
            prediction = model.model_predict()
    else:
        form = CarPredictionForm()
    return render(request, 'ridge.html', {'form': form, 'prediction': prediction})



def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')

def howtouse(request):
    return render(request,'howtouse.html')
    



