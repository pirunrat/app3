import joblib 
import os 
from django.conf import settings
import numpy as np
from .MultinomialRegression import MultinomialRegression, Ridge
from .utils import Utility
import mlflow
import pickle



class Model:

    def __init__(self, data, model_path):
        self.data = data
        self.model_path = model_path
        self.model = self.load_model()
       
        
        

    def model_predict(self):
        my_dict = {
        0: "The range of the predicted price is less than or equal to 252500",
        1: "The range of the predicted price is less than or equal to 450000",
        2: "The range of the predicted price is less than or equal to 675000",
        3: "The range of the predicted price is more than 675000",
        '': 'None'
    }
        mileage = self.z_score_scale(self.data['mileage'])

        maxPower = self.min_max_scale_max_power(self.data['max_power'])
        
        year = self.min_max_scale_year(self.data['year'])

        input = np.array([year,maxPower,mileage]).reshape(-1,3)
        result = self.model.predict(input)[0]
       
        return f"{my_dict[result]}"
    
    def model_predict_multi(self):
        my_dict = {
        0: "The range of the predicted price is less than or equal to 252500",
        1: "The range of the predicted price is less than or equal to 450000",
        2: "The range of the predicted price is less than or equal to 675000",
        3: "The range of the predicted price is more than 675000",
        '': 'None'
    }
        mileage = self.z_score_scale(self.data['mileage'])

        maxPower = self.min_max_scale_max_power(self.data['max_power'])
        
        year = self.min_max_scale_year(self.data['year'])

        input = np.array([year,maxPower,mileage]).reshape(-1,3)
        result = self.multinomial.predict(input)[0]
    
        return f"{my_dict[result]}"
    
    def model_predict_ridge(self):
        my_dict = {
        0: "The range of the predicted price is less than or equal to 252500",
        1: "The range of the predicted price is less than or equal to 450000",
        2: "The range of the predicted price is less than or equal to 675000",
        3: "The range of the predicted price is more than 675000",
        '': 'None'
        }
        mileage = self.z_score_scale(self.data['mileage'])

        maxPower = self.min_max_scale_max_power(self.data['max_power'])
        
        year = self.min_max_scale_year(self.data['year'])

        input = np.array([year,maxPower,mileage]).reshape(-1,3)
        result = self.ridge.predict(input)[0]
        
        return f"{my_dict[result]}"


    def save(self,filename:str, obj:object):
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # def load(self):
    #     file_path = os.path.join(settings.BASE_DIR, "app3/" + self.model_path)
    #     with open(file_path, 'rb') as handle:
    #         b = pickle.load(handle)
    #     return b
    
    def load_model(self):
        loaded_model = joblib.load(self.model_path )
        return loaded_model

    def min_max_scale_max_power(self, input_data):
        minVal = 0
        maxVal = 282
        return (input_data - minVal) / (maxVal - minVal)
    



    def min_max_scale_year(self, input_data):
        minVal = 1983
        maxVal = 2020
        return (input_data - minVal) / (maxVal - minVal)



    def z_score_scale(self, input_data):
        mean = 19.405247616118118
        std = 3.9714218411022917
        return (input_data - mean) / std


    
    
