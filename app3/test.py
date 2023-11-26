import numpy as np
from app3.MultinomialRegression import MultinomialRegression, Ridge
from django.test import TestCase
from .utils import Utility
import numpy as np
import pandas as pd
import mlflow 
import os


class MultinomialRegressionTests(TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        self.model_name = "st124003-a3-model"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        self.stage = "Staging"
        self.util = Utility()
        super().__init__(methodName)

    
    def setUp(self):
        #self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{self.model_name}/{self.stage}")
        self.model = self.util.load('app3/model.pkl')
        self.X = np.array([0.1,0.2,0.3]).reshape(-1,3)
        
    
    def test_load_model(self):
        print(type(self.model))
        self.assertIsNotNone(self.model)

    def test_model_input(self):
        pred = self.model.predict(self.X)
        self.assertIsNotNone(pred)

    def test_model_output(self):
        pred = self.model.predict(self.X)
        self.assertEqual(pred.shape, (1,))

  