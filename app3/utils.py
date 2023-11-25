import pickle
import joblib
import mlflow
import os
from mlflow.client import MlflowClient


class Utility:

    def __init__(self):
        self.model_name = 'st124003-a3-model'
        mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th")
        if not self.model_name:
           raise ValueError("Environment variable 'APP_MODEL_NAME' is not set.")
        

    def save(self,filename:str, obj:object):
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,filename:str) -> object:
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b



    def load_mlflow(self, stage='Staging'):
        cache_path = os.path.join("app3/models", stage)
        os.makedirs(cache_path, exist_ok=True)
        
        path = os.path.join(cache_path, self.model_name)
        print(path)
        if not os.path.exists(path):
            try:
                model = mlflow.pyfunc.load_model(model_uri=f"models:/{self.model_name}/{stage}")
                self.save(filename=path, obj=model)
            except Exception as e:
                raise RuntimeError(f"Error loading model from MLFlow: {e}")
        return model


    def register_model_to_production(self):

        # Initializing a MLFlow Client object
        client = MlflowClient()
        for model in client.get_registered_model("st124003-a3-model").latest_versions: #type: ignore

            # find model in Staging
            if(model.current_stage == "Staging"):

                # Push the model to Production
                version = model.version
                client.transition_model_version_stage(
                    name=self.model_name, version=version, stage="Production", archive_existing_versions=True
                )