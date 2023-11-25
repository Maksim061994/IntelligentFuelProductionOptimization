import mlflow
import os
from app.config.settings import get_settings
import torch


settings = get_settings()


class BaseModel:

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.mlflow_s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = settings.aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = settings.aws_secret_access_key
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def __init__(self):
        self.client_mf = mlflow.tracking.MlflowClient()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def predict(self, *args, **kwargs):
        pass

    def list_models_in_production(self, *args, **kwargs):
        pass

    def load_production_pytorch_model_mlflow(self, name: str):
        model_name = f"models:/{name}/Production"
        model = mlflow.pytorch.load_model(
            model_name,
            map_location=self.device
        )
        return model

    def load_production_catboost_model_mlflow(self, name: str):
        model_name = f"models:/{name}/Production"
        model = mlflow.catboost.load_model(model_name)
        return model

    def load_production_pyfunc_model_mlflow(self, name: str):
        model_name = f"models:/{name}/Production"
        model = mlflow.pyfunc.load_model(model_name)
        return model

