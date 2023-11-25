from pydantic import BaseSettings
import os

ENV_API = os.getenv("ENVIRONMENT")


class Settings(BaseSettings):
    # user
    secret_key: str
    algorithm: str
    access_token_expires_hours: int

    # clickhouse
    db_ch_host: str
    db_ch_port: int
    db_ch_user: str
    db_ch_protocol: str

    # mlflow
    mlflow_tracking_uri: str
    mlflow_s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str

    # rl settings
    db_oven_time_optimization: str
    table_series_upload_ovens: str

    class Config:
        env_file = ".env" if not ENV_API else f".env.{ENV_API}"


def get_settings():
    return Settings()