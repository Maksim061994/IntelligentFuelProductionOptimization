from pydantic import BaseModel
from typing import Dict


class RequestDataOvenModel(BaseModel):
    date: str
    one_brigad: bool = True
    data: Dict


class ResponseDataResulOvenModel(BaseModel):
    result: str
    metric: float
