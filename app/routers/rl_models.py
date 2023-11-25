from fastapi import APIRouter
from app.handlers.rl_models.schemas import RequestDataOvenModel, ResponseDataResulOvenModel
from app.handlers.rl_models.oven_optimization import OvenTimeOptimizationModel
from fastapi.responses import StreamingResponse

rl_models_router = APIRouter()


@rl_models_router.post('/oven_optimization/upload')
def upload_data_oven(request: RequestDataOvenModel):
    model = OvenTimeOptimizationModel()
    result = model.start_calc(request.dict())
    return ResponseDataResulOvenModel(**result)


@rl_models_router.get('/oven_optimization')
async def get_report_by_oven(date: str):
    model = OvenTimeOptimizationModel()
    excel_file = await model.make_report(date)
    return StreamingResponse(
        content=excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={'Content-Disposition': 'attachment; filename="report.xlsx"'}
    )

