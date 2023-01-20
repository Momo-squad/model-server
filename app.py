from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile
from core.crop_prediction import get_crop_prediction
from core.disease_prediction import get_disease_prediction
from core.helper import load_image_into_numpy_array

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/maris")
async def maris():
    return "chaina"

@app.get("/pred/{n}/{p}/{k}/{temp}/{hum}/{ph}/{rain}")
async def pred(n:float,p:float,k:float, temp:float, hum:float, ph:float, rain:float):
    return get_crop_prediction([n,p,k, temp, hum, ph, rain]);


@app.post("/disease/")
async def disease(file: UploadFile):
    image = load_image_into_numpy_array(await file.read())
    return get_disease_prediction(image)

