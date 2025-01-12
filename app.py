from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

app = FastAPI()

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


class ItemCollection(BaseModel):
    items: List[Item]


@app.get("/")
def main():
    return {"model is ready"}

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    
    prediction = model.predict(data)
    
    return prediction[0]

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> pd.DataFrame:
    contents = await file.read()
    df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
    predictions = model.predict(df)
    
    df['predictions'] = predictions
    
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)
    
    return {"result": f"Predictions saved to {output_file}"}
