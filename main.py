from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

class Item(BaseModel):
    GRE_Score : int
    TOEFL_Score : int
    CGPA : float
    Research : int
    University_Rating  : int


app = FastAPI()



model= pickle.load(open("/home/abdoun0hocine/devs/fastapi/ml_model/model/model_np.pkl", "rb"))




@app.get("/")
def homepage():
    return {"welcome": "home"}



@app.post("/predict/")
def make_prediction(item: Item):

    df = pd.DataFrame([item.dict().values()], item.dict().keys())
    
    df = df.to_numpy()
    pred = model.predict(df)
    return {"chance of admit": pred[0]}