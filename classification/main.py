

import pandas as pd
# from dependancies.file import *

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def runing():
    return "Hi classification microserivce is runing"


@app.get("/classification")
def classification():
    df = get_data()
    
    save_data(df)

    return "Done"



def save_data(df): 
    return True

def get_data(): 
    
    return True