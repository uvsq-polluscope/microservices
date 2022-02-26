from flask import Flask
from flask_restful import Resource, Api, reqparse, abort, marshal, fields

import pandas as pd
# from dependancies.file import *


# Initialize Flask
app = Flask(__name__)
api = Api(app)

@app.route("/")
def runing():
    return "Hi classification microserivce is runing"


@app.route("/classification")
def classification():
    df = get_data()
    
    save_data(df)

    return "Done"



def save_data(df): 
    return True

def get_data(): 
    
    return True