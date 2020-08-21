import pandas as pd 
import numpy as np


def clean_car_data():
	columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name","mile_category"]
	df = pd.read_csv("auto-mpg.data", delimiter = "\\s+", names = columns)
	clean_df = df[df.horsepower != '?']
	reset_df = clean_df.reset_index(drop=True)
	return reset_df 

#problem #1: this function takes data and splits the set into either 'low','medium', 'high', or 'very high' based on mpg 
#we do this categoriziation by takinng the quartile distribution 
def quartile_categorize():
	data = clean_car_data()
	categories = pd.qcut(x=data["mpg"], q=4, labels=['low','medium', 'high', 'very high'])
	data["mile_category"] = categories
	return data

print(quartile_categorize())