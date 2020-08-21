import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt  
import seaborn as sns

#This function simply adds labels and cleans the data 
def clean_car_data():
	columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name","mile_category"]
	df = pd.read_csv("auto-mpg.data", delimiter = "\\s+", names = columns)
	clean_df = df[df.horsepower != '?']
	reset_df = clean_df.reset_index(drop=True)
	return reset_df 


#this function takes data and splits the set into either 'low','medium', 'high', or 'very high' based on mpg 
#we do this categoriziation by takinng the quartile distribution 
def quartile_categorize():
	data = clean_car_data()
	categories = pd.qcut(x=data["mpg"], q=4, labels=['low','medium', 'high', 'very high'])
	data["mile_category"] = categories
	return data


#problem #2: this function is able to create a 2D scatterplot matrix by utilizing seaborn 
#the  2D scatterplot will be created and saved as 1_seaborn_pair_plot.png 
def plot_features():
	sns.set(style="ticks")

	data = quartile_categorize()
	data.dropna(inplace=True)
	plot_data = data.drop(columns=['mpg', 'car_name'])
	plt.figure()

	sns.pairplot(plot_data, hue = 'mile_category')
	plt.savefig("problem2_plot.png") 

plot_features()