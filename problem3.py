import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns

from sklearn.metrics import mean_squared_error 

#This function simply re-shuffles the dataset 
#this function is applied in problem number: 4 and 5 
def shuffle_data(): 
	data = quartile_categorize() 
	df = data.sample(frac=1).reset_index(drop=True)
	return df 


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


#This function is able to find a single-feature's matrix to the Nth order 
def single_feature_order(data, order, feature): 
	
	total_rows = data.shape[0]  # gives number of row count

	total_columns = order + 1


	stats = (total_rows, total_columns)
	feature = data[feature].reset_index(drop=True)
	
	current_col = 0 
	degree = 0 

	while current_col <= order:

		if(current_col == 0): 
			#print('linsl')
			matrix = np.ones(shape = stats, dtype = float)

		elif(current_col == 1): 
			matrix[:, 1] = feature 
	
		else: 

			new_feature = np.power(feature.astype(float), current_col)
			matrix[:, current_col] = new_feature

		current_col = current_col + 1 
		degree = degree + 1
	return matrix


#this function is incharge of implementing OLS on a single variable for prediction of MPG.
#Problem also utilizes function single_feature_order which can be found right above
def single_feature_solver(order, feature):
	#convert 'mpg' column into a numpy vector that is of size (392, 1)
	data = quartile_categorize() 
	y = data['mpg'].values 
	y = np.array(y)
	#y = np.matrix(y)
	#y = np.transpose(y)

	#get a column that has feature to nth order 
	x = single_feature_order(data, order, feature)

	#weight = (X_transpose * X)^-1 * X^T * Y
	weight = np.dot(np.dot(np.linalg.pinv(np.dot(x.T,x)),x.T),y)

	#mpg = x * weight 
	mpg = x.dot(weight)

	return mpg 


def single_feature_weight(data, order, feature):
	#convert 'mpg' column into a numpy vector that is of size (392, 1)
	y = data['mpg'].values 
	y = np.array(y)
	#y = np.matrix(y)
	#y = np.transpose(y)

	#get a column that has feature to nth order 
	x = single_feature_order(data, order, feature)

	#weight = (X_transpose * X)^-1 * X^T * Y
	weight = np.dot(np.dot(np.linalg.pinv(np.dot(x.T,x)),x.T),y)

	return weight


#This matrix produces all of the variables to whatever order they belong to. 

def feature_matrix(data, num_orders = 3):

	original_data = data 

	original_mpg = data['mpg']
	mile_category = data['mile_category']

	d = {'original_mpg': original_mpg, 'mile_category': mile_category}
	df = pd.DataFrame(data=d)

	total_rows = data.shape[0] 
	total_columns = 7 * num_orders 

	stats = (total_rows , total_columns)

	matrix = np.ones(shape = stats, dtype = float) 

	cf = pd.DataFrame(data=matrix) 
	feature_matrix = df.join(cf)


	data = data.drop(['mile_category','car_name','mpg'], axis=1)

	features = data.columns

	f_length = 0 

	
	index_start = 0 
	while f_length <= 6: 
		
		o_length = 0 
		while (o_length <= num_orders):
			current_feature = features[f_length]
			estimated_mpg = single_feature_solver(original_data, o_length, current_feature)
			feature_matrix[index_start] = estimated_mpg
			index_start = index_start + 1 
			o_length = o_length + 1 
		

		f_length = f_length + 1 
	

	return feature_matrix 

order = 2 
feature = 'displacement'
print(single_feature_solver(order,feature))