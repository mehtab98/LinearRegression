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
def single_feature_solver(data, order, feature):
	#convert 'mpg' column into a numpy vector that is of size (392, 1)
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

#Function generates the MSE for both the Test and Tested Data for all 7 features 
def data_mse(data): 


	#Generating a panda's dataframe by using  
	stats = (4,14) 
	matrix = np.ones(shape = stats, dtype = float)
	columnz = pd.MultiIndex.from_product([["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"], ["train", "test"]])
	cf = pd.DataFrame(data=matrix, columns = columnz, index = ['0th','1st','2nd','3rd'] ) 
	col_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]

	#Split the training and testing Data 
	df1 = data.iloc[:292]
	dataz_train = df1 
	df2 = data.iloc[292:392]
	dataz_test = df2 
	

	feature_order = feature_matrix(df1)

	matrix = np.ones(shape = (100,35), dtype = float)

	#Generating a dataframe for ploting data for part b 
	columnz = pd.MultiIndex.from_product([["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"], ["in", "0", "1","2","3"]])
	plot_data = pd.DataFrame(data=matrix, columns = columnz) 
	plot_data['mpg']= df2['mpg'].reset_index(drop=True)


	#Assigns MSE to training data 
	total_index = 0
	i = 0
	while i <= 6: 
		f = 0
		while (f <= 3):
			mse = mean_squared_error(feature_order['original_mpg'],feature_order[total_index])
			order = total_index % 4

			if(order == 0):
				cf[col_names[i],'train']['0th'] = mse 
			elif(order == 1):
				cf[col_names[i],'train']['1st'] = mse 
			elif(order == 2): 
				cf[col_names[i],'train']['2nd']= mse 
			else: 
			 	cf[col_names[i],'train']['3rd']= mse 

			total_index = total_index + 1 
			f = f + 1 
		i = i + 1



	#Assigns MSE to testing data 
	total_index = 0
	i = 0
	while i <= 6: 
		f = 0
		while (f <= 3):
			order = total_index % 4
			feature = col_names[i] 
			plot_data[feature,"in"] = df2[feature].reset_index(drop=True)

			feature_weight = single_feature_weight(df1, order, feature) 

			x = single_feature_order(df2, order, feature)
			if(order == 0):

				feature_found =  x.dot(feature_weight)
				plot_data[feature,'0']= feature_found
				mse = mean_squared_error(df2['mpg'],feature_found) 
				cf[col_names[i],'test']['0th'] = mse 
			elif(order == 1):

				feature_found =  x.dot(feature_weight)
				plot_data[feature,'1'] = feature_found
				mse = mean_squared_error(df2['mpg'],feature_found) 

				cf[col_names[i],'test']['1st'] = mse 
			elif(order == 2): 

				feature_found =  x.dot(feature_weight)
				plot_data[feature,'2'] = feature_found
				mse = mean_squared_error(df2['mpg'],feature_found) 
				cf[col_names[i],'test']['2nd']= mse 
			else: 

				feature_found =  x.dot(feature_weight)

				plot_data[feature,'3'] = feature_found
			
				mse = mean_squared_error(df2['mpg'],feature_found) 
				cf[col_names[i],'test']['3rd']= mse 

			total_index = total_index + 1 
			f = f + 1
		i = i + 1 

	#prints the final MSE returns a plot_data to be used by plt_data 
	print('\n Our Mean Score\'s calculated to the third order for both training and test:')
	print(cf)
	return plot_data

#This function plots all 7 plots for testing error 
def plt_data(data): 
	data = data_mse(data)
	col_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"] 

	for feature in col_names: 

		plt.gca().set_ylim([0,50])
		plt.ylabel('mpg')
		plt.xlabel(feature)
		plt.scatter(data[feature,'in'], data["mpg"])

		file_name = 'problem4'+ feature + '.png'

		order = 0 
		feature_data = data[feature]
		
		while order <= 3:
			feat = feature_data.sort_values(by='in', ascending=True)
			label = str(order) + ' order'

			plt.plot(feat['in'],feat[str(order)],label = label)
			order = order + 1 
		
		plt.gca().legend()
		plt.savefig(file_name)
		print('plotting', file_name)
		plt.clf()

	
global dataz 
global dataz_train 
global dataz_test 


dataz = shuffle_data()
dataz_train = dataz.iloc[:292]
dataz_test  = dataz.iloc[292:392]

plt_data(dataz)


def share_var():
	global datar
	global data_train 
	global data_test
	datar = dataz 
	data_test = dataz_test
	data_train = dataz_train


