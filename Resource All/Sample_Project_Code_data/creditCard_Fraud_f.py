
# E.K 5/5/18 Predictive maintence for some devices e.g. engines

class CCFD:
	""" A Credit Card Fraud detection using Kaggle data """
	
	def __init__(self):
		""" Constructor """
		# Set up network size
		#if np.ndim(inputs)>1:
		#	self.nIn = np.shape(inputs)[1]
		#else: 
		#	self.nIn = 1
	
		#if np.ndim(targets)>1:
		#	self.nOut = np.shape(targets)[1]
		#else:
		#	self.nOut = 1

		#self.nData = np.shape(inputs)[0]
	
		# Initialise network
		#self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

	def LR():# Logistic Regression (a Binary Classifier)
		""" Logistic Regression """	
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import StratifiedShuffleSplit
		from sklearn.linear_model import LogisticRegression
		from sklearn.metrics import classification_report
		import pdb
		data = pd.read_csv('creditcard.csv')

		pdb.set_trace()
	        
	# Only use the 'Amount' and 'V1', ..., 'V28' features
		features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

	# The target variable which we would like to predict, is the 'Class' variable
		target = 'Class'

	# Now create an X variable (containing the features) and an y variable (containing only the target variable)
		X = data[features]
		y = data[target]

		# Define the model
		model = LogisticRegression()


		# Define the splitter for splitting the data in a train set and a test set
		splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

		# Loop through the splits (only one)
		for train_indices, test_indices in splitter.split(X, y):
   		 # Select the train and test data
			pdb.set_trace()
			X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
			X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
			# Normalize the data
			#X_train = p.normalize(X_train)
			#X_test = p.normalize(X_test)
			pdb.set_trace()
			# Fit and predict!
			model.fit(X_train, y_train) # Train the model
			y_pred = model.predict(X_test) # Test the model
			# And finally: show the results
			print("Logistic Regression Classification Results")
			print(classification_report(y_test, y_pred))

	def SVM():# Support Vecotr Machine (a Binary Classifier)
		""" Support Vector Machine """	
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import StratifiedShuffleSplit
		from sklearn import svm
		#from sklearn.linear_model import SVC
		from sklearn.metrics import classification_report

		data = pd.read_csv('creditcard.csv')

		import pdb; pdb.set_trace()
	        
	# Only use the 'Amount' and 'V1', ..., 'V28' features
		features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

	# The target variable which we would like to predict, is the 'Class' variable
		target = 'Class'

	# Now create an X variable (containing the features) and an y variable (containing only the target variable)
		X = data[features]
		y = data[target]

		# Define the model
		model = svm.SVC(gamma='scale')

		# Define the splitter for splitting the data in a train set and a test set
		splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

		# Loop through the splits (only one)
		for train_indices, test_indices in splitter.split(X, y):
   		 # Select the train and test data
			pdb.set_trace()
			X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
			X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
			
			X_train.columns = [''] * len(X_train.columns)
			X_test.columns = [''] * len(X_test.columns)
			#y_train.columns = [''] * len(y_train.columns)
			#y_test.columns = [''] * len(y_test.columns)

			# Normalize the data
			#X_train = p.normalize(X_train)
			#X_test = p.normalize(X_test)
			pdb.set_trace()
			# Fit and predict!
			model.fit(X_train, y_train)
			SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    			decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    			max_iter=-1, probability=False, random_state=None, shrinking=True,
    			tol=0.001, verbose=False)
			y_pred = model.predict(X_test)
			# And finally: show the results
			print("Support Vector Machine Classification Results")
			print(classification_report(y_test, y_pred))

	def MLP():# Multilayer Perceptron
		""" Multilayer Perceptron """	
		import numpy as np
		import pandas as pd
		from sklearn.neural_network import MLPClassifier
		from sklearn.model_selection import StratifiedShuffleSplit
		
		from sklearn.metrics import classification_report

		data = pd.read_csv('creditcard.csv')

		import pdb; pdb.set_trace()
	        
	# Only use the 'Amount' and 'V1', ..., 'V28' features
		features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

	# The target variable which we would like to predict, is the 'Class' variable
		target = 'Class'

	# Now create an X variable (containing the features) and an y variable (containing only the target variable)
		X = data[features]
		y = data[target]

		# Define the model
		model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

		# Define the splitter for splitting the data in a train set and a test set
		splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

		# Loop through the splits (only one)
		for train_indices, test_indices in splitter.split(X, y):
   		 # Select the train and test data
			pdb.set_trace()
			X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
			X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
			# Normalize the data
			#X_train = p.normalize(X_train)
			#X_test = p.normalize(X_test)
			pdb.set_trace()
			# Fit and predict!
			model.fit(X_train, y_train) # Train the model
			#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(5, 2),learning_rate='constant', learning_rate_init=0.001,max_iter=200, momentum=0.9, n_iter_no_change=10,nesterovs_momentum=True, power_t=0.5, random_state=1,shuffle=True, solver='lbfgs', tol=0.0001,validation_fraction=0.1, verbose=False, warm_start=False)
			y_pred = model.predict(X_test) # Test the model
			# And finally: show the results
			print("MLP Classification Results")
			print(classification_report(y_test, y_pred))

	def normalize(X):# For now, skipping normalization
		""" Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation. """
		for feature in X.columns:
			X[feature] -= X[feature].mean()
			X[feature] /= X[feature].std()
		return X
	# Model with Logistic Regression
	#


#p = CCFD
#p.LR()
#p.SVM()