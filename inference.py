import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def read_model(model):
	return pickle.load(open(model, 'rb'))

def inference():
	# Load, normalize testing data

	test_df = pd.read_csv("data/test.csv")

	y_test = test_df['# Letter'].values
	X_test = test_df.drop(test_df.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
	print("Shape of the test data")
	print(X_test.shape)
	print(y_test.shape)

	# Data normalization (0,1)
	X_test = preprocessing.normalize(X_test, norm='l2')

	# Models training

	# Run model
	clf_lda = read_model('models/Inference_lda.pkl')
	print("LDA score and classification:")
	print(accuracy_score(clf_lda.predict(X_test), y_test))
	print(clf_lda.predict(X_test)[:20])
	    
	# Run model
	clf_nn = read_model('models/Inference_NN.pkl')
	print("NN score and classification:")
	print(clf_nn.score(X_test, y_test))
	print(clf_nn.predict(X_test)[:20])

	# Run model
	clf_grb = read_model('models/Inference_GRB.pkl')
	print("GRB score and classification:")
	print(clf_grb.score(X_test, y_test))
	print(clf_grb.predict(X_test)[:20])

if __name__ == '__main__':
    inference()