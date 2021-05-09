import os

import numpy as np
import pandas as pd


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pickle

from sklearn import preprocessing

def save_model(filename, model):
	# Save model
    # filename = 'Inference_lda.pkl'
    pickle.dump(model, open(filename, 'wb'))

def train():
	# Load, read and normalize training data

	df = pd.read_csv("data/train.csv")

	y_train = df['# Letter']

	X_train = df.drop(df.loc[:, 'Line':'# Letter'].columns, axis = 1)

	print("Shape of the training data")
	print(X_train.shape)
	print(y_train.shape)

	# Data normalization (0,1)
	print("\nNormalizing the train data using L2 Norm")
	X_train = preprocessing.normalize(X_train, norm='l2')

	# Models training

	# Linear Discrimant Analysis (Default parameters)
	clf_lda = LinearDiscriminantAnalysis()
	print("\n\nTraining the LDA model\n")
	clf_lda.fit(X_train, y_train)

	# Save model
	filename = 'models/Inference_lda.pkl'
	save_model(filename, clf_lda)


	# Neural Networks multi-layer perceptron (MLP) algorithm
	clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(300,), random_state=0, max_iter=700)
	print("Training Neural Network classifier\n")
	clf_NN.fit(X_train, y_train)

	# Save model
	filename = 'models/Inference_NN.pkl'
	save_model(filename, clf_NN)

	# # GradientBoostingClassifier
	# grb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
	# print("Training GradientBoostingClassifier Model\n")
	# grb_model.fit(X_train, y_train)

	# # save model
	# filename = "models/Inference_GRB.pkl"
	# save_model(filename, grb_model)


if __name__ == '__main__':
    train()