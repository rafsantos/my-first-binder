import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import pandas
from sklearn.neighbors import KNeighborsClassifier


df = pandas.read_csv('IRIS.csv')
model = KNeighborsClassifier(n_neighbors=3)

features = list(zip(df["sepal_length"], df["sepal_width"]))

model.fit(features,df["species"])
pickle.dump(model, open('model.pkl','wb'))
