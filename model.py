import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        self.df = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        
    def load_data(self):
        self.df = pd.read_csv('weather_dataset.csv')
        self.df = pd.get_dummies(self.df, columns=['weather'], drop_first=True)
        self.X = self.df[['temperature', 'humidity', 'wind_speed', 'pressure'] + list(self.df.columns[self.df.columns.str.startswith('weather_')])]
        self.Y = self.df['go out (1=yes, 0=No)']
        
    def train(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.Y_train)
        
    def predict(self, data):
        return self.model.predict(data)
    
    def score(self):
        Y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.Y_test, Y_pred)
    
    def plot(self):
        y_pred_train = self.model.predict(self.X_train)
        plt.scatter(self.X['temperature'], self.Y, color='blue', label='Real data')
        plt.scatter(self.X_train['temperature'], y_pred_train, color='red', label='Predicted data')
        plt.xlabel('Temperature')
        plt.ylabel('Go out')
        plt.title('Logistic Regression Model')
        plt.legend()
        plt.show()