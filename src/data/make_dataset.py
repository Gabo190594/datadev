# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections as co
import os

import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Para un modelo supervisado
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from collections import Counter

# Crea una semilla cualesquiera
seed = 19
np.random.seed(seed)

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

def data_preparation(df):
    exclude = ['customerID','Churn','Churn_F']
    df.loc[df['TotalCharges'] == ' ',:]
    df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = np.nan
    df['TotalCharges'] = [float(x) for x in df['TotalCharges']]
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    df['Churn_F'] = [ 1 if x=="Yes" else 0 for x in df['Churn']]
    
    cols = [x for x in df.columns if x not in exclude]
    cols_cat = df[cols].select_dtypes(['object']).columns.tolist()
    index_categorical=[cols.index(x) for x in cols_cat]

    for i in cols_cat:
        le = preprocessing.LabelEncoder()
        le.fit(list(df[i].dropna()))
        df.loc[~df[i].isnull(),i]=le.transform(df[i].dropna())

    return df

def data_exporting(df, filename):
    dfp = df
    dfp.to_csv(os.path.join('../../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

def main():
    target = 'Churn_F'
    exclude = ['customerID','Churn','Churn_F']
    df1 = read_file_csv('TelcoCustomerChurn.csv')
    tdf1 = data_preparation(df1)    
    y=tdf1[target]
    X=tdf1.drop(exclude,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=1,stratify=y)
    
    # Matriz de Entrenamiento

    data_exporting(X_train, 'X_train.csv')
    data_exporting(y_train, 'Y_train.csv')
    data_exporting(X_test, 'X_test.csv')
    data_exporting(y_test, 'Y_test.csv')
        
if __name__ == "__main__":
    main()
