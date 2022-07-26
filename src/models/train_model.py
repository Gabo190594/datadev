import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier


# Cargar la tabla transformada
def train(filename1,filename2):
    X_train = pd.read_csv(os.path.join('../../data/processed', filename1))
    Y_train = pd.read_csv(os.path.join('../../data/processed', filename2))
    print(filename1, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    
    clf_base = RandomForestClassifier(random_state=1)
    clf_base.fit(X_train, Y_train['Churn_F'])
    print('Modelo entrenado')
    
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../../models/best_model.pkl'
    pickle.dump(clf_base, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    train('X_train.csv','Y_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()