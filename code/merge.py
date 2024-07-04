import pandas as pd

class MergeData:
    def __init__(self):
        pass
    
    def run(self):
        # Leer los archivos CSV
        train = pd.read_csv('../datos/bike-sharing-demand/train.csv')
        test = pd.read_csv('../datos/bike-sharing-demand/test.csv')
        weather = pd.read_csv('../datos/bike-sharing-demand/washington dc 2011-01-01 to 2012-12-31.csv')

        # Convertir la columna 'datetime' de train y test a tipo datetime
        train['datetime'] = pd.to_datetime(train['datetime'])
        test['datetime'] = pd.to_datetime(test['datetime'])

        # Convertir la columna 'datetime' de weather a tipo datetime (eliminando la parte del tiempo)
        weather['datetime'] = pd.to_datetime(weather['datetime']).dt.date

        # Eliminar columnas no deseadas de weather
        weather = weather.drop(columns=['name', 'preciptype', 'conditions', 'description', 'icon', 'stations'])

        # Crear una columna 'date' en train y test que contenga solo la parte de la fecha
        train['date'] = train['datetime'].dt.date
        test['date'] = test['datetime'].dt.date

        # Hacer el join entre train y weather
        train_ext = pd.merge(train, weather, left_on='date', right_on='datetime', how='left')

        # Hacer el join entre test y weather
        test_ext = pd.merge(test, weather, left_on='date', right_on='datetime', how='left')

        # Eliminar la columna 'date' de train_ext y test_ext
        train_ext = train_ext.drop(columns=['date'])
        test_ext = test_ext.drop(columns=['date'])

        # Guardar los archivos resultantes
        train_ext.to_csv('../datos/bike-sharing-demand/train_ext.csv', index=False)
        test_ext.to_csv('../datos/bike-sharing-demand/test_ext.csv', index=False)

        print("Archivos guardados correctamente.")
