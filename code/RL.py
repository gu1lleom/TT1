import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class RL:
    def __init__(self):
        pass
    
    def run(self):
        # Crear el directorio de salida si no existe
        output_dir = '../datos/metrics_model'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Cargar datos de entrenamiento y prueba
        train_data = pd.read_csv('../datos/bike-sharing-demand/train_ext.csv')
        test_data = pd.read_csv('../datos/bike-sharing-demand/test_ext.csv')

        # Convertir la columna 'datetime_x' a tipo datetime si no está en ese formato
        train_data['datetime_x'] = pd.to_datetime(train_data['datetime_x'])
        test_data['datetime_x'] = pd.to_datetime(test_data['datetime_x'])

        # Agregar nuevas columnas para cada componente de fecha y hora en el conjunto de entrenamiento
        train_data['year'] = train_data['datetime_x'].dt.year
        train_data['month'] = train_data['datetime_x'].dt.month
        train_data['day'] = train_data['datetime_x'].dt.day
        train_data['hour'] = train_data['datetime_x'].dt.hour

        # Agregar nuevas columnas para cada componente de fecha y hora en el conjunto de prueba
        test_data['year'] = test_data['datetime_x'].dt.year
        test_data['month'] = test_data['datetime_x'].dt.month
        test_data['day'] = test_data['datetime_x'].dt.day
        test_data['hour'] = test_data['datetime_x'].dt.hour

        # Eliminar las columnas no numéricas y la columna objetivo en el conjunto de entrenamiento
        # Eliminar también las columnas 'casual' y 'registered'
        X_train = train_data.drop(columns=['datetime_x', 'count', 'datetime_y', 'sunrise', 'sunset', 'casual', 'registered','severerisk'])
        y_train = train_data['count']

        # Eliminar las columnas no numéricas en el conjunto de prueba
        X_test = test_data.drop(columns=['datetime_x', 'datetime_y', 'sunrise', 'sunset','severerisk'])

        # Crear un pipeline con imputación, normalización y Ridge Regression
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Añadido SimpleImputer aquí
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        # Definir la búsqueda de hiperparámetros
        param_grid = {
            'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
        }

        # Búsqueda de hiperparámetros
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Mejor modelo encontrado
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Mejores hiperparámetros encontrados:", best_params)

        # Guardar el modelo
        joblib.dump(best_pipeline, '../modelos/ridge_model.pkl')

        # Dividir los datos de entrenamiento en entrenamiento y validación (80% - 20%)
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Reentrenar el mejor modelo con el conjunto de entrenamiento dividido
        best_pipeline.fit(X_train_split, y_train_split)

        # Predicciones en el conjunto de validación
        y_val_pred = best_pipeline.predict(X_val)

        # Crear DataFrame para las predicciones de validación
        validation_data = pd.DataFrame(X_val, columns=X_train.columns)
        validation_data['count'] = y_val
        validation_data['predicted_count'] = y_val_pred

        # Guardar las predicciones de validación en el archivo de salida
        validation_data.to_csv('../datos/bike-sharing-demand/validation_with_predictions_rl.csv', index=False)

        # Predicciones en el conjunto de prueba
        y_pred = best_pipeline.predict(X_test)

        # Guardar las predicciones en el archivo de salida
        test_data['predicted_count'] = y_pred

        test_data[['datetime_x', 'predicted_count']].to_csv('../datos/bike-sharing-demand/test_with_predictions_rl.csv', index=False)

        # Calcular métricas
        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        # Guardar métricas en un archivo CSV
        metrics = pd.DataFrame({
            'RMSE': [rmse],
            'MAE': [mae],
            'R2_Score': [r2]
        })
        metrics.to_csv(os.path.join(output_dir, 'RL_metrics.csv'), index=False)

        print(f"RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}")


