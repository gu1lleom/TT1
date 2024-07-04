import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class RF:
    def __init__(self):
        pass
    
    def run(self):
        # Cargar datos
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
        train_data['minute'] = train_data['datetime_x'].dt.minute
        train_data['second'] = train_data['datetime_x'].dt.second
        train_data['day_of_week'] = train_data['datetime_x'].dt.dayofweek

        # Agregar nuevas columnas para cada componente de fecha y hora en el conjunto de prueba
        test_data['year'] = test_data['datetime_x'].dt.year
        test_data['month'] = test_data['datetime_x'].dt.month
        test_data['day'] = test_data['datetime_x'].dt.day
        test_data['hour'] = test_data['datetime_x'].dt.hour
        test_data['minute'] = test_data['datetime_x'].dt.minute
        test_data['second'] = test_data['datetime_x'].dt.second
        test_data['day_of_week'] = test_data['datetime_x'].dt.dayofweek

        # Seleccionar características y objetivo en el conjunto de entrenamiento
        X_train = train_data.drop(columns=['datetime_x', 'count', 'casual', 'registered','severerisk'])
        y_train = train_data['count']

        # Seleccionar características en el conjunto de prueba
        X_test = test_data.drop(columns=['datetime_x','severerisk'])

        # Convertir todas las características a tipo numérico (si es necesario)
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')

        # Verificar si hay valores faltantes en X_train o X_test y eliminarlos o imputarlos
        if X_train.isnull().sum().sum() > 0:
            X_train = X_train.fillna(0)
        if X_test.isnull().sum().sum() > 0:
            X_test = X_test.fillna(0)

        # Dividir los datos de entrenamiento en entrenamiento y validación (80% - 20%)
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Normalizar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Definir el modelo de Random Forest
        rf = RandomForestRegressor(random_state=42)

        # Definir la búsqueda de hiperparámetros
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30, 50, 100, None],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'bootstrap': [True, False]
        }

        param_grid = {
            'n_estimators': [200],
            'max_features': ['sqrt'],
            'max_depth': [50],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'bootstrap': [False]
        }

        # Búsqueda de hiperparámetros
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

        # Entrenar el modelo
        grid_search.fit(X_train_scaled, y_train_split)

        # Mejor modelo encontrado
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Mejores hiperparámetros encontrados:", best_params)

        # Guardar el modelo
        joblib.dump(best_pipeline, '../modelos/random_forest_model.pkl')

        # Predicciones en el set de validación
        y_val_pred = best_pipeline.predict(X_val_scaled)

        # Preparar los datos para guardar las predicciones de validación
        validation_data = pd.DataFrame(X_val, columns=X_train.columns)
        validation_data['count'] = y_val
        validation_data['predicted_count'] = y_val_pred

        # Guardar las predicciones de validación en el archivo de salida
        validation_data.to_csv('../datos/bike-sharing-demand/validation_with_predictions_rf.csv', index=False)

        # Predicciones en el set de prueba
        y_test_pred = best_pipeline.predict(X_test_scaled)

        # Preparar los datos para guardar las predicciones
        test_data['predicted_count'] = y_test_pred

        # Guardar las predicciones en el archivo de salida
        test_data[['datetime_x', 'predicted_count']].to_csv('../datos/bike-sharing-demand/test_with_predictions_rf.csv', index=False)

        # Calcular métricas en el conjunto de validación
        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        # Guardar métricas en un archivo CSV
        metrics = pd.DataFrame({
            'RMSE': [rmse],
            'MAE': [mae],
            'R2_Score': [r2]
        })
        metrics.to_csv('../datos/metrics_model/RF_metrics.csv', index=False)

        print(f"Predicciones guardadas en '../datos/bike-sharing-demand/test_with_predictions_rf.csv'")
        print(f"Predicciones de validación guardadas en '../datos/bike-sharing-demand/validation_with_predictions_rf.csv'")
        print(f"RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}")
