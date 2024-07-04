# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EDA:
    def __init__(self):
        pass
    
    def run(self):
        # Crear el directorio de salida si no existe
        output_dir = '../datos/eda_outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Cargar los datos
        data = pd.read_csv('../datos/bike-sharing-demand/train_ext.csv')

        # Mostrar las primeras filas del conjunto de datos y guardarlas en un archivo
        head_data = data.head()
        head_data.to_csv(os.path.join(output_dir, 'head_data.csv'), index=False)

        # Descripción general de los datos y guardarlas en un archivo
        info_buffer = pd.DataFrame({'info': data.dtypes})
        info_buffer.to_csv(os.path.join(output_dir, 'data_info.csv'))

        describe_data = data.describe()
        describe_data.to_csv(os.path.join(output_dir, 'data_description.csv'))

        # Convertir la columna 'datetime_x' a formato datetime
        data['datetime_x'] = pd.to_datetime(data['datetime_x'])

        # Crear nuevas características a partir de 'datetime_x'
        data['hour'] = data['datetime_x'].dt.hour
        data['day'] = data['datetime_x'].dt.day
        data['month'] = data['datetime_x'].dt.month
        data['year'] = data['datetime_x'].dt.year
        data['day_of_week'] = data['datetime_x'].dt.dayofweek
        data['week_of_year'] = data['datetime_x'].dt.isocalendar().week
        data['quarter'] = data['datetime_x'].dt.quarter
        data['semester'] = np.where(data['month'] <= 6, 1, 2)
        data['am_pm'] = np.where(data['hour'] < 12, 'AM', 'PM')

        # Evaluación de la calidad de los datos
        # Verificar valores nulos y guardar en un archivo
        missing_data = data.isnull().sum()
        missing_data.to_csv(os.path.join(output_dir, 'missing_data.csv'))

        # Descripción estadística de las características numéricas y guardarlas en un archivo
        numeric_description = data.describe()
        numeric_description.to_csv(os.path.join(output_dir, 'numeric_description.csv'))

        # Análisis univariado
        # Distribución de la demanda de bicicletas
        plt.figure(figsize=(10, 6))
        sns.histplot(data['count'], bins=50, kde=True)
        plt.title('Distribución de la Demanda de Bicicletas')
        plt.xlabel('Número de Alquileres')
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(output_dir, 'count_distribution.png'))
        plt.close()

        # Análisis bivariado
        # Relación entre la demanda y las características temporales
        plt.figure(figsize=(20, 15))

        plt.subplot(3, 2, 1)
        sns.boxplot(x='hour', y='count', data=data)
        plt.title('Demanda por Hora del Día')

        plt.subplot(3, 2, 2)
        sns.boxplot(x='day_of_week', y='count', data=data)
        plt.title('Demanda por Día de la Semana')

        plt.subplot(3, 2, 3)
        sns.boxplot(x='month', y='count', data=data)
        plt.title('Demanda por Mes')

        plt.subplot(3, 2, 4)
        sns.boxplot(x='year', y='count', data=data)
        plt.title('Demanda por Año')

        plt.subplot(3, 2, 5)
        sns.boxplot(x='am_pm', y='count', data=data)
        plt.title('Demanda por AM/PM')

        plt.subplot(3, 2, 6)
        sns.boxplot(x='quarter', y='count', data=data)
        plt.title('Demanda por Trimestre')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_demand_analysis.png'))
        plt.close()

        # Relación entre la demanda y las características meteorológicas
        plt.figure(figsize=(20, 15))

        plt.subplot(3, 2, 1)
        sns.scatterplot(x='temp_x', y='count', data=data)
        plt.title('Demanda vs. Temperatura')

        plt.subplot(3, 2, 2)
        sns.scatterplot(x='feelslike', y='count', data=data)
        plt.title('Demanda vs. Sensación Térmica')

        plt.subplot(3, 2, 3)
        sns.scatterplot(x='humidity_x', y='count', data=data)
        plt.title('Demanda vs. Humedad')

        plt.subplot(3, 2, 4)
        sns.scatterplot(x='windspeed_x', y='count', data=data)
        plt.title('Demanda vs. Velocidad del Viento')

        plt.subplot(3, 2, 5)
        sns.scatterplot(x='sealevelpressure', y='count', data=data)
        plt.title('Demanda vs. Presión Barométrica')

        plt.subplot(3, 2, 6)
        sns.scatterplot(x='visibility', y='count', data=data)
        plt.title('Demanda vs. Visibilidad')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weather_demand_analysis.png'))
        plt.close()

        # Análisis de correlación
        correlation_matrix = data[['temp_x', 'feelslike', 'humidity_x', 'windspeed_x', 'sealevelpressure', 'visibility', 'count']].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

        # Análisis de la demanda por estaciones del año
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='season', y='count', data=data)
        plt.title('Demanda por Estaciones del Año')
        plt.xlabel('Estación')
        plt.ylabel('Número de Alquileres')
        plt.savefig(os.path.join(output_dir, 'seasonal_demand_analysis.png'))
        plt.close()

        # Análisis de la demanda en días festivos vs días laborales
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='holiday', y='count', data=data)
        plt.title('Demanda en Días Festivos vs Días No Festivos')
        plt.xlabel('Festivo')
        plt.ylabel('Número de Alquileres')
        plt.savefig(os.path.join(output_dir, 'holiday_demand_analysis.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='workingday', y='count', data=data)
        plt.title('Demanda en Días Laborales vs Días No Laborales')
        plt.xlabel('Día Laboral')
        plt.ylabel('Número de Alquileres')
        plt.savefig(os.path.join(output_dir, 'workingday_demand_analysis.png'))
        plt.close()

        # Guardar el conjunto de datos preprocesado
        data.to_csv(os.path.join(output_dir, 'bike_sharing_demand_preprocessed.csv'), index=False)

        print("EDA completado y gráficos guardados correctamente.")
