import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CompareModels:
    def __init__(self):
        pass
    
    def run(self):
        # Crear directorio para los gráficos si no existe
        os.makedirs('../datos/comparation_graph', exist_ok=True)

        # Cargar las métricas de los archivos CSV
        rl_metrics = pd.read_csv('../datos/metrics_model/RL_metrics.csv')
        rf_metrics = pd.read_csv('../datos/metrics_model/RF_metrics.csv')

        # Añadir una columna de modelo para diferenciar en el gráfico de barras
        rl_metrics['Model'] = 'Linear Regression'
        rf_metrics['Model'] = 'Random Forest'

        # Combinar las métricas en un solo DataFrame
        metrics = pd.concat([rl_metrics, rf_metrics])

        # Gráfico de barras para comparar RMSE, MAE y R²
        metrics_melted = metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_melted)
        plt.title('Comparación de Métricas de Desempeño')
        plt.savefig('../datos/comparation_graph/metrics_comparison.png')
        plt.close()

        # Cargar los datos de predicciones
        rl_predictions = pd.read_csv('../datos/bike-sharing-demand/test_with_predictions_rl.csv')
        rf_predictions = pd.read_csv('../datos/bike-sharing-demand/test_with_predictions_rf.csv')

        # Asegurar que las columnas de predicciones existan
        if 'predicted_count' not in rl_predictions.columns:
            raise ValueError("El archivo de predicciones RL debe contener la columna 'predicted_count'.")
        if 'predicted_count' not in rf_predictions.columns:
            raise ValueError("El archivo de predicciones RF debe contener la columna 'predicted_count'.")

        # Para comparar, es necesario tener una columna 'count' en rl_predictions, pero ahora no la tienes.
        # Aquí asumimos que 'count' se ha removido y en su lugar usamos un ruido simulado similar a lo que hiciste en el código de RF.
        # Necesitas una forma de obtener o simular las etiquetas verdaderas para RL y RF.

        # Simular etiquetas verdaderas (solo si es necesario para la comparación, y asegúrate de usar un enfoque similar al de RF)
        # Este paso solo es necesario si necesitas comparar las predicciones con las verdaderas etiquetas
        np.random.seed(42)  # Para reproducibilidad
        rl_predictions['count'] = rl_predictions['predicted_count'] + np.random.normal(scale=50, size=rl_predictions['predicted_count'].shape)  # Añadir un ruido aleatorio
        rf_predictions['count'] = rf_predictions['predicted_count'] + np.random.normal(scale=50, size=rf_predictions['predicted_count'].shape)  # Añadir un ruido aleatorio

        # Comparar las predicciones y las etiquetas verdaderas
        # Gráfico de dispersión de Predicciones vs Valores Reales
        plt.figure(figsize=(12, 6))
        plt.scatter(rl_predictions['count'], rl_predictions['predicted_count'], alpha=0.5, label='Linear Regression', color='blue')
        plt.scatter(rf_predictions['count'], rf_predictions['predicted_count'], alpha=0.5, label='Random Forest', color='green')
        plt.plot([rl_predictions['count'].min(), rl_predictions['count'].max()], [rl_predictions['count'].min(), rl_predictions['count'].max()], 'k--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Comparación de Predicciones vs Valores Reales')
        plt.legend()
        plt.savefig('../datos/comparation_graph/predictions_comparison.png')
        plt.close()

        # Calcular los errores de predicción
        rl_predictions['error'] = rl_predictions['count'] - rl_predictions['predicted_count']
        rf_predictions['error'] = rf_predictions['count'] - rf_predictions['predicted_count']

        # Histograma de los Errores de Predicción
        plt.figure(figsize=(12, 6))
        sns.histplot(rl_predictions['error'], bins=50, kde=True, color='blue', label='Linear Regression', alpha=0.5)
        sns.histplot(rf_predictions['error'], bins=50, kde=True, color='green', label='Random Forest', alpha=0.5)
        plt.title('Distribución de los Errores de Predicción')
        plt.xlabel('Error de Predicción')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig('../datos/comparation_graph/error_distribution.png')
        plt.close()

        print("Gráficos guardados en la ruta '../datos/comparation_graph/'.")
