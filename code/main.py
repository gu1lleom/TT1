from merge import MergeData
from eda import EDA
from RL import RL
from RF import RF
from compare import CompareModels

def main():
    # Crear instancias de las clases
    merge_data = MergeData()
    eda = EDA()
    rl = RL()
    rf = RF()
    compare_models = CompareModels()
    
    # Ejecutar los métodos run() de cada clase en el orden correcto
    print("Ejecutando MergeData...")
    merge_data.run()
    
    print("Ejecutando EDA...")
    eda.run()
    
    print("Ejecutando Reinforcement Learning...")
    rl.run()
    
    print("Ejecutando Random Forest Model...")
    rf.run()
    
    print("Ejecutando Comparación de Modelos...")
    compare_models.run()
    
    print("Todos los procesos han sido ejecutados.")

if __name__ == "__main__":
    main()