import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
import warnings
import sys
import joblib  # Necesario para guardar los estudios de Optuna

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reducir verbosidad de Optuna

# -----------------------------------------------------------------------------
# FASE 1: CONFIGURACIÓN Y PREPARACIÓN DE DATOS (ESTRUCTURA FINAL)
# -----------------------------------------------------------------------------
print("FASE 1: Configuración y Preparación de Datos")

# --- Constantes de Negocio y Semillas ---
DATA_PATH = './data/'
DATA_FILE = 'competencia_01.csv'
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
SEEDS = [761249, 762001, 763447, 762233, 761807]
N_TRIALS_OPTUNA = 500 # Número de trials por experimento (puedes aumentarlo a 50 o más)
TIMEOUT_OPTUNA = 1800 # 30 minutos por experimento

# --- Carga de datos usando Pandas ---
try:
    print(f"Cargando el archivo: {DATA_PATH}{DATA_FILE}")
    full_df = pd.read_csv(f"{DATA_PATH}{DATA_FILE}")
except FileNotFoundError:
    print(f"Error: El archivo '{DATA_FILE}' no se encontró en la ruta '{DATA_PATH}'.")
    sys.exit(1)

# --- Separación Temporal de Datos (NUEVA ESTRUCTURA) ---
# Estos son los datos *crudos*. Se procesarán en cada loop del experimento.
df_predict = full_df[full_df['foto_mes'] == 202106].copy()
df_train_optuna = full_df[full_df['foto_mes'].isin([202101, 202102])].copy()
df_val_optuna = full_df[full_df['foto_mes'] == 202103].copy()
df_test = full_df[full_df['foto_mes'] == 202104].copy()
df_train_final = full_df[full_df['foto_mes'].isin([202101, 202102, 202103, 202104])].copy()

print("Datos crudos cargados y separados temporalmente.")

# -----------------------------------------------------------------------------
# FASE 2 (PARCIAL): FUNCIONES REUTILIZABLES
# -----------------------------------------------------------------------------

def create_features(df, target_classes):
    """
    Aplica preprocesamiento y crea nuevas características de negocio.
    Ahora acepta una lista de 'target_classes' para definir el target.
    """
    d = df.copy()
    if 'clase_ternaria' in d.columns:
        # Usar .isin() para chequear contra la lista de clases objetivo
        d['target'] = np.where(d['clase_ternaria'].isin(target_classes), 1, 0)
    
    # Aquí se podrían agregar más features si fuera necesario
    # ...
    
    return d

# --- Funciones de Cálculo de Ganancia (no cambian) ---
def calculate_max_profit_from_curve(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    df['ganancia_individual'] = np.where(df['y_true'] == 1, GANANCIA_ACIERTO - COSTO_ESTIMULO, -COSTO_ESTIMULO)
    df['ganancia_acumulada'] = df['ganancia_individual'].cumsum()
    max_ganancia = df['ganancia_acumulada'].max()
    return max_ganancia if max_ganancia > 0 else 0

def find_optimal_n(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    df['ganancia_individual'] = np.where(df['y_true'] == 1, GANANCIA_ACIERTO - COSTO_ESTIMULO, -COSTO_ESTIMULO)
    df['ganancia_acumulada'] = df['ganancia_individual'].cumsum()
    if df['ganancia_acumulada'].max() > 0:
        return df['ganancia_acumulada'].idxmax() + 1
    return 0

# -----------------------------------------------------------------------------
# DEFINICIÓN DE EXPERIMENTOS
# -----------------------------------------------------------------------------

experiments = [
    {'name': 'B1_only', 'targets': ['BAJA+1']},
    {'name': 'B2_only', 'targets': ['BAJA+2']},
    {'name': 'B1_B2', 'targets': ['BAJA+1', 'BAJA+2']},
    {'name': 'B1_B2_B3', 'targets': ['BAJA+1', 'BAJA+2', 'BAJA+3']},
    {'name': 'B1_to_B4', 'targets': ['BAJA+1', 'BAJA+2', 'BAJA+3', 'BAJA+4']},
    {'name': 'B1_to_B5', 'targets': ['BAJA+1', 'BAJA+2', 'BAJA+3', 'BAJA+4', 'BAJA+5']},
]

# --- Columnas a excluir (comunes a todos los experimentos) ---
features_to_exclude = ['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'target', 'total_deudas', 'uso_limite_visa', 'uso_limite_master']
# Tomamos las features del primer dataframe (asumimos que son las mismas)
temp_features = [col for col in df_train_optuna.columns if col not in features_to_exclude]


# -----------------------------------------------------------------------------
# BUCLE PRINCIPAL DE EXPERIMENTOS
# -----------------------------------------------------------------------------

for experiment in experiments:
    exp_name = experiment['name']
    exp_targets = experiment['targets']
    
    print(f"\n\n{'='*80}")
    print(f"INICIANDO EXPERIMENTO: {exp_name}")
    print(f"Clases objetivo (Target=1): {exp_targets}")
    print(f"{'='*80}\n")

    # --- FASE 2: INGENIERÍA DE CARACTERÍSTICAS (Específica del experimento) ---
    print("FASE 2: Ingeniería de Características")
    
    # Aplicar a todos los conjuntos de datos con la definición de target actual
    df_train_optuna_fe = create_features(df_train_optuna, exp_targets)
    df_val_optuna_fe = create_features(df_val_optuna, exp_targets)
    df_test_fe = create_features(df_test, exp_targets)
    df_train_final_fe = create_features(df_train_final, exp_targets)
    df_predict_fe = create_features(df_predict, exp_targets) # Este no tendrá 'target'
    
    # Definir variables del modelo
    features = temp_features # Usamos las features definidas fuera del loop

    X_train_optuna = df_train_optuna_fe[features]
    y_train_optuna = df_train_optuna_fe['target']
    X_val_optuna = df_val_optuna_fe[features]
    y_val_optuna = df_val_optuna_fe['target']
    X_test = df_test_fe[features]
    y_test = df_test_fe['target']

    print(f"Número de características para el modelo: {len(features)}")
    print(f"Distribución del target en Train (Optuna): \n{y_train_optuna.value_counts(normalize=True)}")
    print(f"Distribución del target en Val (Optuna): \n{y_val_optuna.value_counts(normalize=True)}")

    # --- FASE 3: OPTIMIZACIÓN CON PROMEDIO DE SEMILLAS ---
    print("\nFASE 3: Optimización con Promedio de Semillas por Trial")

    def objective(trial):
        # Esta función 'objective' se redefine en cada loop
        # y captura las variables de su scope (X_train_optuna, y_train_optuna, etc.)
        
        undersampling_ratio = trial.suggest_int('undersampling_ratio', 1, 30)
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50)
        }
        
        gains_per_seed = []
        for seed in SEEDS:
            # Re-crear el set de training submuestreado para cada semilla
            train_data = pd.concat([X_train_optuna, y_train_optuna], axis=1)
            bajas = train_data[train_data['target'] == 1]
            continuas = train_data[train_data['target'] == 0]
            
            if len(bajas) == 0:
                # Caso extremo: no hay targets positivos en el set de train
                return 0 
                
            n_continuas_to_keep = int(len(bajas) * undersampling_ratio)
            
            if n_continuas_to_keep > len(continuas):
                 n_continuas_to_keep = len(continuas) # Evitar error si hay pocos negativos
            
            continuas_undersampled = continuas.sample(n=n_continuas_to_keep, random_state=seed)
            train_undersampled = pd.concat([bajas, continuas_undersampled])
            X_train_us, y_train_us = train_undersampled[features], train_undersampled['target']
            
            params['random_state'] = seed
            model = lgb.LGBMClassifier(**params)
            
            model.fit(X_train_us, y_train_us, 
                      eval_set=[(X_val_optuna, y_val_optuna)], 
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            
            val_probs = model.predict_proba(X_val_optuna)[:, 1]
            gains_per_seed.append(calculate_max_profit_from_curve(y_val_optuna.values, val_probs))
            
        return np.mean(gains_per_seed)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, timeout=TIMEOUT_OPTUNA)
    
    # Guardar el estudio de Optuna
    study_filename = f'optuna_study_{exp_name}.pkl'
    joblib.dump(study, study_filename)

    print(f"\n--- Optimización Robusta Completada para {exp_name} ---")
    print(f"Estudio guardado en: {study_filename}")
    print(f"Mejor ganancia promedio en validación: {study.best_value:,.0f}")
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)

    # --- FASE 4: TESTEO DEL MEJOR MODELO ENCONTRADO ---
    print(f"\nFASE 4: Testeo del Mejor Modelo (Train: 01-02, Test: 04) para {exp_name}")
    
    best_params = study.best_params.copy() # Usar .copy() para seguridad
    optimal_undersampling_ratio = best_params.pop('undersampling_ratio')
    best_model_params = best_params

    # Preparar datos de testeo (usando los datos de Optuna)
    train_data_test = pd.concat([X_train_optuna, y_train_optuna], axis=1)
    bajas_test = train_data_test[train_data_test['target'] == 1]
    continuas_test = train_data_test[train_data_test['target'] == 0]
    n_continuas_to_keep_test = int(len(bajas_test) * optimal_undersampling_ratio)
    
    if n_continuas_to_keep_test > len(continuas_test):
        n_continuas_to_keep_test = len(continuas_test)
        
    continuas_undersampled_test = continuas_test.sample(n=n_continuas_to_keep_test, random_state=SEEDS[0])
    train_optimal_us_test = pd.concat([bajas_test, continuas_undersampled_test])
    X_train_optimal_us_test, y_train_optimal_us_test = train_optimal_us_test[features], train_optimal_us_test['target']

    test_model = lgb.LGBMClassifier(**best_model_params, n_estimators=1000, random_state=SEEDS[0])
    test_model.fit(X_train_optimal_us_test, y_train_optimal_us_test)
    
    test_probs = test_model.predict_proba(X_test)[:, 1]
    test_ganancia = calculate_max_profit_from_curve(y_test.values, test_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    optimal_N_for_production = find_optimal_n(y_test.values, test_probs)

    print(f"Resultados en el conjunto de Test (202104) para {exp_name}:")
    print(f"  - Ganancia estimada: {test_ganancia:,.0f}")
    print(f"  - AUC: {test_auc:.4f}")
    print(f"  - Número de envíos óptimo para producción: {optimal_N_for_production}")

    # --- FASE 5: ENTRENAMIENTO DEL MODELO FINAL Y PREDICCIÓN PARA 202106 ---
    print(f"\nFASE 5: Entrenamiento Final (Train: 01-04) y Predicción para {exp_name}")
    
    # Usar el set de entrenamiento final FE
    final_train_data = pd.concat([df_train_final_fe[features], df_train_final_fe['target']], axis=1)
    final_bajas = final_train_data[final_train_data['target'] == 1]
    final_continuas = final_train_data[final_train_data['target'] == 0]
    final_n_continuas_to_keep = int(len(final_bajas) * optimal_undersampling_ratio)
    
    if final_n_continuas_to_keep > len(final_continuas):
        final_n_continuas_to_keep = len(final_continuas)
        
    final_continuas_undersampled = final_continuas.sample(n=final_n_continuas_to_keep, random_state=SEEDS[0])
    final_train_undersampled = pd.concat([final_bajas, final_continuas_undersampled])
    X_train_final, y_train_final = final_train_undersampled[features], final_train_undersampled['target']

    final_model = lgb.LGBMClassifier(**best_model_params, n_estimators=2000, random_state=SEEDS[0])
    final_model.fit(X_train_final, y_train_final)
    print("Modelo final re-entrenado con todos los datos históricos (submuestreados).")

    # Usar los datos de 202106 para predecir
    X_predict = df_predict_fe[features]
    customer_ids = df_predict_fe['numero_de_cliente']
    final_probabilities = final_model.predict_proba(X_predict)[:, 1]

    df_final_pred = pd.DataFrame({'numero_de_cliente': customer_ids, 'prob': final_probabilities})
    df_final_pred = df_final_pred.sort_values('prob', ascending=False)
    
    # Seleccionar los N óptimos encontrados en la FASE 4
    clientes_a_contactar = df_final_pred.head(optimal_N_for_production)['numero_de_cliente']
    df_final_pred['prediction'] = np.where(df_final_pred['numero_de_cliente'].isin(clientes_a_contactar), 1, 0)

    # Guardar el archivo de salida con nombre único
    output_filename = f'predicciones_{exp_name}_202106.csv'
    output_df = df_final_pred[['numero_de_cliente', 'prediction']]
    output_df.to_csv(output_filename, index=False)

    print(f"\nArchivo de predicciones '{output_filename}' generado con éxito.")
    print(f"Se utilizó un N óptimo de: {optimal_N_for_production} envíos.")
    print(f"Total de clientes a contactar (prediction=1): {output_df['prediction'].sum()} de {len(output_df)}")
    print(f"--- EXPERIMENTO {exp_name} COMPLETADO ---")

print(f"\n\n{'='*80}")
print("TODOS LOS EXPERIMENTOS HAN FINALIZADO.")
print(f"{'='*80}")

