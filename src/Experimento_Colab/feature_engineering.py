# -*- coding: utf-8 -*-
import os
import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- Configuración y Variables Globales ---

# Rutas de archivos y directorios
BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, 'data')
DB_PATH = os.path.join(BASE_PATH, 'db')
MODELS_PATH = os.path.join(BASE_PATH, 'models')

# Crear directorios si no existen
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Nombres de los archivos de datos
RAW_DATA_FILE = 'competencia_01_crudo.csv'
CURATED_DATA_FILE = 'competencia_01.csv'
FEATURE_ENGINEERED_FILE = 'competencia_01_fe.csv'
FEATURE_ENGINEERED_RF_FILE = 'competencia_01_fe_rf.csv'

# Semillas para reproducibilidad (necesaria en el paso de Random Forest)
SEEDS = [761249, 762001, 763447, 762233, 761807]

# --- Funciones del Pipeline ---

def step1_create_target_variable():
    """
    Carga los datos crudos, define la variable objetivo usando SQL y guarda el dataset curado.
    """
    print("--- Paso 1: Definiendo la Variable Objetivo ---")
    
    raw_file_path = os.path.join(DATA_PATH, RAW_DATA_FILE)
    if not os.path.exists(raw_file_path):
        print(f"ERROR: No se encontró el archivo de datos crudos en {raw_file_path}")
        print("Por favor, coloque 'competencia_01_crudo.csv' en el directorio 'data'.")
        return False

    conn = duckdb.connect(database=':memory:', read_only=False)
    
    conn.execute(
        f"CREATE OR REPLACE TABLE competencia_01_crudo AS\n"
        f"SELECT * FROM read_csv_auto('{raw_file_path}');"
    )

    query = (
        "CREATE OR REPLACE TABLE clases_ternarias AS\n"
        "WITH ranked_clients AS (\n"
        "    SELECT\n"
        "        foto_mes,\n"
        "        numero_de_cliente,\n"
        "        row_number() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes DESC) as rank\n"
        "    FROM competencia_01_crudo\n"
        ")\n"
        "SELECT\n"
        "    foto_mes,\n"
        "    numero_de_cliente,\n"
        "    CASE\n"
        "        WHEN rank = 1 AND foto_mes < 202106 THEN 'BAJA+1'\n"
        "        WHEN rank = 2 AND foto_mes < 202105 THEN 'BAJA+2'\n"
        "        WHEN rank = 3 AND foto_mes < 202104 THEN 'BAJA+3'\n"
        "        WHEN rank = 4 AND foto_mes < 202103 THEN 'BAJA+4'\n"
        "        WHEN rank = 5 AND foto_mes < 202102 THEN 'BAJA+5'\n"
        "        ELSE 'CONTINUA'\n"
        "    END AS clase_ternaria\n"
        "FROM ranked_clients;\n"
        "\n"
        "CREATE OR REPLACE TABLE competencia_01_curado AS\n"
        "SELECT\n"
        "    crudo.*,\n"
        "    clase.clase_ternaria\n"
        "FROM competencia_01_crudo AS crudo\n"
        "INNER JOIN clases_ternarias AS clase\n"
        "    ON crudo.foto_mes = clase.foto_mes AND crudo.numero_de_cliente = clase.numero_de_cliente;"
    )
    conn.execute(query)

    output_path = os.path.join(DATA_PATH, CURATED_DATA_FILE)
    conn.execute(f"COPY competencia_01_curado TO '{output_path}' (FORMAT CSV, HEADER);")
    
    print(f"Datos curados guardados en '{output_path}'")
    conn.close()
    return True

def step2_engineer_features():
    """
    Añade un conjunto extenso de características.
    Los bucles de creación de columnas han sido optimizados para evitar advertencias de rendimiento.
    """
    print("\n--- Paso 2: Ingeniería Extensiva de Características ---")
    
    input_file_path = os.path.join(DATA_PATH, CURATED_DATA_FILE)
    if not os.path.exists(input_file_path):
        print(f"ERROR: No se encontró el archivo de datos curados en {input_file_path}")
        return False
        
    df = pd.read_csv(input_file_path, low_memory=False)
    
    print("  - Generando características intra-mes (combinaciones Visa/Mastercard)...")
    for col_group in [
        ('Master_msaldototal', 'Visa_msaldototal', 'vm_msaldototal'),
        ('Master_mconsumospesos', 'Visa_mconsumospesos', 'vm_mconsumospesos'),
        ('Master_mlimitecompra', 'Visa_mlimitecompra', 'vm_mlimitecompra'),
        ('Master_madelantopesos', 'Visa_madelantopesos', 'vm_madelantopesos'),
        ('Master_mpagado', 'Visa_mpagado', 'vm_mpagado')
    ]:
        if col_group[0] in df.columns and col_group[1] in df.columns:
            df[col_group[2]] = df[col_group[0]].fillna(0) + df[col_group[1]].fillna(0)
    
    if 'vm_msaldototal' in df.columns and 'vm_mlimitecompra' in df.columns:
        df['vmr_msaldototal_div_mlimitecompra'] = (df['vm_msaldototal'] / df['vm_mlimitecompra']).replace([np.inf, -np.inf], 0)
    if 'vm_mconsumospesos' in df.columns and 'vm_mlimitecompra' in df.columns:
        df['vmr_mconsumospesos_div_mlimitecompra'] = (df['vm_mconsumospesos'] / df['vm_mlimitecompra']).replace([np.inf, -np.inf], 0)
    
    df.sort_values(['numero_de_cliente', 'foto_mes'], inplace=True)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = ['numero_de_cliente', 'foto_mes']
    cols_to_process = [col for col in numeric_cols if col not in exclude_cols]

    # Generación de lags (rezagos) y deltas (diferencias)
    print(f"  - Generando lags y deltas para {len(cols_to_process)} columnas numéricas...")
    lag_delta_features = {}
    for col in cols_to_process:
        for lag in [1, 2]:
            lag_serie = df.groupby('numero_de_cliente')[col].shift(lag)
            lag_delta_features[f'{col}_lag{lag}'] = lag_serie
            lag_delta_features[f'{col}_delta{lag}'] = df[col] - lag_serie
    df = pd.concat([df, pd.DataFrame(lag_delta_features)], axis=1)

    # Generación de características de tendencia histórica (ventana móvil)
    print("  - Generando características de tendencia histórica (ventana móvil)...")
    def get_slope(y):
        if y.isnull().all() or len(y.dropna()) < 2: return np.nan
        x = np.arange(len(y))
        y = y.values
        not_nan_mask = ~np.isnan(y)
        if not_nan_mask.sum() < 2: return np.nan
        slope, _ = np.polyfit(x[not_nan_mask], y[not_nan_mask], 1)
        return slope

    rolling_features = {}
    for col in cols_to_process:
        for window in [3]:
            rolling_grp = df.groupby('numero_de_cliente')[col].rolling(window, min_periods=2)
            rolling_features[f'{col}_rol{window}_min'] = rolling_grp.min().reset_index(level=0, drop=True)
            rolling_features[f'{col}_rol{window}_max'] = rolling_grp.max().reset_index(level=0, drop=True)
            rolling_features[f'{col}_rol{window}_mean'] = rolling_grp.mean().reset_index(level=0, drop=True)
            rolling_features[f'{col}_rol{window}_std'] = rolling_grp.std().reset_index(level=0, drop=True)
            if col in ['mrentabilidad', 'mrentabilidad_annual', 'vm_msaldototal', 'ctrx_quarter', 'cproductos']:
                rolling_features[f'{col}_rol{window}_tend'] = rolling_grp.apply(get_slope, raw=False).reset_index(level=0, drop=True)
    df = pd.concat([df, pd.DataFrame(rolling_features)], axis=1)

    # Aplicación de rangos para corregir el "data drift"
    print("  - Aplicando 'rank_cero_fijo' para corrección de data drift...")
    rank_features = {}
    for col in cols_to_process:
        new_rank_col = f'{col}_rank_cf'
        rank_serie = pd.Series(index=df.index, dtype=float)
        pos_mask = df[col] > 0
        if pos_mask.any():
            rank_serie.loc[pos_mask] = df.loc[pos_mask].groupby('foto_mes')[col].rank(pct=True)
        neg_mask = df[col] < 0
        if neg_mask.any():
            rank_serie.loc[neg_mask] = -df.loc[neg_mask].groupby('foto_mes')[col].rank(pct=True, ascending=False)
        rank_serie.fillna(0, inplace=True)
        rank_features[new_rank_col] = rank_serie
    df = pd.concat([df, pd.DataFrame(rank_features)], axis=1)
        
    df.fillna(0, inplace=True)
    output_path = os.path.join(DATA_PATH, FEATURE_ENGINEERED_FILE)
    df.to_csv(output_path, index=False)
    
    print(f"Datos con nuevas características guardados en '{output_path}'")
    return True

def step2b_add_rf_features():
    """
    Entrena un modelo Random Forest y utiliza los índices de sus hojas como nuevas
    características categóricas. Guarda el resultado en un nuevo archivo.
    """
    print("\n--- Paso 2b: Ingeniería de Atributos con Random Forest ---")
    input_file_path = os.path.join(DATA_PATH, FEATURE_ENGINEERED_FILE)
    if not os.path.exists(input_file_path):
        print(f"ERROR: No se encontró el archivo de características en {input_file_path}")
        return False

    df = pd.read_csv(input_file_path, low_memory=False)
    df['target_baja_1_2'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)
    
    rf_train_months = [202101, 202102, 202103]
    train = df[df['foto_mes'].isin(rf_train_months)]
    
    if len(train['target_baja_1_2'].unique()) < 2:
        print("ERROR en step2b: El conjunto de entrenamiento para RF contiene una sola clase.")
        return False
        
    non_feature_cols = ['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'target_baja_1_2']
    features = [c for c in df.columns if c not in non_feature_cols]
    
    X_train = train[features]
    y_train = train['target_baja_1_2']
    
    print("  - Entrenando LightGBM en modo Random Forest...")
    rf_model = lgb.LGBMClassifier(
        boosting_type='rf', objective='binary', n_estimators=100, num_leaves=31,
        max_depth=-1, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        n_jobs=-1, random_state=SEEDS[0]
    )
    rf_model.fit(X_train, y_train)
    
    print("  - Prediciendo los índices de las hojas para todo el dataset...")
    leaf_indices = rf_model.predict(df[features], pred_leaf=True)
    
    for i in range(leaf_indices.shape[1]):
        df[f'rf_leaf_{i}'] = leaf_indices[:, i]
        
    df.drop(columns=['target_baja_1_2'], inplace=True)
    
    output_path = os.path.join(DATA_PATH, FEATURE_ENGINEERED_RF_FILE)
    df.to_csv(output_path, index=False)
    
    print(f"Se agregaron {leaf_indices.shape[1]} características de hojas de RF y se guardó en '{output_path}'.")
    
    return True

if __name__ == "__main__":
    print("Iniciando Pipeline de Ingeniería de Características (hasta atributos de RF)...")
    
    run_step1 = not os.path.exists(os.path.join(DATA_PATH, CURATED_DATA_FILE))
    run_step2 = not os.path.exists(os.path.join(DATA_PATH, FEATURE_ENGINEERED_FILE))
    run_step2b = not os.path.exists(os.path.join(DATA_PATH, FEATURE_ENGINEERED_RF_FILE))
    
    if run_step1:
        if not step1_create_target_variable(): exit()
        
    print("\nPipeline de Ingeniería de Características finalizado con éxito.")
