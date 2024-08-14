"""
Lógica del pipeline detección de anomalías
"""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

from src.analitica_cumplimiento.utils import Utils
logger = Utils.setup_logging()


class PipelineAnomalias:

    # 1. Preparación de datos
    def df_filter_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filtra un DataFrame en base a una lista de variables y separa el DataFrame en columnas categóricas y numéricas.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame a filtrar y separar.
        params : Dict[str, Any]
            Diccionario de parámetros anomalías.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Una tupla con dos DataFrames: uno con columnas categóricas y otro con columnas numéricas.
        """
        # Registra un mensaje de información indicando el inicio del proceso de filtrado
        logger.info("Iniciando el proceso de filtrado y separación del DataFrame...")

        # Filtra el DataFrame basado en las variables proporcionadas
        filtered_df = df[params['cols_filter']]

        # Selecciona las columnas categóricas
        categorical_df = filtered_df.select_dtypes(include=['object', 'category', 'period[M]'])

        # Selecciona las columnas numéricas
        numerical_df = filtered_df.select_dtypes(include=['number'])

        # Registra el número de columnas categóricas y numéricas encontradas
        logger.info(f"Se han identificado {len(categorical_df.columns)} columnas categóricas.")
        logger.info(f"Se han identificado {len(numerical_df.columns)} columnas numéricas.")

        # Retorna la tupla de DataFrames categóricos y numéricos
        return categorical_df, numerical_df


    # 2. Escalamiento de datos
    @staticmethod
    def min_max_scaler_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas que se estandarizará.

        Returns
        -------
        pd.DataFrame
            DataFrame estandarizado.
        """
        logger.info("Iniciando la estandarización con Min-Max Scaler...")

        # Identificar las columnas numéricas
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns

        # Filtrar solo las columnas numéricas no binarias (excluyendo aquellas que solo toman valores 0 y 1)
        numeric_cols = [col for col in numeric_cols if not ((df[col].nunique() == 2) & (df[col].isin([0, 1]).sum() == len(df)))]

        # Crear una copia del DataFrame para evitar el SettingWithCopyWarning
        df_copy = df.copy()

        # Aplicar Min-Max Scaler solo a las columnas numéricas no binarias
        for col in numeric_cols:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            range_val = max_val - min_val
            if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
                df_copy[col] = df_copy[col].astype('float64')  # Convertir la columna a float64
                df_copy.loc[:, col] = (df_copy[col] - min_val) / range_val

        logger.info("Estandarización con Min-Max Scaler completada!")

        return df_copy


    # 3. Cáluclo de zscore
    @staticmethod
    def calculate_z_scores_pd(
        categorical_df: pd.DataFrame, 
        numerical_df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calcula la puntuación Z para todas las variables numéricas en un DataFrame, agrupadas por columnas específicas,
        y combina el resultado con un DataFrame categórico. Los valores NaN se tratan como cero.

        Parameters
        ----------
        categorical_df : pandas.DataFrame
            DataFrame de pandas que contiene las columnas categóricas.
        numerical_df : pandas.DataFrame
            DataFrame de pandas que contiene las columnas numéricas a las cuales se les calculará la puntuación Z.
        params : Dict[str, Any]
            Diccionario de parámetros para la detección de anomalías.

        Returns
        -------
        pd.DataFrame
            DataFrame combinado con las columnas categóricas originales, las columnas numéricas originales,
            y las columnas adicionales que contienen la puntuación Z.
        """
        # Registra un mensaje de información indicando el inicio del cálculo de la puntuación Z
        logger.info("Iniciando el cálculo de la puntuación Z para las variables numéricas...")

        # Parámetros
        groupby_cols = params['groupby_cols']
        min_group_size = params['min_group_size']  # Umbral mínimo de observaciones por grupo

        # Combinar DataFrames categórico y numérico antes de agrupar
        combined_df = pd.concat([categorical_df[groupby_cols], numerical_df], axis=1)

        # Crear un DataFrame para almacenar las puntuaciones Z
        z_scores_df = pd.DataFrame(index=combined_df.index)

        # Calcular la puntuación Z solo para las columnas numéricas en el DataFrame combinado
        for col in numerical_df.columns:
            grouped_data = combined_df.groupby(groupby_cols)[col]

            # Verifica que el grupo tenga al menos 'min_group_size' observaciones
            z_scores_df[f'z_score_{col}'] = grouped_data.transform(
                lambda x: zscore(x.dropna(), ddof=1) if len(x.dropna()) >= min_group_size else pd.Series([0]*len(x), index=x.index)
            ).fillna(0)  # Reemplaza NaN por 0

            # Registra un mensaje informativo para cada columna procesada
            logger.info(f"Calculada la puntuación Z para la columna {col}.")

        # Combinar las puntuaciones Z con el DataFrame categórico y numérico original
        result_df = pd.concat([categorical_df, numerical_df, z_scores_df], axis=1)

        # Registra un mensaje de información indicando la finalización del proceso
        logger.info("Cálculo de la puntuación Z completado.")

        # Retorna el DataFrame resultante con las puntuaciones Z añadidas
        return result_df


    # 4. Detectar anomalías
    @staticmethod
    def detectar_anomalias_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Detecta anomalías en un DataFrame basado en varios valores de contaminación y guarda los resultados.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los datos a analizar.
        params : Dict[str, Any]
            Diccionario de parámetros anomalías.
        Returns
        -------
        pd.DataFrame
            DataFrame original con las columnas de resultados de anomalías añadidas.
        """
        logger.info("Iniciando la detección de anomalías...")

        # Parámetros
        contamination_values = params['contamination_values']

        # Columnas relevantes para el modelo Isolation Forest
        relevant_columns = [col for col in df if col.startswith('z_score_')]
    
        # Convertir a numpy array
        numerical_data = df[relevant_columns].values

        # Iterar sobre cada valor de contaminación
        for contamination in contamination_values:
            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(numerical_data)

            # Obtener predicciones de anomalía
            pred = clf.predict(numerical_data)

            # Agregar las predicciones al DataFrame original
            df[f'anomaly_{contamination}'] = pred

        logger.info("Detección de anomalías completada.")
        
        return df


    # 5. Resumen de anomalías
    @staticmethod
    def procesar_anomalias_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Marca todas las transacciones como Anómala o No Anómala basado en un valor específico de contaminación
        y genera un resumen de anomalías para cada combinación de CUENTA y MES_ANIO en el DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los resultados de la detección de anomalías.
        params : Dict[str, Any]
            Diccionario de parámetros anomalías.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Una tupla con dos DataFrames: uno con las transacciones marcadas como Anómalas o No Anómalas
            y otro con el resumen de anomalías por CUENTA y MES_ANIO.
        """
        logger.info("Procesando anomalías...")

        # Parámetros
        contamination_values = params['contamination_values']
        contamination_value = params['contamination_value']

        # Crear una columna de marca de anomalía basada en el valor de contaminación seleccionado
        columna_anomalia = f'anomaly_{contamination_value}'
        df['MARCA_ANOMALIA'] = df[columna_anomalia].apply(lambda x: 'ANOMALO' if x == -1 else 'NO ANOMALO')

        # Seleccionar solo las columnas relevantes para el informe de transacciones
        resultado_df = df[['CUENTA', 'MES_ANIO', 'PAIS_ORIGEN_DESTINO_TRX', 'MONTO', 'MARCA_ANOMALIA']]

        # Crear una lista para almacenar el resumen de anomalías
        resumen_anomalias_list = []

        # Agrupar por CUENTA y MES_ANIO una sola vez para evitar repetir operaciones
        grouped = df.groupby(['CUENTA', 'MES_ANIO'])

        # Iterar sobre cada grupo para generar el resumen
        for (cuenta, mes_anio), sub_df in grouped:
            resumen = {'CUENTA': cuenta, 'MES_ANIO': mes_anio}
            for contamination in contamination_values:
                resumen[f'anomaly_{contamination}_anomalas'] = (sub_df[f'anomaly_{contamination}'] == -1).sum()
                resumen[f'anomaly_{contamination}_no_anomalas'] = (sub_df[f'anomaly_{contamination}'] == 1).sum()

            resumen_anomalias_list.append(resumen)

        # Convertir la lista de resumen de anomalías en un DataFrame
        resumen_anomalias = pd.DataFrame(resumen_anomalias_list)

        logger.info("Procesamiento de anomalías completado.")

        return resultado_df, resumen_anomalias

   