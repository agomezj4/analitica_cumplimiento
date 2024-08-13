"""
Lógica del pipeline intermediate
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import re

from src.analitica_cumplimiento.utils import Utils
logger = Utils.setup_logging()


class PipelineIntermediate:

    # 1. Cambiar tipado
    @staticmethod
    def change_data_type_pd(
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        params: Dict[str, Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convierte los tipos de columnas en dos DataFrames según los tipos especificados en los parámetros.

        Parameters
        ----------
        df1 : pandas.DataFrame
            Primer DataFrame cuyos tipos de columna se convertirán.
        df2 : pandas.DataFrame
            Segundo DataFrame cuyos tipos de columna se convertirán.
        params: Dict[str, Dict[str, Any]]
            Diccionario de parámetros intermediate.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]: Tupla con los dos DataFrames con los tipos de columna convertidos.
        """

        # Función interna para convertir tipos en un DataFrame
        def convert_types(df: pd.DataFrame, type_params: Dict[str, List[str]]) -> pd.DataFrame:
            if 'date_columns' in type_params:
                for column in type_params['date_columns']:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    logger.info(f"Columna {column} convertida a datetime")

            if 'int_columns' in type_params:
                for column in type_params['int_columns']:
                    df[column] = df[column].astype('int64', errors='ignore')
                    logger.info(f"Columna {column} convertida a int64")

            if 'float_columns' in type_params:
                for column in type_params['float_columns']:
                    df[column] = df[column].astype('float64', errors='ignore')
                    logger.info(f"Columna {column} convertida a float64")

            return df

        # Convierte tipos en el primer DataFrame
        if 'df1' in params:
            logger.info("Iniciando la conversión de tipos de columnas para el primer DataFrame...")
            df1 = convert_types(df1, params['df1'])
            logger.info("Conversión de tipos de columnas para el primer DataFrame finalizada.")

        # Convierte tipos en el segundo DataFrame
        if 'df2' in params:
            logger.info("Iniciando la conversión de tipos de columnas para el segundo DataFrame...")
            df2 = convert_types(df2, params['df2'])
            logger.info("Conversión de tipos de columnas para el segundo DataFrame finalizada.")

        # Retorna los DataFrames con los tipos de columna convertidos
        return df1, df2
    

    # 2. Cambiar nombres
    @staticmethod
    def change_data_name_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Cambia los nombres de las columnas especificadas en el DataFrame según los nombres proporcionados en los parámetros.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame de pandas cuyos nombres de columnas serán cambiados.
        params: Dict[str, Any] 
            Diccionario de parámetros intermediate..

        Returns
        -------
        pd.DataFrame: DataFrame con los nombres de las columnas modificados.
        """
        # Registra un mensaje de información indicando el inicio del cambio de nombres de columnas
        logger.info("Iniciando el cambio de nombres de columnas...")

        # Cambia los nombres de las columnas según lo especificado en el diccionario
        df = df.rename(columns=params['columns_name'])
        
        # Log para cada cambio realizado
        for old_name, new_name in params['columns_name'].items():
            logger.info(f"Columna {old_name} cambiada a {new_name}")
        
        # Registra un mensaje de información indicando la finalización del proceso
        logger.info("Cambio de nombres de columnas finalizado.")

        # Retorna el DataFrame con los nombres de columnas modificados
        return df


    # 3. Estandarizar cadenas de texto
    @staticmethod
    def standarize_data_str_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Estandariza los valores de cadenas de texto en las columnas especificadas eliminando comillas simples o dobles.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame de pandas cuyos valores de cadenas de texto serán estandarizados.
        params: Dict[str, Any] 
            Diccionario de parámetros intermediate.

        Returns
        -------
        pd.DataFrame: DataFrame con los valores de cadenas de texto estandarizados.
        """
        # Registra un mensaje de información indicando el inicio de la estandarización de valores de cadenas de texto
        logger.info("Iniciando la estandarización de valores de cadenas de texto...")

        # Estandariza los valores de las columnas especificadas eliminando comillas simples o dobles
        if 'columns_to_standarize' in params:
            for column in params['columns_to_standarize']:
                df[column] = df[column].apply(lambda x: re.sub(r'["\']', '', x) if isinstance(x, str) else x)
                logger.info(f"Columna {column} estandarizada para eliminar comillas")
        
        # Registra un mensaje de información indicando la finalización del proceso
        logger.info("Estandarización de valores de cadenas de texto finalizada.")

        # Retorna el DataFrame con los valores de cadenas de texto estandarizados
        return df