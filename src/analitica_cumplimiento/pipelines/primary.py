"""
Lógica del pipeline primary
"""

from typing import Any, Dict

import pandas as pd
import pycountry

from src.analitica_cumplimiento.utils import Utils
logger = Utils.setup_logging()


class PipelinePrimary:

    # 1. recategorize_data_pd
    @staticmethod
    def recategorize_data_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Recategoriza los valores de los campos especificados en el DataFrame según el mapeo proporcionado en los parámetros.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame de pandas cuyos valores de campos serán recategorizados.
        params: Dict[str, Any] 
            Diccionario de parámetros primary.

        Returns
        -------
        pd.DataFrame: DataFrame con los campos recategorizados y convertidos a enteros.
        """
        # Registra un mensaje de información indicando el inicio del proceso de recategorización
        logger.info("Iniciando la recategorización de valores de campos...")

        # Parámetros
        recategorization_params = {
            'PEP': params['PEP'],
            'RIESGO': params['RIESGO']
        }

        # Recategoriza los valores de los campos según lo especificado en el diccionario
        for column, mapping in recategorization_params.items():
            if column in df.columns:
                # Mapea los valores según el diccionario
                df[column] = df[column].map(mapping)
                # Reemplaza NaN con -1 antes de la conversión a int64
                df[column] = df[column].fillna(-1).astype('int64')
                logger.info(f"Columna {column} recategorizada y convertida a int64")

        # Registra un mensaje de información indicando la finalización del proceso
        logger.info("Recategorización de valores de campos finalizada.")

        # Retorna el DataFrame con los campos recategorizados
        return df
    
    # 2. recategorize_countrys_pd
    @staticmethod
    def recategorize_countrys_pd(
        df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Recategoriza los valores de las columnas que contienen códigos de países en formato ISO 3166-1 alfa-2
        o códigos numéricos ISO 3166-1 a sus correspondientes códigos numéricos ISO 3166-1.
        Los valores nulos o no reconocidos se asignan a -1.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame que contiene las columnas con códigos de países en formato ISO 3166-1 alfa-2 o numéricos.
        params: Dict[str, Any] 
            Diccionario de parámetros primary. 

        Returns
        -------
        pd.DataFrame
            DataFrame con las recategorizaciones realizadas.
        """

        # Registra un mensaje de información indicando el inicio del proceso de recategorización de países
        logger.info("Iniciando el proceso de recategorización de países...")

        # Extrae las columnas de países especificadas en los parámetros
        country_columns = params['countrys_cols']

        # Mapeo personalizado para códigos específicos como AN, UK, DI, etc.
        custom_mapping = params['custom_country_mapping']

        # Recorre cada columna especificada en los parámetros
        for col in country_columns:
            if col in df.columns:
                def map_country_code(x):
                    if pd.isna(x):
                        return -1
                    elif x.isdigit() and len(x) in [2, 3]:  # Para códigos numéricos que ya son ISO
                        return int(x)
                    elif x in custom_mapping:  # Verifica si el código está en el mapeo personalizado
                        return custom_mapping[x]
                    else:
                        country = pycountry.countries.get(alpha_2=x)
                        return int(country.numeric) if country else -1
                
                # Aplica la función de mapeo a cada valor en la columna
                df[col] = df[col].apply(map_country_code)
                logger.info(f"Recategorizada la columna '{col}' a códigos numéricos ISO 3166-1.")
        
        # Registra un mensaje de información indicando el fin del proceso de recategorización de países
        logger.info("Finalizado el proceso de recategorización de países.")

        # Retorna el DataFrame con las recategorizaciones realizadas
        return df

    # 3. impute_missing_values_pd
    @staticmethod
    def impute_missing_values_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Imputa los valores ausentes en un campo de tipo datetime64[ns] con la fecha más común en dicho campo.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame de pandas que contiene el campo con fechas para imputar.
        params: Dict[str, Any] 
            Diccionario de parámetros primary.

        Returns
        -------
        pd.DataFrame: DataFrame con valores ausentes imputados con la fecha más común.
        """
        # Registra un mensaje de información indicando el inicio del proceso de imputación
        logger.info("Iniciando la imputación de valores ausentes en el campo de fecha...")

        # Obtiene el nombre del campo de fecha desde los parámetros
        date_column = params['missing_cols']

        # Verifica si el campo existe en el DataFrame
        if date_column not in df.columns:
            logger.error(f"El campo {date_column} no se encuentra en el DataFrame.")
            raise ValueError(f"El campo {date_column} no se encuentra en el DataFrame.")

        # Calcula la fecha más común (moda) en el campo
        most_common_date = df[date_column].mode()[0]

        # Imputa los valores ausentes con la fecha más común sin usar inplace
        df[date_column] = df[date_column].fillna(most_common_date)

        # Registra un mensaje de información indicando la fecha más común utilizada
        logger.info(f"Los valores ausentes en el campo {date_column} han sido imputados con la fecha más común: {most_common_date}.")

        # Retorna el DataFrame con los valores imputados
        return df