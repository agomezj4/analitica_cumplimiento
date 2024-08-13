"""
Lógica del pipeline primary
"""

from typing import Any, Dict, Tuple

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
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Recategoriza los valores de las columnas que contienen códigos numéricos ISO 3166-1 o valores de nombre de país
        a sus correspondientes códigos ISO 3166-1 alfa-2. Los valores nulos o no reconocidos se asignan a 'SIN INFO'.

        Parameters
        ----------
        df1 : pd.DataFrame
            Primer DataFrame que contiene las columnas con códigos numéricos ISO 3166-1 o valores de nombre de país.
        df2 : pd.DataFrame
            Segundo DataFrame que contiene las columnas con códigos numéricos ISO 3166-1 o valores de nombre de país.
        params: Dict[str, Any]
            Diccionario de parámetros primary.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tupla de DataFrames con las recategorizaciones realizadas.
        """
        # Contadores para los loggers
        null_count = 0
        country_mapping = {}

        # Función para mapear los códigos de país
        def map_country_code(x):
            nonlocal null_count
            if pd.isna(x) or x in ['None', 'NaN', '']:
                null_count += 1
                return 'SIN INFO'
            elif x.isdigit():
                country = pycountry.countries.get(numeric=x.zfill(3))
            else:
                country = pycountry.countries.get(alpha_2=x) or pycountry.countries.get(name=x)
            
            if country:
                country_code = country.alpha_2
                country_mapping[x] = country_code
                return country_code
            else:
                null_count += 1
                return 'SIN INFO'

        # Procesar el primer DataFrame
        for col in params['df1']['countrys_cols']:
            if col in df1.columns:
                df1[col] = df1[col].apply(map_country_code)

        # Procesar el segundo DataFrame
        for col in params['df2']['countrys_cols']:
            if col in df2.columns:
                df2[col] = df2[col].apply(map_country_code)

        # Log final de información
        logger.info(f"Total de valores asignados a 'SIN INFO': {null_count}")
        logger.info("Mapeo de valores originales a nombres de países (códigos ISO 3166-1 alfa-2):")
        for original_value, mapped_value in country_mapping.items():
            logger.info(f"{original_value} -> {mapped_value}")

        return df1, df2


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