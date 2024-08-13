"""
Lógica del pipeline raw
"""

from typing import Any, Dict

import pandas as pd
import re

from src.analitica_cumplimiento.utils import Utils
logger = Utils.setup_logging()


class PipelineRaw:

    # 1. Cargar datos de clientes
    @staticmethod
    def load_data_customers_pd(filepath: str) -> pd.DataFrame:
        """
        Carga y procesa un archivo de datos de clientes desde un filepath local o de S3.

        Parameters
        ----------
        filepath : str
            Ruta al archivo de datos de clientes. Puede ser una ruta local o un enlace a S3.

        Returns
        -------
        pd.DataFrame
            DataFrame de pandas que contiene los datos procesados, con columnas extraídas del encabezado.
        """
        lines = Utils.load_lines_from_filepath(filepath)
        header = lines[0].strip().split()
        logger.info("Inicio de carga de datos de clientes...")
        lines = lines[1:]  # Omitir la primera línea

        # Expresión regular ajustada para capturar los patrones específicos
        pattern = re.compile(
            r'(\S+)\s+'  # CODIGO: cualquier texto sin espacios
            r'(NATURAL|JURIDICO)\s+'  # TIPO_CLIENTE: solo 'NATURAL' o 'JURIDICO'
            r'(\d{8}\.000|\d{8}|0\.000)\s*'  # FECHA_ACTUALIZACION: formato numérico, con .000, o 0.000
            r'(NO|SI|MEDIO|SIN\sINFO)?\s*' # PEP: 'NO', 'SI', o 'SIN INFO'
            r'(MEDIO BAJO|MEDIO ALTO|ALTO|MEDIO|BAJO|SIN\sINFO)?\s*'   # RIESGO: 'ALTO', 'MEDIO', 'BAJO', o 'SIN INFO'
            r'([A-Z]{2}|SIN\sINFO)?'  # PAIS: exactamente 2 letras mayúsculas o 'SIN INFO'
        )

        # Valores por defecto si los campos están ausentes
        default_values = {
            'PEP': 'SIN INFO',
            'RIESGO': 'SIN INFO',
            'PAIS': 'SIN INFO',
            'FECHA_ACTUALIZACION': '0'  # Asignar '0' si el campo está ausente
        }

        logger.info("Carga de datos de clientes completada!")
        return Utils.process_lines_to_df(lines, pattern, header, default_values)


    # 2. Cargar datos de transacciones
    @staticmethod
    def load_data_trx_pd(filepath: str) -> pd.DataFrame:
        """
        Carga y procesa un archivo de datos de transacciones desde un filepath local o de S3.

        Parameters
        ----------
        filepath : str
            Ruta al archivo de datos de transacciones. Puede ser una ruta local o un enlace a S3.

        Returns
        -------
        pd.DataFrame
            DataFrame de pandas que contiene los datos procesados, con columnas extraídas del encabezado.
        """
        lines = Utils.load_lines_from_filepath(filepath)
        header = lines[0].strip().split()
        logger.info("Inicio de carga de datos de transacciones...")
        lines = lines[1:]  # Omitir la primera línea

        # Expresión regular ajustada para capturar los patrones específicos
        pattern = re.compile(
            r"('S\d+')?\s*"  # CUENTA: comienza con 'S' seguido de un número
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})?\s*"  # FECHA_TRANSACCION: formato de fecha y hora
            r"('Wires Out')?\s*"  # TIPO_TRANSACCION: exactamente 'Wires Out'
            r"(\d+\.\d{3})?\s*"  # MONTO: número con tres decimales
            r"('[A-Z0-9]{2,3}'|''|\d{3})?\s*"  # PAIS_ORIGEN_TRANSACCION: código ISO alfanumérico o numérico
            r"('[A-Z0-9]{2,3}'|''|\d{3})?"  # PAIS_DESTINO_TRANSACCION: código ISO alfanumérico o numérico
        )

        # Valores por defecto si los campos están ausentes
        default_values = {
            'PAIS_ORIGEN_TRANSACCION': 'SIN INFO',
            'PAIS_DESTINO_TRANSACCION': 'SIN INFO'
        }

        logger.info("Carga de datos de transacciones completada!")

        return Utils.process_lines_to_df(lines, pattern, header, default_values)
    
    # 3. Join de datos: primero se une clientes con productos y luego se unen con transacciones
    @staticmethod
    def merge_dataframes_pd(
        df_clientes: pd.DataFrame, 
        df_productos: pd.DataFrame, 
        df_trx: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Realiza la unión de tres DataFrames utilizando claves especificadas en el diccionario de parámetros.
        
        Parameters
        ----------
        df_clientes : pandas.DataFrame
            DataFrame que contiene la información de los clientes.
        df_productos : pandas.DataFrame
            DataFrame que contiene la información de los productos.
        df_trx : pandas.DataFrame
            DataFrame que contiene la información de las transacciones.
        params : dict
            Diccionario de parámetros raw. 

        Returns
        -------
        pd.DataFrame
            DataFrame resultante después de realizar los joins.
        """
        # Registra el inicio del proceso de join
        logger.info("Iniciando el proceso de unión de DataFrames...")

        # Parámetros
        codigo_join_clientes_prod = params['codigo_join_clientes_prod']
        codigo_join_prod_trx = params['codigo_join_prod_trx']

        # Realiza el primer join entre df_clientes y df_productos
        df_merged = pd.merge(
            df_clientes, 
            df_productos, 
            how='left', 
            left_on=codigo_join_clientes_prod, 
            right_on=codigo_join_clientes_prod
        )
        logger.info("Join entre df_clientes y df_productos completado.")

        # Limpia los valores de la columna 'cuenta' en df_trx
        df_trx[codigo_join_prod_trx] = df_trx[codigo_join_prod_trx].str.replace("'", "")
        logger.info("Limpieza de valores en la columna 'cuenta' de df_trx completada.")

        # Realiza el segundo join entre df_merged y df_trx
        df_final = pd.merge(
            df_merged, 
            df_trx, 
            how='left', 
            left_on=codigo_join_prod_trx, 
            right_on=codigo_join_prod_trx
        )
        logger.info("Join entre df_merged y df_trx completado.")

        # Registra el número de registros en el DataFrame final
        final_records = len(df_final)
        logger.info(f"El proceso de unión ha generado un DataFrame con {final_records} registros.\n")

        return df_final