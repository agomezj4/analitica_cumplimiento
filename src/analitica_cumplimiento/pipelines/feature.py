"""
Lógica del pipeline feature
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.analitica_cumplimiento.utils import Utils
logger = Utils.setup_logging()


class PipelineFeature:

    # 1. Crear nuevas características
    @staticmethod
    def create_features_pd(
        df_customer_prod: pd.DataFrame, 
        df_prod_trx: pd.DataFrame, 
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Crea nuevas características en dos DataFrames, uno relacionado con cliente-producto y otro con producto-transacción.

        Parameters
        ----------
        df_customer_prod : pandas.DataFrame
            DataFrame que contiene información de cliente-producto.
        df_prod_trx : pandas.DataFrame
            DataFrame que contiene información de transacciones de producto.
        params: Dict[str, Any] 
            Diccionario de parámetros feature.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]: Una tupla de DataFrames con las nuevas características añadidas.
        """

        # Registra un mensaje de información indicando el inicio de la creación de características
        logger.info("Iniciando la creación de nuevas características...")

        # Parámetros
        cols_name = params['cols_name']

        # Creación de características para df_customer_prod
        df_customer_prod['TIEMPO_ESTADO_CUENTA'] = (
            pd.Timestamp.now() - df_customer_prod[cols_name[0]]
        ).dt.days

        df_customer_prod['RATIO_TRX_ENVIADAS_RECIBIDAS'] = (
            df_customer_prod[cols_name[1]] / df_customer_prod[cols_name[2]]
        ).fillna(0)

        df_customer_prod['MONTO_PROM_TRX_RECIBIDA'] = (
            df_customer_prod[cols_name[3]] / df_customer_prod[cols_name[2]]
        ).fillna(0)

        df_customer_prod['MONTO_PROM_TRX_ENVIADA'] = (
            df_customer_prod[cols_name[4]] / df_customer_prod[cols_name[1]]
        ).fillna(0)

        df_customer_prod['CANT_PROD'] = df_customer_prod.groupby(cols_name[5])[cols_name[6]].transform('nunique')


        # Creación de características para df_prod_trx
        df_prod_trx = df_prod_trx.sort_values(by=[cols_name[7], cols_name[8]])
        df_prod_trx['DIAS_ENTRE_TRX'] = df_prod_trx.groupby(cols_name[7])[cols_name[8]].diff().dt.days.fillna(0).astype(int)

        df_prod_trx['VARIACION_MONTO_MES_ANIO'] = df_prod_trx.groupby(cols_name[7])[cols_name[9]].pct_change().fillna(0)

        df_prod_trx['PAIS_ORIGEN_DESTINO_TRX'] = df_prod_trx[cols_name[10]] + '_' + df_prod_trx[cols_name[11]]

        df_prod_trx['MES_ANIO'] = df_prod_trx[cols_name[8]].dt.to_period('M')
        df_prod_trx['ACUM_MONTO_MES_ANIO'] = df_prod_trx.groupby([cols_name[7], 'MES_ANIO'])[cols_name[9]].cumsum()

        # Registra un mensaje de información indicando la finalización del proceso de creación de características
        logger.info("Finalizada la creación de nuevas características.")

        # Retorna la tupla de DataFrames con las nuevas características añadidas
        return df_customer_prod, df_prod_trx