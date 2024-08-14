"""
Lógica del pipeline series de tiempo
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

from src.analitica_cumplimiento.utils import Utils
from .anomalias import PipelineAnomalias

logger = Utils.setup_logging()


class PipelineSeriesTiempo:
    
    # 1. Tratamiento de outliers
    @staticmethod
    def treat_outliers_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata los outliers de un DataFrame en múltiples columnas numéricas (float64 e int64)
        utilizando el rango intercuartílico (IQR). Reemplaza los outliers por el valor
        del percentil más bajo o más alto, dependiendo del caso.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada cuyas columnas serán tratadas para manejar los outliers.

        Returns
        -------
        pd.DataFrame: DataFrame con los outliers tratados en las columnas especificadas.
        """
        logger.info("Iniciando el proceso de tratamiento de outliers...")

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        logger.info(f"Columnas numéricas identificadas para el tratamiento de outliers: {numeric_columns.tolist()}")

        df_clean = df.copy()

        for column_name in numeric_columns:
            q1 = df_clean[column_name].quantile(0.25)
            q3 = df_clean[column_name].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if pd.api.types.is_integer_dtype(df_clean[column_name]):
                lower_bound = int(lower_bound)
                upper_bound = int(upper_bound)

            outliers_mask = (df_clean[column_name] < lower_bound) | (df_clean[column_name] > upper_bound)
            has_outliers = outliers_mask.sum() > 0

            if has_outliers:
                logger.info(f"Se encontraron outliers en la columna '{column_name}'. Tratando outliers...")
                df_clean.loc[df_clean[column_name] < lower_bound, column_name] = lower_bound
                df_clean.loc[df_clean[column_name] > upper_bound, column_name] = upper_bound
                logger.info(f"Outliers en la columna '{column_name}' han sido reemplazados por los valores límite.")
            else:
                logger.info(f"No se encontraron outliers en la columna '{column_name}'.")

        logger.info("Tratamiento de outliers completado.")
        return df_clean


    # 2. Extracción de componentes de fecha
    @staticmethod
    def extract_date_components(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae los componentes de mes y año de la columna de fecha.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con la columna 'FECHA_ACTUALIZACION'.

        Returns
        -------
        pd.DataFrame: DataFrame con las columnas 'MES' y 'AÑO' añadidas.
        """
        logger.info("Extrayendo componentes de fecha (mes y año)...")
        df['MES'] = df['FECHA_ACTUALIZACION'].dt.month
        df['AÑO'] = df['FECHA_ACTUALIZACION'].dt.year
        logger.info("Componentes de fecha extraídos.")
        return df


    # 3. Filtrado de datos
    @staticmethod
    def filter_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra los datos para excluir registros con años posteriores a 2024.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con la columna 'AÑO'.

        Returns
        -------
        pd.DataFrame: DataFrame filtrado hasta el año 2024.
        """
        logger.info("Filtrando datos hasta el año 2024...")
        df_filtered = df[df['AÑO'] <= 2024].copy()
        logger.info("Filtrado completado.")
        return df_filtered


    # 4. Agrupación de datos
    @staticmethod
    def group_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrupa los datos por tipo de cuenta, año y mes, y calcula la suma de los montos enviados y recibidos.
        Luego, calcula la diferencia entre estos montos y elimina filas donde alguna de las columnas es 0.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas 'TIPO_CUENTA', 'AÑO', 'MES', 'MONTO_TRX_ENVIADA', 'MONTO_TRX_RECIBIDA'.

        Returns
        -------
        pd.DataFrame: DataFrame agrupado y filtrado.
        """
        logger.info("Agrupando datos por tipo de cuenta, año y mes...")
        df_grouped = df.groupby(['TIPO_CUENTA', 'AÑO', 'MES']).agg({
            'MONTO_TRX_ENVIADA': 'sum',
            'MONTO_TRX_RECIBIDA': 'sum'
        }).reset_index()

        df_grouped['DIFERENCIA_TRX'] = df_grouped['MONTO_TRX_ENVIADA'] - df_grouped['MONTO_TRX_RECIBIDA']
        df_filtered = df_grouped.query('MONTO_TRX_ENVIADA != 0 and MONTO_TRX_RECIBIDA != 0 and DIFERENCIA_TRX != 0')

        logger.info("Agrupación y filtrado completados.")
        return df_filtered


    # 5. Escala de columnas
    @staticmethod
    def scale_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Escala las columnas seleccionadas utilizando MinMaxScaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas 'MONTO_TRX_ENVIADA', 'MONTO_TRX_RECIBIDA', 'DIFERENCIA_TRX'.

        Returns
        -------
        pd.DataFrame: DataFrame con las columnas escaladas.
        """
        logger.info("Iniciando la escala de columnas seleccionadas...")
        columnas_a_escalar = ['MONTO_TRX_ENVIADA', 'MONTO_TRX_RECIBIDA', 'DIFERENCIA_TRX']
        df_to_scale = df[columnas_a_escalar].copy()
        df_scaled = PipelineAnomalias.min_max_scaler_pd(df_to_scale)
        df[columnas_a_escalar] = df_scaled
        logger.info("Escalado de columnas completado.")
        return df


    # 6. Filtrado por tipo de cuenta
    @staticmethod
    def filter_by_account_type(df: pd.DataFrame, account_type: str) -> pd.DataFrame:
        """
        Filtra el DataFrame por el tipo de cuenta especificado, renombra columnas, 
        y asegura que los datos estén ordenados por año y mes con frecuencia mensual.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas 'TIPO_CUENTA', 'AÑO', 'MES'.
        account_type : str
            Tipo de cuenta por la cual se desea filtrar.

        Returns
        -------
        pd.DataFrame: DataFrame filtrado y preparado para análisis de series de tiempo.
        """
        logger.info(f"Filtrando datos por tipo de cuenta: {account_type}...")
        df_cuenta = df[df['TIPO_CUENTA'] == account_type]
        df_cuenta = df_cuenta.rename(columns={'AÑO': 'year', 'MES': 'month'})
        df_cuenta = df_cuenta.set_index(pd.to_datetime(df_cuenta[['year', 'month']].assign(DAY=1)))
        df_cuenta = df_cuenta.sort_index()
        df_cuenta = df_cuenta.asfreq('MS')
        df_cuenta['DIFERENCIA_TRX'] = df_cuenta['DIFERENCIA_TRX'].interpolate(method='linear')
        logger.info("Filtrado y preparación de datos completados.")
        return df_cuenta


    # 7. Plotting de descomposición
    @staticmethod
    def plot_decomposition(result: Any) -> plt.Figure:
        """
        Realiza y visualiza la descomposición de la serie temporal.

        Parameters
        ----------
        result : Any
            Resultado de la descomposición de la serie temporal.

        Returns
        -------
        plt.Figure: Figura con la descomposición de la serie temporal.
        """
        logger.info("Iniciando la descomposición de la serie temporal...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convertir los índices y datos a arrays unidimensionales
        index_array = np.array(result.observed.index)
        observed_array = np.array(result.observed).flatten()
        trend_array = np.array(result.trend).flatten()
        seasonal_array = np.array(result.seasonal).flatten()
        resid_array = np.array(result.resid).flatten()

        components = [
            ('Observed', observed_array, ax1, '#2C3E50'),
            ('Trend', trend_array, ax2, '#27AE60'),
            ('Seasonal', seasonal_array, ax3, '#E67E22'),
            ('Residual', resid_array, ax4, '#95A5A6')
        ]

        for title, data, ax, color in components:
            ax.plot(index_array, data, color=color)
            ax.set_title(title, fontsize=14, fontweight='bold')
            PipelineSeriesTiempo.style_axis(ax)

        fig.suptitle('Descomposición de la Diferencia de Transacciones Mensuales para "CUENTA"', fontsize=16, y=1.02)
        plt.tight_layout()
        logger.info("Descomposición de la serie temporal completada.")
        return fig


    # 8. Plotting de componente estacional
    @staticmethod
    def plot_seasonal_component(result: Any) -> plt.Figure:
        """
        Visualiza el componente estacional de la serie temporal.

        Parameters
        ----------
        result : Any
            Resultado de la descomposición de la serie temporal.

        Returns
        -------
        plt.Figure: Figura con el componente estacional de la serie temporal.
        """
        logger.info("Visualizando el componente estacional de la serie temporal...")
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Convertir el índice a un array de NumPy para evitar problemas de indexación multidimensional
        index_array = np.array(result.seasonal.index)
        
        # Asegurarse de que los datos sean un array unidimensional
        seasonal_array = np.array(result.seasonal).flatten()
        
        # Usar los arrays en lugar de las series directamente
        ax.plot(index_array, seasonal_array, color='#E67E22')
        ax.fill_between(index_array, seasonal_array, alpha=0.3, color='#E67E22')
        ax.axhline(y=0, color='#7F8C8D', linestyle='--')
        ax.set_title('Componente Estacional de la Diferencia de Transacciones para "CUENTA"', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Diferencia de Transacciones (Estacionalidad)', fontsize=12)
        PipelineSeriesTiempo.style_axis(ax)

        peak = np.argmax(seasonal_array)
        trough = np.argmin(seasonal_array)
        
        # Convertir numpy.datetime64 a datetime
        peak_date = pd.to_datetime(index_array[peak]).strftime("%b")
        trough_date = pd.to_datetime(index_array[trough]).strftime("%b")

        ax.annotate(f'Pico: {peak_date}', xy=(index_array[peak], seasonal_array[peak]), xytext=(10, 10), 
                    textcoords='offset points', ha='left', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        ax.annotate(f'Valle: {trough_date}', xy=(index_array[trough], seasonal_array[trough]), xytext=(10, -10), 
                    textcoords='offset points', ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        logger.info("Visualización del componente estacional completada.")
        return plt.gcf()


    # 9. Plotting de pronóstico
    @staticmethod
    def plot_forecast(df_cuenta: pd.DataFrame) -> plt.Figure:
        """
        Realiza un pronóstico de la diferencia de transacciones utilizando el método Holt-Winters.

        Parameters
        ----------
        df_cuenta : pd.DataFrame
            DataFrame con la serie temporal 'DIFERENCIA_TRX' para realizar el pronóstico.

        Returns
        -------
        plt.Figure: Figura con el pronóstico y el intervalo de confianza.
        """
        logger.info("Iniciando el pronóstico utilizando el método Holt-Winters...")
        model = ExponentialSmoothing(df_cuenta['DIFERENCIA_TRX'], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        predictions = fit.forecast(steps=120)
        prediction_dates = pd.date_range(df_cuenta.index[-1] + pd.offsets.MonthEnd(1), periods=120, freq='MS')

        residuals = fit.resid
        std_resid = np.std(residuals)
        ci_lower = predictions - 1.96 * std_resid
        ci_upper = predictions + 1.96 * std_resid

        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        
        # Asegurarse de que los datos sean arrays unidimensionales
        index_array = np.array(df_cuenta.index)
        diferencias_trx_array = np.array(df_cuenta['DIFERENCIA_TRX']).flatten()
        predictions_array = np.array(predictions).flatten()
        ci_lower_array = np.array(ci_lower).flatten()
        ci_upper_array = np.array(ci_upper).flatten()

        ax.plot(index_array, diferencias_trx_array, label='Datos Históricos', color='#2980B9')
        ax.plot(prediction_dates, predictions_array, label='Pronóstico hasta 2034', linestyle='--', color='#2ECC71')
        ax.fill_between(prediction_dates, ci_lower_array, ci_upper_array, color='#2ECC71', alpha=0.2)
        ax.axvline(x=index_array[-1], color='red', linestyle=':', label='Inicio del Pronóstico')

        ax.set_title('Pronóstico de la Diferencia de Transacciones hasta 2034 para "CUENTA"', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Diferencia de Transacciones', fontsize=12)
        PipelineSeriesTiempo.style_axis(ax)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        logger.info("Pronóstico completado.")
        return plt.gcf()


    # 10. Generación de análisis
    @staticmethod
    def style_axis(ax: plt.Axes) -> None:
        """
        Aplica estilo consistente a los ejes de un gráfico.

        Parameters
        ----------
        ax : plt.Axes
            Ejes del gráfico a estilizar.

        Returns
        -------
        None
        """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))


    # 11. Generación de análisis
    @staticmethod
    def generate_analysis(df: pd.DataFrame) -> tuple:
        """
        Genera el análisis completo de la serie de tiempo, incluyendo la descomposición,
        la visualización del componente estacional y el pronóstico.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada con datos transaccionales.

        Returns
        -------
        tuple: Tupla que contiene las figuras de la descomposición, el componente estacional y el pronóstico.
        """
        logger.info("Iniciando el análisis de series de tiempo...")
        df = PipelineSeriesTiempo.extract_date_components(df)
        df = PipelineSeriesTiempo.filter_data(df)
        df_grouped = PipelineSeriesTiempo.group_data(df)
        df_grouped = PipelineSeriesTiempo.scale_columns(df_grouped)
        df_cuenta = PipelineSeriesTiempo.filter_by_account_type(df_grouped, 'CUENTA')

        result = seasonal_decompose(df_cuenta['DIFERENCIA_TRX'], model='additive', period=12)
        
        fig_decomposition = PipelineSeriesTiempo.plot_decomposition(result)
        fig_seasonal = PipelineSeriesTiempo.plot_seasonal_component(result)
        fig_forecast = PipelineSeriesTiempo.plot_forecast(df_cuenta)

        logger.info("Análisis de series de tiempo completado.")
        return (fig_decomposition, fig_seasonal, fig_forecast)
