import os
from .utils import Utils

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')

parameters = Utils.load_parameters(parameters_directory)


class PipelineOrchestration:
    
    # 1. Pipeline Raw
    @staticmethod
    def run_pipeline_raw():
        logger.info('Inicio Pipeline Raw\n')

        from .pipelines.raw import PipelineRaw

        # 1.1 Carga de datos input
        data_clientes_pd = PipelineRaw.load_data_customers_pd(parameters['parameters_catalog']['data_clientes_input_path'])
        data_productos_pd = Utils.load_csv_pd(parameters['parameters_catalog']['data_producto_input_path'])
        data_trx_pd = PipelineRaw.load_data_trx_pd(parameters['parameters_catalog']['data_trx_input_path'])

        # 1.2 Join de datos input
        data_customers_raw_pd, data_trx_raw_pd = PipelineRaw.merge_dataframes_pd(data_clientes_pd, data_productos_pd, data_trx_pd, parameters['parameters_raw'])

        # 1.3 Guardar datos raw en formato parquet en s3
        Utils.save_parquet_to_s3(data_customers_raw_pd, parameters['parameters_catalog']['data_customers_raw_path'])
        Utils.save_parquet_to_s3(data_trx_raw_pd, parameters['parameters_catalog']['data_trx_raw_path'])
        
        logger.info('Fin Pipeline Raw')


    # 2. Pipeline Intermediate
    @staticmethod
    def run_pipeline_intermediate():
        logger.info('Inicio Pipeline Intermediate\n')

        from .pipelines.intermediate import PipelineIntermediate

        # 2.1 Carga de datos raw
        data_customers_raw_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_customers_raw_path'])
        data_trx_raw_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_trx_raw_path'])

        # 2.2 Cambio de tipos de datos
        data_customers_change_type,  data_trx_change_type = PipelineIntermediate.change_data_type_pd(data_customers_raw_pd, data_trx_raw_pd, parameters['parameters_intermediate'])

        # 2.3 Cambio de nombres de columnas
        data_customers_intermediate_pd = PipelineIntermediate.change_data_name_pd(data_customers_change_type, parameters['parameters_intermediate'])

        #2.4 Estandarización de columnas string
        data_trx_intermediate_pd = PipelineIntermediate.standarize_data_str_pd(data_trx_change_type, parameters['parameters_intermediate'])

        # 2.5 Guardar datos intermediate en formato parquet en s3
        Utils.save_parquet_to_s3(data_customers_intermediate_pd, parameters['parameters_catalog']['data_customers_intermediate_path'])
        Utils.save_parquet_to_s3(data_trx_intermediate_pd, parameters['parameters_catalog']['data_trx_intermediate_path'])

        logger.info('Fin Pipeline Intermediate')

    
    # 3. Pipeline Primary
    @staticmethod
    def run_pipeline_primary():
        logger.info('Inicio Pipeline Primary\n')

        from .pipelines.primary import PipelinePrimary

        # 3.1 Carga de datos intermediate
        data_customers_intermediate_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_customers_intermediate_path'])
        data_trx_intermediate_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_trx_intermediate_path'])

        # 3.2 Recategorización de valores de campos
        data_customers_primary_pd = PipelinePrimary.recategorize_data_pd(data_customers_intermediate_pd, parameters['parameters_primary'])

        # 3.3 Recategorización de países
        data_customers_country, data_trx_primary_pd = PipelinePrimary.recategorize_countrys_pd(data_customers_primary_pd, data_trx_intermediate_pd, parameters['parameters_primary'])

        # 3.4 Impuntación de valores faltantes
        data_customers_primary_pd = PipelinePrimary.impute_missing_values_pd(data_customers_country, parameters['parameters_primary'])

        # 3.5 Guardar datos primary en formato parquet en s3
        Utils.save_parquet_to_s3(data_customers_primary_pd, parameters['parameters_catalog']['data_customers_primary_path'])
        Utils.save_parquet_to_s3(data_trx_primary_pd, parameters['parameters_catalog']['data_trx_primary_path'])

        logger.info('Fin Pipeline Primary')

    
    # 4. Pipeline Feature Engineering
    @staticmethod
    def run_pipeline_feature_engineering():
        logger.info('Inicio Pipeline Feature Engineering\n')

        from .pipelines.feature import PipelineFeature

        # 4.1 Carga de datos primary
        data_customers_primary_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_customers_primary_path'])
        data_trx_primary_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_trx_primary_path'])

        #4.2 Creación de nuevas características
        data_customers_feature_pd, data_trx_feature_pd = PipelineFeature.create_features_pd(data_customers_primary_pd, data_trx_primary_pd, parameters['parameters_feature'])

        # 4.3 Guardar datos feature en formato parquet en s3
        Utils.save_parquet_to_s3(data_customers_feature_pd, parameters['parameters_catalog']['data_customers_feature_path'])
        Utils.save_parquet_to_s3(data_trx_feature_pd, parameters['parameters_catalog']['data_trx_feature_path'])

        logger.info('Fin Pipeline Feature Engineering')


    # 5. Pipeline Detección de Anomalías
    @staticmethod
    def run_pipeline_anomaly_detection():
        logger.info('Inicio Pipeline Detección de Anomalías\n')

        from .pipelines.anomalias import PipelineAnomalias

        # 5.1 Carga de datos feature
        data_trx_feature_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_trx_feature_path'])

        # 5.2 Preparar datos para detección de anomalías
        data_filter_anomalia_pd = PipelineAnomalias.df_filter_pd(data_trx_feature_pd, parameters['parameters_anomalias'])
        data_anomalia_scaled_pd = PipelineAnomalias.min_max_scaler_pd(data_filter_anomalia_pd[1])

        # 5.3 Calcular el score Z 
        data_score_z_pd = PipelineAnomalias.calculate_z_scores_pd(data_filter_anomalia_pd[0], data_anomalia_scaled_pd, parameters['parameters_anomalias'])

        # 5.4 Detectar anomalías
        df_con_anomalias_pd = PipelineAnomalias.detectar_anomalias_pd(data_score_z_pd, parameters['parameters_anomalias'])

        # 5.5 Resumen de anomalías
        transacciones_marcadas_pd, resumen_anomalias = PipelineAnomalias.procesar_anomalias_pd(df_con_anomalias_pd, parameters['parameters_anomalias'])

        # 5.6 Guardar datos con anomalías en formato parquet en s3
        Utils.save_parquet_to_s3(transacciones_marcadas_pd, parameters['parameters_catalog']['data_info_anomaly_trx_path'])
        Utils.save_parquet_to_s3(resumen_anomalias, parameters['parameters_catalog']['sumary_anomaly_trx_path'])

        logger.info('Fin Pipeline Detección de Anomalías')


    # 6. Pipeline de Series Temporales
    @staticmethod
    def run_pipeline_time_series():
        logger.info('Inicio Pipeline de Series Temporales\n')

        from .pipelines.series_tiempo import PipelineSeriesTiempo

        # 6.1 Carga de datos feature
        data_customers_feature_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_customers_feature_path'])

        # 6.2 Tratar outliers
        treat_outliers_pd = PipelineSeriesTiempo.treat_outliers_pd(data_customers_feature_pd)

        # 6.3 Análisis series temporales
        fig_decomposition, fig_seasonal, fig_forecast = PipelineSeriesTiempo.generate_analysis(treat_outliers_pd)

        # 6.4 Guardar figuras en formato png en s3
        Utils.save_image_to_s3(fig_decomposition, parameters['parameters_catalog']['plot_decomposition_path'])
        Utils.save_image_to_s3(fig_seasonal, parameters['parameters_catalog']['plot_seasonal_component_path'])
        Utils.save_image_to_s3(fig_forecast, parameters['parameters_catalog']['plot_forecast_path'])

        logger.info('Fin Pipeline de Series Temporales')
