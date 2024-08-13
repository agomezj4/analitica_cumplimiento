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
        data_raw_pd = PipelineRaw.merge_dataframes_pd(data_clientes_pd, data_productos_pd, data_trx_pd, parameters['parameters_raw'])

        # 1.3 Guardar datos raw en formato parquet en s3
        Utils.save_parquet_to_s3(data_raw_pd, parameters['parameters_catalog']['data_raw_path'])
        
        logger.info('Fin Pipeline Raw')


    # 2. Pipeline Intermediate
    @staticmethod
    def run_pipeline_intermediate():
        logger.info('Inicio Pipeline Intermediate\n')

        from .pipelines.intermediate import PipelineIntermediate

        # 2.1 Carga de datos raw
        data_raw_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_raw_path'])

        # 2.2 Cambio de tipos de datos
        data_change_type = PipelineIntermediate.change_data_type_pd(data_raw_pd, parameters['parameters_intermediate'])

        # 2.3 Cambio de nombres de columnas
        data_change_name = PipelineIntermediate.change_data_name_pd(data_change_type, parameters['parameters_intermediate'])

        #2.4 Estandarización de columnas string
        data_intermediate_pd = PipelineIntermediate.standarize_data_str_pd(data_change_name, parameters['parameters_intermediate'])

        # 2.5 Guardar datos intermediate en formato parquet en s3
        Utils.save_parquet_to_s3(data_intermediate_pd, parameters['parameters_catalog']['data_intermediate_path'])

        logger.info('Fin Pipeline Intermediate')

    
    # 3. Pipeline Primary
    @staticmethod
    def run_pipeline_primary():
        logger.info('Inicio Pipeline Primary\n')

        from .pipelines.primary import PipelinePrimary

        # 3.1 Carga de datos intermediate
        data_intermediate_pd = Utils.load_parquet_from_s3(parameters['parameters_catalog']['data_intermediate_path'])

        # 3.2 Recategorización de datos
        data_recategorize = PipelinePrimary.recategorize_data_pd(data_intermediate_pd, parameters['parameters_primary'])

        # 3.3 Recategorización de países
        data_country_rec = PipelinePrimary.recategorize_countrys_pd(data_recategorize, parameters['parameters_primary'])

        # 3.4 Imputación de valores faltantes
        data_primary_pd = PipelinePrimary.impute_missing_values_pd(data_country_rec, parameters['parameters_primary'])

        # 3.3 Guardar datos primary en formato parquet en s3
        Utils.save_parquet_to_s3(data_primary_pd, parameters['parameters_catalog']['data_primary_path'])

        logger.info('Fin Pipeline Primary')
