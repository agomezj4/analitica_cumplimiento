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
        data_clientes_pd = PipelineRaw.load_data_customers_pd(parameters['parameters_catalog']['data_clientes_raw_path'])
        data_productos_pd = Utils.load_csv_pd(parameters['parameters_catalog']['data_producto_raw_path'])
        data_trx_pd = PipelineRaw.load_data_trx_pd(parameters['parameters_catalog']['data_trx_raw_path'])

        # 1.2 Join de datos input
        data_raw_pd = PipelineRaw.merge_dataframes_pd(data_clientes_pd, data_productos_pd, data_trx_pd, parameters['parameters_raw'])

        # 1.3 Guardar datos raw en formato parquet en s3
        Utils.save_parquet_to_s3(data_raw_pd, parameters['parameters_catalog']['data_raw_path'])
        
        logger.info('Fin Pipeline Raw')
