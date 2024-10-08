import sys
from src.analitica_cumplimiento.utils import Utils
from src.analitica_cumplimiento.orchestration import PipelineOrchestration

Utils.add_src_to_path()


def main():
    if len(sys.argv) > 1:
        pipeline = sys.argv[1]
        if pipeline == 'All Pipelines':
            PipelineOrchestration.run_pipeline_raw()
            PipelineOrchestration.run_pipeline_intermediate()
            PipelineOrchestration.run_pipeline_primary()
            PipelineOrchestration.run_pipeline_feature_engineering()
            PipelineOrchestration.run_pipeline_anomaly_detection()
            PipelineOrchestration.run_pipeline_time_series()

        elif pipeline == 'Pipeline Raw':
            PipelineOrchestration.run_pipeline_raw()

        elif pipeline == 'Pipeline Intermediate':
            PipelineOrchestration.run_pipeline_intermediate()

        elif pipeline == 'Pipeline Primary':
            PipelineOrchestration.run_pipeline_primary()

        elif pipeline == 'Pipeline Feature Engineering':
            PipelineOrchestration.run_pipeline_feature_engineering()

        elif pipeline == 'Pipeline Anomaly Detection':
            PipelineOrchestration.run_pipeline_anomaly_detection()

        elif pipeline == 'Pipeline Time Series':
            PipelineOrchestration.run_pipeline_time_series()

        else:
            print(f"Pipeline '{pipeline}' no reconocido.")
    else:
        print("No se especificó un pipeline. Uso: python __main__.py [pipeline]")


if __name__ == "__main__":
    main()

