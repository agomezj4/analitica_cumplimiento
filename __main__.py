import sys
from src.anomalias.utils import Utils
from src.anomalias.orchestration import PipelineOrchestration

Utils.add_src_to_path()


def main():
    if len(sys.argv) > 1:
        pipeline = sys.argv[1]
        if pipeline == 'All Pipelines':
            PipelineOrchestration.run_pipeline_raw()
            PipelineOrchestration.run_pipeline_intermediate()

        elif pipeline == 'Pipeline Raw':
            PipelineOrchestration.run_pipeline_raw()

        elif pipeline == 'Pipeline Intermediate':
            PipelineOrchestration.run_pipeline_intermediate()

        else:
            print(f"Pipeline '{pipeline}' no reconocido.")
    else:
        print("No se especificó un pipeline. Uso: python __main__.py [pipeline]")


if __name__ == "__main__":
    main()

