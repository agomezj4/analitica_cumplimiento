from typing import Tuple, Dict

import os
import sys
import yaml
import pickle
import logging
import pandas as pd
import polars as pl
import boto3
import re
import pyarrow.parquet as pq
import io



class Utils:
    """
    Clase para funciones de utilidad comunes.
    """

    @staticmethod
    def setup_logging() -> logging.Logger:
        """
        Configura el logging para la aplicación.

        Returns
        -------
        logging.Logger
            El logger configurado para la aplicación.
        """
        import logging
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def add_src_to_path() -> None:
        """
        Agrega la ruta del directorio 'src' al sys.path para facilitar las importaciones.

        Returns
        -------
        None
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        sys.path.append(project_root)

    @staticmethod
    def get_project_root() -> str:
        """
        Obtiene la ruta raíz del proyecto.

        Returns
        -------
        str
            Ruta raíz del proyecto.
        """
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def load_parameters(parameters_directory: str) -> Dict[str, dict]:
        """
        Carga los archivos de parámetros en formato YAML desde un directorio específico.

        Parameters
        ----------
        parameters_directory : str
            Directorio donde se encuentran los archivos YAML.

        Returns
        -------
        Dict[str, dict]
            Diccionario con los parámetros cargados.
        """
        yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]
        parameters = {}
        for yaml_file in yaml_files:
            with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
                data = yaml.safe_load(file)
                key_name = f'parameters_{yaml_file.replace(".yml", "")}'
                parameters[key_name] = data
        return parameters

    @staticmethod
    def load_csv_pd(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Cargar datos desde un archivo CSV usando Pandas.

        Parameters
        ----------
        file_path : str
            Ruta del archivo CSV a cargar.
        encoding : str, optional
            Tipo de codificación del archivo CSV (por defecto 'utf-8').

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """
        return pd.read_csv(file_path, encoding=encoding)

    @staticmethod
    def load_csv_pl(file_path: str, encoding: str = 'utf-8') -> pl.DataFrame:
        """
        Cargar datos desde un archivo CSV usando Polars.

        Parameters
        ----------
        file_path : str
            Ruta del archivo CSV a cargar.
        encoding : str, optional
            Tipo de codificación del archivo CSV (por defecto 'utf-8').

        Returns
        -------
        pl.DataFrame
            DataFrame con los datos cargados.
        """
        return pl.read_csv(file_path, encoding=encoding)

    @staticmethod
    def load_parquet_pd(file_path: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo Parquet usando Pandas.

        Parameters
        ----------
        file_path : str
            Ruta del archivo Parquet a cargar.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """
        return pd.read_parquet(file_path)

    @staticmethod
    def load_parquet_pl(file_path: str) -> pl.DataFrame:
        """
        Cargar datos desde un archivo Parquet usando Polars.

        Parameters
        ----------
        file_path : str
            Ruta del archivo Parquet a cargar.

        Returns
        -------
        pl.DataFrame
            DataFrame con los datos cargados.
        """
        return pl.read_parquet(file_path)

    @staticmethod
    def save_csv_pd(data: pd.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo CSV usando Pandas.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo CSV donde se guardará el DataFrame.
        """
        data.to_csv(path, index=False)

    @staticmethod
    def save_csv_pl(data: pl.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo CSV usando Polars.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo CSV donde se guardará el DataFrame.
        """
        data.write_csv(path)

    @staticmethod
    def save_parquet_pd(data: pd.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo Parquet usando Pandas.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo Parquet donde se guardará el DataFrame.
        """
        data.to_parquet(path)

    @staticmethod
    def save_parquet_pl(data: pl.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo Parquet usando Polars.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo Parquet donde se guardará el DataFrame.
        """
        data.write_parquet(path)

    @staticmethod
    def load_pickle(file_path: str) -> object:
        """
        Carga un objeto desde un archivo pickle.

        Parameters
        ----------
        file_path : str
            Ruta del archivo pickle.

        Returns
        -------
        object
            Objeto cargado desde el archivo pickle.
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def save_pickle(data: object, file_path: str) -> None:
        """
        Guarda un objeto en un archivo pickle.

        Parameters
        ----------
        data : object
            Objeto a guardar.
        file_path : str
            Ruta del archivo pickle donde se guardará el objeto.

        Returns
        -------
        None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load_data_raw_from_s3(filepath: str) -> Tuple:
        """
        Cargar datos desde un archivo en S3.

        Parameters
        ----------
        filepath : str
            Ruta del archivo en S3.

        Returns
        -------
        Tuple
            Tupla con los datos cargados.
        """
        # Configurar cliente S3
        s3_client = boto3.client('s3')
        bucket_name = filepath.split('/')[2]
        key = '/'.join(filepath.split('/')[3:])
        
        # Descargar el archivo desde S3 a la memoria
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_content = response['Body'].read().decode('utf-8')
        
        # Leer el contenido como líneas
        lines = file_content.splitlines()
        
        return lines
    
    @staticmethod
    def load_lines_from_filepath(filepath: str) -> list:
        """
        Carga líneas de un archivo desde un filepath local o de S3.

        Parameters
        ----------
        filepath : str
            Ruta al archivo de datos. Puede ser una ruta local o un enlace a S3.

        Returns
        -------
        list
            Lista de líneas del archivo.
        """
        Utils.setup_logging().info("Iniciando la carga de datos desde '%s'", filepath)

        # Cargar datos desde S3 o el sistema de archivos local
        if filepath.startswith('s3://'):
            Utils.setup_logging().info("Cargando datos desde S3...")
            lines = Utils.load_data_raw_from_s3(filepath)
        
        # Cargar datos desde el sistema de archivos local
        else:
            Utils.setup_logging().info("Cargando datos desde el sistema de archivos local...")
            with open(filepath, encoding='utf-8') as file:
                lines = file.readlines()
        
        Utils.setup_logging().info("Archivo cargado correctamente, procesando líneas...")
        return lines

    @staticmethod
    def process_lines_to_df(lines: list, pattern: re.Pattern, header: list, default_values: dict) -> pd.DataFrame:
        """
        Procesa las líneas utilizando un patrón de expresión regular y devuelve un DataFrame.
        Si algunos valores están faltantes, se asignan valores por defecto.

        Parameters
        ----------
        lines : list
            Lista de líneas de texto.
        pattern : re.Pattern
            Expresión regular para capturar los datos.
        header : list
            Lista de nombres de las columnas.
        default_values : dict
            Diccionario con valores por defecto para campos faltantes.

        Returns
        -------
        pd.DataFrame
            DataFrame de pandas con los datos procesados.
        """
        data = []

        # Procesar cada línea
        for line in lines:
            match = pattern.match(line.strip())
            
            # Si hay un match, extraer los grupos, de lo contrario, asignar valores por defecto
            if match:
                row = list(match.groups())
            else:
                parts = line.strip().split()
                row = [parts[i] if i < len(parts) else default_values.get(col, None) for i, col in enumerate(header)]
            
            # Corrección específica para la columna FECHA_ACTUALIZACION
            if 'FECHA_ACTUALIZACION' in header:
                idx = header.index('FECHA_ACTUALIZACION')
                if row[idx] == '0.000':
                    row[idx] = ''
                elif row[idx] and row[idx] != '0.000':
                    row[idx] = f"{row[idx][:4]}-{row[idx][4:6]}-{row[idx][6:8]}"

            data.append(row)
        
        df = pd.DataFrame(data, columns=header)
        Utils.setup_logging().info("DataFrame creado con éxito con %d registros.\n", len(df))
        
        return df
    
    @staticmethod
    def load_parquet_from_s3(filepath: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo Parquet en S3 y devolverlo como un DataFrame de pandas.

        Parameters
        ----------
        filepath : str
            Ruta del archivo en S3, en formato 's3://bucket_name/path/to/file.parquet'.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """

        Utils.setup_logging().info("Cargando datos desde S3...")

        # Configurar cliente S3
        s3_client = boto3.client('s3')
        bucket_name = filepath.split('/')[2]
        key = '/'.join(filepath.split('/')[3:])
        
        # Descargar el archivo desde S3 a la memoria
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_content = response['Body'].read()
        
        # Cargar el archivo Parquet en un DataFrame
        buffer = io.BytesIO(file_content)
        df = pq.read_table(buffer).to_pandas()

        Utils.setup_logging().info("Datos cargados correctamente desde S3.\n")
        
        return df

    @staticmethod
    def save_parquet_to_s3(df: pd.DataFrame, filepath: str) -> None:
        """
        Guardar un DataFrame de pandas como un archivo Parquet en S3.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame que se desea guardar.
        filepath : str
            Ruta del archivo en S3, en formato 's3://bucket_name/path/to/file.parquet'.

        Returns
        -------
        None
        """

        Utils.setup_logging().info("Guardando datos en S3...")

        # Configurar cliente S3
        s3_client = boto3.client('s3')
        bucket_name = filepath.split('/')[2]
        key = '/'.join(filepath.split('/')[3:])
        
        # Guardar el DataFrame como Parquet en un buffer
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Subir el archivo Parquet a S3
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())

        Utils.setup_logging().info("Datos guardados correctamente en S3.\n")

