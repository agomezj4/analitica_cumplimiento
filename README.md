# analitica_cumplimiento
Análisis de series temporales e identificación de anomalías en transacciones bancarias de clientes con diferentes productos.

# Metadata del Proyecto

## Fuentes de Datos Originales

El proyecto utiliza tres fuentes de datos principales:

1. **CLIENTES**: Base de datos de clientes
2. **PRODUCTOS**: Base de datos de productos de los clientes
3. **TRANSACCIONES**: Contiene las transacciones de los clientes, identificando el producto, la fecha, el monto y el país de origen o destino de la transacción. La codificación de los países está en ISO 3166-1, en algunos casos utilizando el código alfanumérico y en otros el numérico.

## Procesamiento de Datos

Se realizaron las siguientes operaciones de procesamiento:

1. Unión de las bases de datos de CLIENTES y PRODUCTOS.
2. Procesamiento de la base de datos de TRANSACCIONES.
3. Creación de nuevas variables derivadas.

## Datasets Resultantes

### 1. data_trx_feature_pd

Dataset de transacciones con características adicionales.

| # | Columna | Tipo de Dato |
|---|---------|--------------|
| 0 | CUENTA | object |
| 1 | FECHA_TRANSACCION | datetime64[ns] |
| 2 | TIPO_TRANSACCION | object |
| 3 | MONTO | float64 |
| 4 | PAIS_ORIGEN_TRANSACCION | object |
| 5 | PAIS_DESTINO_TRANSACCION | object |
| 6 | DIAS_ENTRE_TRX | int64 |
| 7 | VARIACION_MONTO_MES_ANIO | float64 |
| 8 | PAIS_ORIGEN_DESTINO_TRX | object |
| 9 | MES_ANIO | period[M] |
| 10 | ACUM_MONTO_MES_ANIO | float64 |

### 2. data_customers_feature_pd

Dataset de clientes con características adicionales.

| # | Columna | Tipo de Dato | No Nulos |
|---|---------|--------------|----------|
| 0 | CODIGO | object | 49321 |
| 1 | TIPO_CLIENTE | object | 49321 |
| 2 | FECHA_ACTUALIZACION | datetime64[ns] | 49321 |
| 3 | PEP | int64 | 49321 |
| 4 | RIESGO | int64 | 49321 |
| 5 | PAIS | object | 49321 |
| 6 | CUENTA | object | 49321 |
| 7 | TIPO_CUENTA | object | 49321 |
| 8 | ESTADO_CUENTA | object | 49321 |
| 9 | MONTO_TRX_RECIBIDA | float64 | 49321 |
| 10 | FRECUENCIA_TRX_RECIBIDA | int64 | 49321 |
| 11 | MONTO_TRX_ENVIADA | float64 | 49321 |
| 12 | FRECUENCIA_TRX_ENVIADA | int64 | 49321 |
| 13 | TIEMPO_ESTADO_CUENTA | int64 | 49321 |
| 14 | RATIO_TRX_ENVIADAS_RECIBIDAS | float64 | 49321 |
| 15 | MONTO_PROM_TRX_RECIBIDA | float64 | 49321 |
| 16 | MONTO_PROM_TRX_ENVIADA | float64 | 49321 |
| 17 | CANT_PROD | int64 | 49321 |

Tipos de datos: datetime64[ns](1), float64(5), int64(6), object(6)
