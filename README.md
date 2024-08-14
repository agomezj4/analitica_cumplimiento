# analitica_cumplimiento
Análisis de series temporales e identificación de anomalías en transacciones bancarias de clientes con diferentes productos.


# Metadata del Proyecto

## Fuentes de Datos Originales

El proyecto utiliza tres fuentes de datos principales:

1. **CLIENTES**: Base de datos de clientes
2. **PRODUCTOS**: Base de datos de productos de los clientes
3. **TRANSACCIONES**: Contiene las transacciones de los clientes, identificando el producto, la fecha, el monto y el país de origen o destino de la transacción. La codificación de los países está en ISO 3166-1, en algunos casos utilizando el código alfanumérico y en otros el numérico.

## Datasets Procesados

### 1. data_trx_feature_pd

Dataset de transacciones realizadas por los clientes con sus productos.

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

Dataset de clientes y sus productos.

| # | Columna | Tipo de Dato |
|---|---------|--------------|
| 0 | CODIGO | object |
| 1 | TIPO_CLIENTE | object |
| 2 | FECHA_ACTUALIZACION | datetime64[ns] |
| 3 | PEP | int64 |
| 4 | RIESGO | int64 |
| 5 | PAIS | object |
| 6 | CUENTA | object |
| 7 | TIPO_CUENTA | object |
| 8 | ESTADO_CUENTA | object |
| 9 | MONTO_TRX_RECIBIDA | float64 |
| 10 | FRECUENCIA_TRX_RECIBIDA | int64 |
| 11 | MONTO_TRX_ENVIADA | float64 |
| 12 | FRECUENCIA_TRX_ENVIADA | int64 |
| 13 | TIEMPO_ESTADO_CUENTA | int64 |
| 14 | RATIO_TRX_ENVIADAS_RECIBIDAS | float64 |
| 15 | MONTO_PROM_TRX_RECIBIDA | float64 |
| 16 | MONTO_PROM_TRX_ENVIADA | float64 |
| 17 | CANT_PROD | int64 |
