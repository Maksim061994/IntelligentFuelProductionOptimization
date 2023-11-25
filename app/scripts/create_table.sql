create table RL_MODELS.series_upload_ovens
(
    id_serie        Int32,
    dt              DateTime,
    date            DATE,
    operation_code  Int32,
    operation_serie String,
    temp_serie      Int32,
    id_oven         Int32
)
    engine = MergeTree ORDER BY (id_serie, dt, operation_serie, id_oven)
        SETTINGS index_granularity = 8192;