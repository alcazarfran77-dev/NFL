# Tarjeta del modelo — Proyecto NFL v3.1.0

Corrida de entrega: almacén MLflow limpio `mlflow/mlflow.db`, experimento `cgmt-combine-translation`,
`data_version = a75ba51ae4b9` (diez archivos originales, con tracking de partido de temporada completa de 2023
y 2024), semilla 42, {{N_RUNS}} corridas en el commit `{{COMMIT}}`, 17 modelos registrados. Todos los puntajes
son fuera de muestra con validación cruzada anidada (5 pliegues × 3 repeticiones) salvo que se indique otra
cosa. La versión 3.1.0 reorganiza el repositorio y traduce la documentación; los modelos se volvieron a
entrenar con el código reorganizado y reproducen los de v3.0.0.

Los archivos originales se revisan contra la especificación de datos del proveedor en cada corrida (contrato
de datos: 435 revisiones, 0 errores, 32 advertencias; ver el [README](../README.md#datos)).

## Modelos registrados

Los nombres del registro se conservan en inglés para que sean los mismos entre versiones.

| Nombre en el registro @ alias | Tarea | Datos | Modelo | Hiperparámetros finales | Puntaje fuera de muestra |
| --- | --- | --- | --- | --- | --- |
| `cgmt-starter-classifier@champion` | P(titular) | 510 prospectos, Combine + capital de draft | XGBoost | 200 árboles, profundidad 2, tasa de aprendizaje 0.03, peso mínimo por hijo 3, submuestreo de filas y columnas 0.8 | ROC-AUC 0.915 ± 0.025; F1 0.646; sensibilidad 0.80; precisión 0.55 |
| `cgmt-starter-classifier-movement@challenger` | P(titular) | 299 jugadores con tracking (93 titulares), + movimiento en partido | XGBoost | 200 árboles, profundidad 2, tasa de aprendizaje 0.1, peso mínimo por hijo 3 | ROC-AUC 0.884 ± 0.038 (sin ganancia sobre 0.884 sin movimiento; DeLong p = 0.81) |
| `cgmt-snap-burst-regressor@champion` | Snap Burst Index | 299 jugadores, todas las posiciones | Random forest | 200 árboles, profundidad 6, máx. variables 0.33, hoja mínima 3 | RMSE 9.57 vs 10.06 de la referencia (−5%); R² agrupado 0.07 [0.00, 0.12] |
| `cgmt-snap-burst-regressor-dl@champion` | Snap Burst Index | 72 linieros defensivos | Random forest | 200 árboles, profundidad 3, máx. variables 0.66, hoja mínima 10 | RMSE 7.85 vs 9.99 de la referencia (−21%); R² agrupado 0.37 [0.17, 0.50]; t corregida, Holm p = 0.001 |
| `cgmt-snap-burst-regressor-wr@champion` (v3.0) | Snap Burst Index | 63 receptores abiertos | Ridge | alpha = 300 | RMSE 8.45 vs 9.38 de la referencia (−10%); R² agrupado 0.22 [−0.01, 0.31]; t corregida p = 0.30 |
| `cgmt-starter-classifier-onfield@challenger` (v2.0) | P(titular) | 255 jugadores con índice de campo, + rendimiento de campo | Random forest | 200 árboles, profundidad 3, máx. variables 0.33, hoja mínima 10 | ROC-AUC 0.874 ± 0.057 (vs 0.867 sin datos de campo) |
| `cgmt-onfield-dl-getoff@champion` (v2.0) | Arranque DL sobre lo esperado (s) | 88 DL | Random forest | 200 árboles, profundidad 3, máx. variables 0.66, hoja mínima 3 | RMSE 0.060 vs 0.064 de la referencia; R² agrupado 0.13 [−0.02, 0.25] |
| `cgmt-onfield-dl-pressure@champion` (v2.0) | Tasa de presión DL sobre lo esperado | 89 DL | Random forest | 200 árboles, profundidad 6, máx. variables 0.33, hoja mínima 3 | No mejor que la referencia (R² agrupado 0.05 [−0.13, 0.17]) |
| `cgmt-onfield-ol-protection@champion` (v2.0) | Prob. de presión OL sobre lo esperado | 70 OL | Random forest | 200 árboles, profundidad 3, máx. variables 0.33, hoja mínima 10 | No mejor que la referencia (R² agrupado −0.01) |
| `cgmt-onfield-wrte-separation@champion` (v2.0) | Separación WR/TE sobre lo esperado (yd) | 96 WR/TE | Ridge | alpha = 100 | RMSE 0.169 vs 0.186 de la referencia; R² agrupado 0.19 [−0.01, 0.30] |
| `cgmt-onfield-dl-getoff-ct@challenger` (v2.2) | Arranque DL sobre lo esperado (s) | 88 DL, + tracking de pruebas | Random forest | 200 árboles, profundidad 3, máx. variables 0.66, hoja mínima 3 | RMSE 0.060, igual que sin tracking; cambio de R² −0.004 (p = 0.92) |
| `cgmt-starter-classifier-combine-tracking@challenger` (v2.2) | P(titular) | 510 prospectos, Combine + draft + tracking de pruebas | XGBoost | 200 árboles, profundidad 2, tasa de aprendizaje 0.03, peso mínimo por hijo 3 | ROC-AUC 0.913 vs 0.915 sin tracking (DeLong p = 0.30) |
| `cgmt-snap-burst-regressor-{dl,wr}-ct`, `cgmt-onfield-{dl-pressure,ol-protection,wrte-separation}-ct` @challenger (v2.2; WR v3.0) | Las de sus campeones | + tracking de pruebas | El mejor de los cuatro por tarea | Ver `reportes/metricas/regression_final_models.json` | Sin ganancia sobre el campeón (cambio de R² +0.004 snap burst DL, −0.007 WR, −0.03 a −0.05 en campo) |

**Entrada.** Un DataFrame de pandas con `combine_position` más las columnas del conjunto de variables elegido
(`FEATURE_SETS` en `fuentes/datos_prep.py`); la firma y el ejemplo de entrada registrados definen el esquema
exacto. Se permiten pruebas faltantes (se imputan dentro del pipeline). Cada modelo se guarda junto con
`datos_prep.py` y `train.py`, así carga sin instalar el proyecto:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
clf = mlflow.sklearn.load_model("models:/cgmt-starter-classifier@champion")
```

## Uso previsto

- **Campeón de titulares:** una estimación previa al draft de si un prospecto llegará a titular regular, dados
  su perfil del Combine y un turno de draft esperado. Solo apoyo a la decisión.
- **Regresores de la DL y de receptores:** proyectan el snap burst en partido (SBI) de un liniero defensivo o
  un receptor a partir de su perfil del Combine; la brecha entre el SBI proyectado y el observado señala
  jugadores para revisar en video. El modelo de la DL es el más sólido (ganancia significativa sobre la
  posición sola); la ganancia del de receptores es consistente entre pliegues pero no significativa con la t
  corregida.
- **Regresores de campo (v2.0):** proyectan el rendimiento de campo de temporada de un prospecto a partir de su
  perfil del Combine; útiles solo para el arranque de la DL y la separación de WR/TE. Los modelos de presión DL
  y protección OL se registran por completitud y no superan a la media por posición.
- **Retadores con tracking de pruebas (v2.2):** se registran para que la comparación sea reproducible; ninguno
  supera a su campeón. Necesitan las cinco variables `ct_*` del tracking de las pruebas al predecir.
- **No previsto:** ordenar DB, OL o TE por snap burst (ICC 0.26, 0.40 y 0.08; no son lo bastante estables para
  ordenar jugadores), ni leer el SBI de un liniero defensivo como habilidad individual en lugar de rol (sin el
  contexto de jugada el ICC de la DL baja de 0.63 a 0.14); tampoco usar los retadores de movimiento o de campo
  para decisiones de draft (sus variables solo existen después del draft).
- **Nota sobre el proyector:** el pipeline campeón no cambió en v2.0. El conjunto `combine_draft_proj` (con el
  `OnFieldProjector` apilado) no superó a Combine + draft (DeLong p ≥ 0.05), así que no se promovió. En
  cualquier pipeline con el proyector, las entradas `onfield_target_*` son etiquetas de entrenamiento y pueden
  pasarse como NaN al predecir.

## Datos de entrenamiento y objetivo

- 510 prospectos de las clases 2023–2025; carreras hasta la temporada 2025.
- Titular = inició el 50% o más de los partidos de temporada regular disponibles (17 × temporadas en la liga):
  103 positivos.
- Movimiento en partido del tracking a 10 Hz: temporada 2023, semanas 0–17 de 2024, semana 18 y playoffs de
  2024 y 2025 (temporada regular y playoffs, sin pretemporada); 299 jugadores con 3+ jugadas, mediana de 39
  jugadas.

## Limitaciones y riesgos conocidos

- El capital de draft domina el modelo de titulares (importancia por permutación 0.29 de AUC frente a menos de
  0.01 de cualquier otra variable); en parte refleja la inversión del equipo, así que pronostica oportunidad
  tanto como habilidad.
- Muestras chicas: 72 DL y 63 WR en los modelos del snap burst. La ganancia de la DL pasa la prueba t corregida
  para remuestreo (Holm p = 0.001); la de los WR no (p = 0.30) y su R² por pliegue varía mucho (DE 0.56).
- El tracking no tiene las semanas 1–17 de 2025, y `game_tracking_2023_5.csv` se detiene exactamente en
  1,048,575 filas (el límite de una hoja de cálculo), así que falta al azar una parte de 2023; el tracking no
  trae rivales ni balón. Las métricas de campo cubren las temporadas regulares 2023–2025, con "lo esperado"
  calculado con las jugadas de los prospectos, no de toda la liga.
- Tracking de pruebas: el inicio de un intento registrado no se repite, así que no se usan métricas
  cronometradas desde la salida. La fiabilidad del shuttle y el 3 conos descansa en unos 40 jugadores con dos
  intentos.
- El retador con datos de campo es circular por diseño (el rendimiento y los inicios comparten temporadas).
- Los modelos se serializan con cloudpickle (el pipeline contiene un transformador del proyecto): cárguelos
  solo desde este registro de confianza.

## Equidad y ética

Las variables son mediciones atléticas, edad, conferencia y turno de draft; no se usa ningún atributo
protegido. Los resultados tratan del rendimiento atlético profesional y deben apoyar, no reemplazar, el
criterio de scouts y entrenadores.
