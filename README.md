# Proyecto NFL: del Combine al juego (v3.1.0)

¿Las pruebas del NFL Combine pronostican cómo se mueve y rinde un prospecto en los partidos de temporada
regular, y eso dice algo sobre quién llega a ser titular? Este proyecto lo responde para una franja concreta
del movimiento, **los primeros segundos después del snap**, medida de dos formas:

- el **Snap Burst Index (SBI)**: la distancia recorrida 2.0 s después del snap según el tracking a 10 Hz (el
  análogo en partido del parcial de 10 yardas), promediada sobre las jugadas de cada jugador, contraída hacia
  su grupo de posición por Bayes empírico y escalada a 50 ± 10 dentro de la posición;
- **métricas de campo de temporada completa sobre lo esperado** (desde `player_play`): arranque y tasa de
  presión de la línea defensiva, probabilidad de presión y presiones permitidas de la línea ofensiva, y
  separación de receptores y alas cerradas al lanzamiento, cada una comparada con la jugada media de un
  prospecto en el mismo contexto (alineación, duración del dropback, ruta).

**Respuesta corta.** Las pruebas de sprint y salto sí se trasladan al juego, con más claridad en la línea
defensiva: el Combine pronostica el snap burst de un liniero con 21% menos error que la posición sola, y su
arranque en partido, que es el camino por el que el atletismo se vuelve presión al pasador. Pero una vez que se
conoce el turno de draft, ni el movimiento en partido ni el tracking de las pruebas mejoran la predicción de
quién será titular.

El **reporte técnico final** (métodos, figuras, métricas, pruebas estadísticas y conclusiones) y el
**historial de versiones** están en [CHANGELOG.md](CHANGELOG.md). Este README es la guía para reproducir el
proyecto y explica las decisiones de diseño.

## Estructura del repositorio

La estructura sigue el documento `Estructura_Gitub_NFL.docx`. Las carpetas y archivos marcados como
**complemento** no estaban en ese documento; se agregaron porque el proyecto los necesita para ser
reproducible (configuración, pruebas, resultados, registro de modelos e integración continua).

```
NFL/
│── datos/
│   ├── datos_ini/            # datos originales: los diez CSV comprimidos (.csv.gz) + manifiesto.json
│   └── datos_limp/           # datos limpios listos para modelar (parquet)
│       └── intermedios/      #   cuadros de tracking limpios y contexto de jugada
│── fuentes/
│   ├── entrena.ipynb         # notebook de Colab (entrenamiento y evaluación)
│   ├── datos_prep.py         # funciones de limpieza y preparación de datos
│   └── train.py              # script de entrenamiento, evaluación y registro en MLflow
│── README.md                 # guía de reproducción y decisiones (este archivo)
│── CHANGELOG.md              # reporte final + historial de versiones
│
│   Complementos
│── configuracion/            # complemento: parámetros del flujo
│   ├── config.yaml           #   rutas, reglas de negocio, validación cruzada y rejillas (corrida de entrega)
│   ├── rapida.yaml           #   el mismo flujo con rejillas mínimas (CI, unos 10 min)
│   └── contrato_datos.yaml   #   la especificación del proveedor como contrato que se revisa en cada corrida
│── pruebas/                  # complemento: 35 pruebas unitarias (pytest)
│── reportes/                 # complemento: resultados
│   ├── figuras/              #   las 21 figuras del reporte (PNG)
│   └── metricas/             #   cada tabla detrás del reporte (CSV/JSON)
│── documentos/               # complemento
│   ├── Data_structure_CSV_files.docx   # especificación de los datos del proveedor
│   └── tarjeta_modelo.md     #   tarjeta de los modelos registrados
│── mlflow/                   # complemento: almacén de experimentos y registro de modelos
│   ├── mlflow.db             #   corridas, métricas, etiquetas y modelos registrados (SQLite)
│   └── mlartifacts/          #   artefactos: modelos, puntajes por pliegue, predicciones
│── .github/workflows/ci.yml  # complemento: integración continua (pruebas, hashes, contrato, corrida rápida)
│── requirements.txt          # complemento: dependencias fijadas (Python 3.13)
│── pyproject.toml            # complemento: configuración de pytest
│── Makefile                  # complemento: atajos para cada etapa
└── .gitignore
```

**Qué hay en cada archivo de `fuentes/`.**

| Archivo | Contenido |
| --- | --- |
| `datos_prep.py` | Configuración y semilla; resolución de archivos y manifiesto (versionado de datos); contrato de datos; limpieza de prospectos (reglas R1–R7); tracking de partido → métricas por jugada → perfiles y SBI (R8–R9); `player_play` → métricas de campo sobre lo esperado (R10–R15); tracking de las pruebas del Combine (R16); conjuntos de variables y tablas de modelado. |
| `train.py` | Preprocesamiento y CV anidada (imputador por posición, proyector apilado, pipelines); pruebas estadísticas (t corregida, Wilcoxon-Holm, Friedman, DeLong, bootstrap, mediación); MLflow (seguimiento, etiquetas, registro); análisis de traducción; regresión y clasificación; las 21 figuras; la línea de comandos. |
| `entrena.ipynb` | El flujo completo en Google Colab, paso a paso, con las tablas y figuras de evaluación y la carga de los modelos registrados. |

## Datos

Los diez archivos originales están en `datos/datos_ini/`, comprimidos con gzip: `player_play_3.csv` pesa 147 MB
y GitHub no acepta archivos de más de 100 MB. Comprimidos ocupan 100 MB en lugar de 427 MB. pandas los lee sin
descomprimir a disco, y el código acepta igual un `.csv` que un `.csv.gz`, así que una entrega nueva sin
comprimir funciona sin cambios.

| Archivo | Contenido | Registros |
| --- | --- | --- |
| `combine_results.csv.gz` | Resultados oficiales del Combine 2023–2025 | 510 prospectos |
| `players.csv.gz` | Datos biográficos y de draft | 510 |
| `player_career_successes.csv.gz` | Partidos jugados e iniciados hasta 2025 | 510 |
| `games.csv.gz` | Calendario y tipo de partido | 1,002 partidos |
| `game_tracking_2023_5.csv.gz` | Tracking a 10 Hz, temporada 2023 (cortado en el límite de una hoja de cálculo) | 1,048,575 |
| `game_tracking_2024_1_5.csv.gz` | Tracking, semanas 0–17 de 2024 | 951,733 |
| `game_tracking_2024_2.csv.gz`, `game_tracking_2025_2.csv.gz` | Tracking, semana 18 y playoffs de 2024 y 2025 | 96,842 y 97,147 |
| `player_play_3.csv.gz` | Una fila por jugador y jugada, 2023–2025 | 314,197 jugador-jugadas |
| `combine_tracking_6.csv.gz` | Tracking a 10 Hz de las pruebas del Combine | 463,189 cuadros |

**Manifiesto.** `datos/datos_ini/manifiesto.json` guarda el SHA-256, el tamaño y el número de líneas del
contenido descomprimido de cada archivo, más una versión combinada `data_version = a75ba51ae4b9`. Comprimir
no cambia la versión de datos. `make verificar` vuelve a calcular los hashes y falla ante cualquier diferencia.

**Contrato de datos.** La especificación del proveedor (`documentos/Data_structure_CSV_files.docx`) está
codificada en `configuracion/contrato_datos.yaml` y se revisa en cada corrida (`make validar`, también el
primer paso de la etapa de datos). El reporte `reportes/metricas/data_contract_report.csv` tiene una fila por
revisión, con su severidad y el manejo que aplica el flujo. Los errores (archivo o columna obligatoria
faltante, clave duplicada, clave foránea rota) detienen la corrida; las desviaciones son advertencias que
manejan las reglas de negocio. Con los archivos entregados: **435 revisiones, 0 errores, 32 advertencias.**

| Especificación | Entregado | Efecto |
| --- | --- | --- |
| `combine_tracking.csv`, 6,310 intentos | `combine_tracking_6.csv`: 6,310 intentos, 510 prospectos | El carril del 40 queda fuera del marco de 0–53.3 yd; las filas del balón superan los rangos de un jugador y se descartan (R16) |
| `game_tracking_2023/2024/2025.csv`, temporadas completas | Temporada 2023, semanas 0–17 de 2024, semana 18 y playoffs de 2024 y 2025; faltan las semanas 1–17 de 2025 | 343 prospectos con tracking tras R8b; el SBI usa 299 jugadores con 3+ jugadas (24,828 jugador-jugadas) |
| Archivos completos | `game_tracking_2023_5.csv` tiene exactamente 1,048,575 filas, el límite de una hoja de cálculo | Casi seguro se cortó al guardarse; R8c descarta la última jugador-jugada (posiblemente parcial) y el contrato lo advierte |
| `player_play.csv`, 316,338 filas | `player_play_3.csv`, 314,197 filas (−0.68%) | Los resultados usan las filas entregadas |
| Partidos REG/POST, semanas 1–18 | También PRE (147 partidos, semanas 0–3) y playoffs en semanas 19–23 | R10: el campo usa REG; R8b: el tracking usa REG + POST |
| `down` 1–4 | 961 filas con down 0 (intentos de dos puntos) | R15 las quita (ninguna era de una familia medida) |
| Indicadores 1.0/nulo | Texto TRUE/FALSE | El lector booleano acepta ambos |

## Cómo reproducirlo

### Opción 1: Google Colab

1. Suba este repositorio a GitHub (ver [Publicar en GitHub](#publicar-en-github)).
2. Abra `fuentes/entrena.ipynb` en Colab: en colab.research.google.com elija *Archivo → Abrir cuaderno →
   GitHub* y pegue la URL del repositorio, o use el enlace
   `https://colab.research.google.com/github/<usuario>/NFL/blob/main/fuentes/entrena.ipynb`.
3. En la primera celda escriba la URL del repositorio en `REPO_URL` y elija `MODO = "rapida"` (unos 10 min,
   rejillas mínimas, salidas en carpetas propias) o `MODO = "completa"` (la corrida de entrega, de 60 a 90 min
   en los 2 núcleos de Colab).
4. Ejecute todo (*Entorno de ejecución → Ejecutar todas*). El notebook clona el repositorio, instala las
   dependencias, valida los datos, prepara los datos limpios, entrena y evalúa, muestra las tablas y figuras, y
   carga los modelos registrados.

Sin entrenar nada, el notebook también puede cargar directamente los 17 modelos de la corrida de entrega que
vienen en `mlflow/`.

### Opción 2: en su computadora

Requiere Python 3.13 (las versiones están fijadas en `requirements.txt`) y `make`.

```bash
git clone https://github.com/<usuario>/NFL.git && cd NFL
python -m pip install -r requirements.txt
make reubicar     # una vez tras clonar: apunta las rutas del almacén MLflow a esta carpeta
make verificar    # hashes de los datos originales vs el manifiesto, unos 5 s
make validar      # solo el contrato de datos, unos 10 s
make pruebas      # 35 pruebas unitarias, unos 5 s
make rapida       # todo el flujo con rejillas mínimas, unos 10 min (carpetas propias, no toca la entrega)
make todo         # corrida de entrega: datos, traducción, modelos, estadística y figuras, unos 60 min
make ui           # interfaz de MLflow: corridas, métricas y modelos registrados
```

Cada etapa también corre sola, en este orden:

| Etapa | Comando | Qué hace | Escribe |
| --- | --- | --- | --- |
| contrato | `make validar` | Revisa los diez CSV contra la especificación | `reportes/metricas/data_contract_*` |
| datos | `make datos` | Contrato, ingesta, manifiesto, limpieza (R1–R16), métricas de movimiento y de campo | `datos/datos_limp/`, auditorías en `reportes/metricas/` |
| traducción | `make traduccion` | Correlaciones prueba-partido y prueba-campo, fiabilidad, mediación, GLM de titulares | `reportes/metricas/` |
| modelos | `make modelos` | Traducción + CV anidada de regresión y clasificación, pruebas, reajuste final y registro | `reportes/metricas/`, `mlflow/` |
| estadística | `make estadistica` | Vuelve a calcular las pruebas ómnibus desde los puntajes por pliegue, sin reentrenar | `reportes/metricas/` |
| figuras | `make figuras` | Dibuja las 21 figuras desde las métricas guardadas | `reportes/figuras/` |

Sin `make`: `python fuentes/train.py --etapa todo` (o `contrato`, `datos`, `traduccion`, `modelos`,
`estadistica`, `figuras`, `reubicar`), y `python fuentes/datos_prep.py --etapa datos|contrato|verificar`.
Ambos aceptan `--config configuracion/rapida.yaml`.

Todo lleva semilla (`seed: 42`); una prueba verifica que dos ajustes con la misma semilla den predicciones
idénticas, y la corrida completa reproduce los mismos números.

### Cargar un modelo registrado

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")       # desde la raíz del repositorio
clf = mlflow.sklearn.load_model("models:/cgmt-starter-classifier@champion")
info = mlflow.models.get_model_info("models:/cgmt-starter-classifier@champion")
print(info.signature)                                         # columnas de entrada esperadas
```

Los modelos se guardan junto con `datos_prep.py` y `train.py`, así cargan en cualquier máquina sin instalar
nada más que `requirements.txt`. La entrada es un DataFrame con `combine_position` más las columnas del
conjunto de variables del modelo; las pruebas faltantes se permiten (se imputan dentro del pipeline). La
[tarjeta del modelo](documentos/tarjeta_modelo.md) describe cada uno de los 17 modelos registrados.

## Decisiones de diseño

Cada decisión responde a un problema que apareció en los datos; el razonamiento completo está en el
[reporte](CHANGELOG.md) y en el historial de versiones.

1. **El SBI es una media contraída, no un percentil alto.** La primera versión usaba el percentil 80 de las
   jugadas de cada jugador ("capacidad"). Con pocas jugadas ese percentil lee de menos, y como los titulares
   juegan más, el sesgo fabricaba una correlación con la titularidad (0.23 → 0.08 al corregirlo). La media con
   contracción por Bayes empírico no tiene ese sesgo; el percentil 80 queda solo como análisis de sensibilidad.
2. **El objetivo se normaliza dentro de la posición.** Con el burst en yardas, la media por posición daba
   R² = 0.94 y ocultaba la pregunta que importa: dentro de una posición, ¿el Combine distingue a quién se mueve
   mejor? El SBI (50 ± 10 dentro de la posición) obliga a responder esa pregunta.
3. **"Sobre lo esperado" para las métricas de campo.** La presión, la separación o el arranque dependen de la
   alineación, la ruta y la duración del dropback. Cada jugada se compara con la media de las jugadas de
   prospectos en el mismo contexto, y después se contrae por Bayes empírico.
4. **Validación cruzada anidada y pruebas conservadoras.** 5 pliegues × 3 repeticiones por fuera y GridSearchCV
   de 3 pliegues por dentro: los hiperparámetros nunca ven el pliegue de prueba. Como los pliegues comparten
   jugadores, se usa la prueba t corregida de Nadeau y Bengio, con Wilcoxon, Holm, Friedman y DeLong como
   controles. Una ganancia solo se llama significativa si pasa la prueba corregida.
5. **Sin fugas de información.** La imputación por mediana de posición y el escalado se aprenden dentro de
   cada pliegue; las variables de movimiento son promedios por jugada, nunca conteos (los conteos siguen al
   tiempo de juego); la posición NFL nunca es variable porque se decide después del draft.
6. **Campeón y retador.** El campeón de titulares solo puede usar lo que se sabe antes del draft (Combine +
   capital de draft). Un conjunto nuevo lo reemplaza solo con un AUC medio mayor y DeLong p < 0.05. Los modelos
   con información posterior al draft (movimiento, rendimiento de campo) quedan como retadores de diagnóstico.
7. **El tracking de las pruebas se filtra por fiabilidad.** Solo se usan las métricas que se repiten entre los
   intentos de un jugador (r ≥ 0.5), una por constructo; el inicio de un intento registrado no se repite y se
   descarta.
8. **Un contrato de datos explícito.** La especificación del proveedor se revisa en cada corrida, así una
   entrega distinta (archivos partidos, columnas extra, un archivo cortado) se detecta antes de llegar a un
   modelo y su manejo queda documentado.
9. **Dos módulos y una configuración.** Toda la preparación vive en `datos_prep.py` y todo el modelado en
   `train.py`, como pide la estructura; los umbrales y rejillas viven en `configuracion/config.yaml`, nunca en
   el código.
10. **MLflow local con SQLite.** El registro de modelos funciona sin servidor ni conexión y viaja con el
    repositorio; `make reubicar` corrige las rutas absolutas que MLflow guarda.
11. **Identificadores en inglés, documentación en español.** Los nombres de funciones, columnas, métricas y
    modelos registrados se conservan en inglés porque los comparten el código, los archivos de métricas y el
    registro de MLflow entre versiones; todo lo que se lee (documentación, comentarios, mensajes, figuras) está
    en español.

## Resultados principales

| Pregunta | Resultado |
| --- | --- |
| ¿Se repite el snap burst de un jugador entre jugadas? | Sí en la línea defensiva (ICC 0.63, casi todo por el rol) y en receptores (0.62, propio del jugador); débil en backs defensivos (0.26) |
| ¿Pronostica el Combine el SBI? | Línea defensiva: RMSE 7.85 vs 9.99 de la posición sola (−21%), R² 0.37, Holm p = 0.001. Receptores: −10%, evidencia moderada. Todas las posiciones: −5% |
| ¿Qué pruebas se trasladan? | El 40, el parcial de 10 yd, el salto largo y el vertical (ρ 0.25–0.43 dentro de la posición); el compuesto NGS y el press de banca no |
| ¿Pronostica el Combine el rendimiento de campo? | Arranque de la DL (R² 0.13) y separación de WR/TE (0.19, por el tamaño); no la presión ni la protección de pase |
| ¿Ayuda el movimiento a predecir titulares? | No después del draft: AUC 0.84 → 0.91 con el capital de draft; el movimiento suma −0.003 a 0 (DeLong p ≥ 0.47) |
| ¿Suma el tracking de las pruebas? | No: reproduce el cronómetro (ρ −0.97 con el 40 oficial) y no mejora ningún modelo |

Las conclusiones para scouts, oficinas generales y entrenadores, con las tablas de brecha de traducción, están
en la sección 10 del [reporte](CHANGELOG.md#10-conclusiones-accionables).

## Versionado y MLOps

- **Código:** git con etiquetas de versión semántica y cada cambio descrito en [CHANGELOG.md](CHANGELOG.md).
  MAYOR = cambió la definición de una métrica u objetivo; MENOR = análisis o modelo nuevo; PARCHE = no cambia
  ningún número de `reportes/metricas/`.
- **Datos:** el manifiesto guarda el hash de cada archivo original y la versión combinada `data_version`; la
  CI falla si un hash no coincide. El contrato de datos se versiona con el código.
- **Experimentos:** cada corrida de MLflow lleva `pipeline_version`, `data_version`, `git_commit` y `seed`, los
  puntajes de cada pliegue, las predicciones fuera de pliegue y los hiperparámetros elegidos.
- **Modelos:** registrados con alias (`champion`, `challenger`), firma, ejemplo de entrada y la etiqueta
  `validation_status = nested-cv-passed`.
- **Integración continua:** `.github/workflows/ci.yml` corre las pruebas, los hashes, el contrato y la corrida
  rápida en cada push a `main`.

## Limitaciones y datos pendientes

- Faltan las semanas 1–17 de 2025 en el tracking de partido, y `game_tracking_2023_5.csv` se detiene en
  1,048,575 filas, así que falta al azar una parte de 2023. Entregar el archivo de 2023 sin cortar y la
  temporada regular de 2025 haría más precisos los modelos del SBI; el flujo encuentra los archivos nuevos por
  patrón y los valida sin cambiar código.
- El tracking trae de 1 a 4 jugadores por jugada, sin rivales ni balón, así que el movimiento se juzga con la
  cinemática del propio jugador.
- Las muestras por posición son chicas (72 linieros defensivos y 63 receptores en los modelos del SBI).
- Los backs defensivos no tienen métrica de campo: `player_play` no trae resultados de cobertura.
- "Lo esperado" se calcula con las jugadas de los propios prospectos, no con las de toda la liga.

## Publicar en GitHub

El repositorio ya tiene su historial de git y sus etiquetas de versión. Cree un repositorio vacío en GitHub
(sin README ni licencia) y súbalo:

```bash
git remote add origin https://github.com/<usuario>/NFL.git
git push -u origin main --tags
```

Ningún archivo supera el límite de 100 MB de GitHub; el más grande es `player_play_3.csv.gz` con 41 MB, y en
GitHub el repositorio ocupa unos 160 MB. Si el repositorio va a ser público, confirme antes que puede publicar
los datos del proveedor; si no, quítelos del control de versiones (`git rm -r --cached datos/datos_ini`) y
compártalos por otro medio: el flujo los encuentra igual al copiarlos en `datos/datos_ini/`.
