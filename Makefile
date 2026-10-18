# Atajos para reproducir el proyecto. En Colab se usa fuentes/entrena.ipynb, que llama a lo mismo.
PY ?= python3
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: instalar pruebas rapida validar verificar datos traduccion modelos estadistica figuras todo reubicar ui limpiar

instalar:           ## instala las dependencias fijadas
	$(PY) -m pip install -r requirements.txt

pruebas:            ## pruebas unitarias (reglas de limpieza, métricas de movimiento, estadística, contrato, determinismo)
	$(PY) -m pytest -q

rapida:             ## todo el flujo con CV y rejillas mínimas (CI, ~10 min; carpetas de salida propias)
	$(PY) fuentes/train.py --config configuracion/rapida.yaml --etapa todo

validar:            ## solo el contrato de datos: originales vs especificación -> reportes/metricas/data_contract_*
	$(PY) fuentes/datos_prep.py --etapa contrato

verificar:          ## compara el hash de cada archivo original con datos/datos_ini/manifiesto.json
	$(PY) fuentes/datos_prep.py --etapa verificar

datos:              ## contrato + ingesta + limpieza + movimiento + métricas de campo + manifiesto
	$(PY) fuentes/datos_prep.py --etapa datos

traduccion:         ## asociaciones prueba-partido, fiabilidad y GLM sobre los datos limpios
	$(PY) fuentes/train.py --etapa traduccion

modelos:            ## traducción + CV anidada de regresión y clasificación + registro en MLflow
	$(PY) fuentes/train.py --etapa modelos

estadistica:        ## recalcula las pruebas ómnibus desde los puntajes por pliegue (sin reentrenar)
	$(PY) fuentes/train.py --etapa estadistica

figuras:            ## regenera reportes/figuras desde las métricas guardadas
	$(PY) fuentes/train.py --etapa figuras

todo:               ## corrida completa reproducible (~90 min en 2 núcleos)
	$(PY) fuentes/train.py --etapa todo

reubicar:           ## una vez tras clonar o extraer: apunta las rutas del almacén MLflow a esta carpeta
	$(PY) fuentes/train.py --etapa reubicar

ui:                 ## explora corridas y el registro de modelos
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

limpiar:            ## borra las salidas de la corrida rápida (conserva datos originales, limpios y mlflow.db)
	rm -rf datos/rapida reportes/rapida mlflow/rapida.db mlflow/rapida_artifacts
