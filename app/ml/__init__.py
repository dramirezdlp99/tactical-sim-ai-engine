"""
Guardar en: tactical-sim-ai-engine/app/ml/__init__.py

Paquete de aprendizaje automatico del AI Engine.

Contenido:
  - features.py : extraccion de caracteristicas (la usan entrenamiento E inferencia).
  - dataset.py  : generador de datos SINTETICOS (no hay datos historicos reales).
  - train.py    : entrenamiento (`python -m app.ml.train`). Usa numpy, que solo
                  hace falta en la maquina donde se entrena, NO en el servidor.
  - models/     : pesos entrenados en JSON (se leen sin numpy en produccion).
"""
