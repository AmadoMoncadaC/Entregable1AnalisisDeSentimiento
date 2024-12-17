# Entregable1AnalisisDeSentimiento

# Análisis de Sentimientos de Reseñas de Productos

Este proyecto tiene como objetivo clasificar reseñas de productos en dos categorías de sentimiento: **Positiva** y **Negativa**. Utiliza técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático para predecir el sentimiento de una reseña ingresada por el usuario.

## Descripción

El modelo se entrena utilizando un conjunto de reseñas de productos, que están etiquetadas con su respectivo sentimiento (1 para positivo y 0 para negativo). El proyecto usa **TF-IDF (Term Frequency-Inverse Document Frequency)** para convertir el texto en vectores numéricos y un **modelo de regresión logística** para realizar las predicciones.

### Características principales:
- **Preprocesamiento de texto**: Uso de stopwords en español y palabras clave adicionales para mejorar la precisión del modelo.
- **Entrenamiento y evaluación del modelo**: El modelo se entrena con un conjunto de datos de reseñas y se evalúa con un conjunto de prueba, mostrando métricas de precisión y un reporte de clasificación.
- **Predicción de sentimiento en tiempo real**: El usuario puede ingresar una nueva reseña y el modelo predice si el sentimiento es positivo o negativo.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas de Python:

- `pandas`
- `scikit-learn`

Puedes instalarlas con pip:

```bash
pip install pandas scikit-learn

 
