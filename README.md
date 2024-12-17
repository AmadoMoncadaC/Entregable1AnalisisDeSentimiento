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

```
## Uso
### Entrenamiento y evaluación:
- **El script carga un conjunto de reseñas de productos predefinidas y las divide en un conjunto de entrenamiento y un conjunto de prueba.
- **Se entrena un modelo de regresión logística utilizando el conjunto de entrenamiento y luego se evalúa la precisión utilizando el conjunto de prueba.
### Predicción:
El usuario puede ingresar una reseña de producto, y el modelo predice si el sentimiento es positivo o negativo.

### Ejemplo:
python
Copiar código
```
# Predicción de una nueva reseña
reseña_usuario = input("Ingresa tu reseña de producto para analizar el sentimiento: ")
sentimiento = predecir_sentimiento(reseña_usuario)
print(f"La reseña es clasificada como: {sentimiento}")
```
Salida del modelo:
El script imprimirá:

La precisión del modelo en porcentaje.
Un reporte detallado de la clasificación, que incluye precisión, recall y f1-score para ambas clases (positiva y negativa).

## Explicación del código
- **Preprocesamiento del texto:
    Se utiliza la técnica TF-IDF para transformar el texto de las reseñas en vectores numéricos. También se incluyen stopwords en español y algunas palabras clave adicionales (como           "pesimo" y "malo") para mejorar el rendimiento del modelo.
- **Modelo de regresión logística:
    El modelo se entrena utilizando el conjunto de entrenamiento y luego se realiza una predicción sobre el conjunto de prueba.
- **Evaluación:
    Se calcula la precisión y se genera un reporte detallado que evalúa el desempeño del modelo en términos de precisión, recall y F1-score.

## Contribuciones
Si deseas contribuir a este proyecto, por favor haz un fork del repositorio y envía un pull request. Asegúrate de que tu código esté bien documentado y de seguir buenas prácticas de programación.

## Licencia
Codigo libre.
