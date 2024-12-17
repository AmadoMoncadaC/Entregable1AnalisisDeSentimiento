# Importación de bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Lista de stopwords en español
stopwords_espanol = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'para', 'con', 'una',
    'su', 'no', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si', 'me', 'es', 'porque',
    'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'también', 'otros', 'fue', 'ha', 'está', 'yo',
    'hasta', 'hay', 'donde', 'quien', 'después', 'te', 'ni', 'nos', 'durante', 'todos', 'algunos', 'este', 'él',
    'ellas', 'ante', 'ese', 'esto', 'esa', 'esos', 'esas', 'unos', 'una', 'su', 'de', 'todo', 'mismo', 'ya'
]

# Crear el conjunto de datos de ejemplo
data = {
    'review': [
        'Este producto es increíble, lo recomiendo mucho',
        'Muy mala calidad, no lo volvería a comprar',
        'Excelente relación calidad-precio, muy satisfecho',
        'No me gustó el producto, llegó defectuoso',
        'Muy bueno, cumple con lo prometido',
        'Horrible, no funciona como debería',
        'Perfecto para lo que necesitaba, lo volveré a comprar',
        'No vale lo que cuesta, decepcionado',
        'Pesimo producto malo',  # Añadido un ejemplo negativo
        'No funciona nada bien, muy malo',  # Otro ejemplo negativo
        # Generación de más datos sintéticos (positivos)
        'Me encantó este producto, es fantástico',
        'Excelente calidad, superó mis expectativas',
        'Lo compraría nuevamente, es perfecto para lo que busco',
        'Es el mejor producto que he comprado, me sorprendió mucho',
        'Estoy muy feliz con la compra, totalmente recomendado',
        'Muy útil, y el precio es justo para lo que ofrece',
        # Generación de más datos sintéticos (negativos)
        'El producto llegó roto y no funcionaba, pésima calidad',
        'Muy decepcionado, no lo recomiendo para nada',
        'El artículo no vale lo que cuesta, no es bueno',
        'No cumple con lo prometido, muy insatisfecho',
        'No lo volvería a comprar, fue una mala experiencia',
        'Producto defectuoso, muy mala compra',
        'No funciona como se describe, no lo recomiendo',
        'Compre este producto y fue una total decepción',
        # Nuevas reseñas
        'Este producto superó mis expectativas, excelente calidad y muy útil',  # Positiva
        'Totalmente decepcionado, llegó roto y no sirve',  # Negativa
        'Muy buen desempeño, lo recomiendo totalmente',  # Positiva
        'No lo compraría de nuevo, el material es muy frágil',  # Negativa
        'Increíble relación calidad-precio, me encanta',  # Positiva
        'Producto defectuoso, no funciona como se espera',  # Negativa
        'El producto es exactamente lo que necesitaba, fantástico',  # Positiva
        'Una gran decepción, no cumple lo prometido',  # Negativa
        'El servicio de entrega fue rápido y el producto es excelente',  # Positiva
        'No vale la pena, se rompió al poco tiempo de uso',  # Negativa
        'Me sorprendió lo bien que funciona, totalmente recomendado',  # Positiva
        'Malísimo, no tiene buena calidad, muy decepcionante',  # Negativa
        'Es el mejor producto que he comprado, sin dudas',  # Positiva
        'Muy mala experiencia, llegó tarde y mal embalado',  # Negativa
        'Me encanta, superó mis expectativas en todo',  # Positiva
        'Definitivamente no lo volveré a comprar, no funciona',  # Negativa
        'Muy bueno, cumple con todo lo prometido',  # Positiva
        'Producto de mala calidad, no lo recomiendo',  # Negativa
        'Es un producto increíble, lo usaré todos los días',  # Positiva
        'No es lo que esperaba, no lo recomiendo en absoluto',  # Negativa
        'Es perfecto para mi casa, me encanta',  # Positiva
        'Decepcionante, no funciona como se describe',  # Negativa
        'Excelente, vale cada centavo que pagué',  # Positiva
        'No lo recomiendo para nada, muy malo',  # Negativa
        'Me sorprendió lo bien que funciona, perfecto para lo que buscaba',  # Positiva
        'No sirve para nada, muy deficiente',  # Negativa
        'Muy buena compra, quedé satisfecho con el producto',  # Positiva
        'Malo, no lo compres'
    ],
    'sentiment': [
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0,  # 1 = positiva, 0 = negativa
        1, 1, 1, 1, 1, 1,  # Positivas adicionales
        0, 0, 0, 0, 0, 0, 0, 0,  # Negativas adicionales
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0

    ]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Agregar algunas palabras clave negativas específicas al vectorizador
stopwords_adicionales = ['pesimo', 'malo', 'horrible', 'defectuoso', 'decepcionado']

# Crear una lista completa de stopwords
stop_words_completo = stopwords_espanol + stopwords_adicionales

# Mejorar la vectorización con bigramas y ajuste de max_df/min_df
tfidf = TfidfVectorizer(stop_words=stop_words_completo, ngram_range=(1, 2), max_df=0.95, min_df=5, lowercase=True)

# Transformar el texto a vectores
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Balancear las clases utilizando SMOTE (sobremuestreo)
smote = SMOTE(random_state=42)

# Crear un pipeline con SMOTE y el modelo de Regresión Logística
pipeline = Pipeline([
    ('smote', smote),
    ('logreg', LogisticRegression(class_weight='balanced'))
])

# Ajuste de hiperparámetros con GridSearchCV para la regresión logística
param_grid = {
    'logreg__C': [0.1, 1, 10, 100],  # Parámetros de regularización
    'logreg__penalty': ['l2'],  # Regularización L2
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Modelo entrenado con los mejores parámetros encontrados
best_model = grid_search.best_estimator_

# Realizar predicciones
y_pred_prob = best_model.predict_proba(X_test_tfidf)[:, 1]  # Predicciones de probabilidad para ajustar el umbral

# Ajuste del umbral para mejorar el recall de la clase minoritaria (0)
y_pred_adjusted = (y_pred_prob > 0.3).astype(int)  # Ajuste el umbral a 0.3

# Evaluar el rendimiento
print(f"Mejores parámetros de la regresión logística: {grid_search.best_params_}")
print(f'Accuracy: {accuracy_score(y_test, y_pred_adjusted) * 100:.2f}%')
print(f'F1-Score: {f1_score(y_test, y_pred_adjusted)}')
print(f'Recall: {recall_score(y_test, y_pred_adjusted)}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_pred_prob)}')

# Mostrar un reporte detallado de clasificación
print(classification_report(y_test, y_pred_adjusted))

# Validación cruzada
cross_val_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Accuracy media con validación cruzada: {cross_val_scores.mean():.2f}")

# Función para predecir el sentimiento de una nueva reseña
def predecir_sentimiento(reseña):
    # Transformar la nueva reseña utilizando el vectorizador TF-IDF
    reseña_tfidf = tfidf.transform([reseña])

    # Realizar la predicción con el modelo entrenado
    prediccion = best_model.predict(reseña_tfidf)

    # Interpretar el resultado
    if prediccion[0] == 1:
        return "Positiva"
    else:
        return "Negativa"

# Pedir al usuario que ingrese una reseña
reseña_usuario = input("Ingresa tu reseña de producto para analizar el sentimiento: ")

# Predecir el sentimiento de la reseña ingresada
sentimiento = predecir_sentimiento(reseña_usuario)

# Mostrar el resultado
print(f"La reseña es clasificada como: {sentimiento}")
