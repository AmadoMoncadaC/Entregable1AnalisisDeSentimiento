import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Lista de stopwords en español (puedes ampliarla si es necesario)
stopwords_espanol = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'para', 'con', 'una',
    'su', 'no', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'si', 'me', 'es', 'porque',
    'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'también', 'otros', 'fue', 'ha', 'está', 'yo',
    'hasta', 'hay', 'donde', 'quien', 'después', 'te', 'ni', 'nos', 'durante', 'todos', 'algunos', 'este', 'él',
    'ellas', 'ante', 'ese', 'esto', 'esa', 'esos', 'esas', 'unos', 'una', 'su', 'de', 'todo', 'mismo', 'ya'
]

# Crear un ejemplo de datos (reseñas de productos)
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
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  # 1 = positiva, 0 = negativa
}

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(data)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Agregar algunas palabras clave negativas específicas al vectorizador
stopwords_adicionales = ['pesimo', 'malo', 'horrible', 'defectuoso', 'decepcionado']

# Crear una lista de stopwords que incluye las palabras en español y las adicionales
stop_words_completo = stopwords_espanol + stopwords_adicionales  # Concatenar las stopwords

# Aplicar TF-IDF vectorización para convertir el texto en vectores
tfidf = TfidfVectorizer(stop_words=stop_words_completo, lowercase=True)  # Añadir las palabras clave negativas y las stopwords en español
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test_tfidf)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Mostrar un reporte detallado de clasificación
print(classification_report(y_test, y_pred))

# Función para predecir el sentimiento de una nueva reseña
def predecir_sentimiento(reseña):
    # Transformar la nueva reseña utilizando el vectorizador TF-IDF
    reseña_tfidf = tfidf.transform([reseña])
    
    # Realizar la predicción con el modelo entrenado
    prediccion = model.predict(reseña_tfidf)
    
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

