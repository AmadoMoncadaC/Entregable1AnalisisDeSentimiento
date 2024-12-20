# Importamos las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positiva, 0 = negativa
}

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(data)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Aplicar TF-IDF vectorización para convertir el texto en vectores
tfidf = TfidfVectorizer(stop_words='english')  # Se eliminan las palabras comunes como "the", "is", etc.
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