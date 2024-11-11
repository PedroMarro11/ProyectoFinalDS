from flask import Flask, request, jsonify
import joblib

# Cargar el modelo guardado
model = joblib.load('regresion_logistica.pkl')

# Crear una instancia de Flask
app = Flask(__name__)

# Definir una ruta para las predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extraer las características necesarias para la predicción
    gdp = data.get('GDP')
    health = data.get('Health')
    freedom = data.get('Freedom')
    
    # Validar que todas las características están presentes
    if gdp is None or health is None or freedom is None:
        return jsonify({'error': 'Faltan datos para la predicción'}), 400
    
    # Realizar la predicción
    input_data = [[gdp, health, freedom]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Enviar la respuesta como JSON
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability)
    })

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
