from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

modelo = joblib.load('modelo_edutrack.pkl')
columnas_modelo = joblib.load('columnas_edutrack.pkl')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json
        # DataFrame inicializado en 0 con la estructura exacta del Colab
        input_data = pd.DataFrame(0, index=[0], columns=columnas_modelo)

        # Cálculos y asignaciones
        est_por_docente = datos['matricula'] / datos['docentes'] if datos['docentes'] > 0 else datos['matricula']

        input_data.at[0, 'MATRICULA'] = datos['matricula']
        input_data.at[0, 'N_DOCENTES'] = datos['docentes']
        input_data.at[0, 'N_ORIENTADORES'] = datos['orientadores']
        input_data.at[0, 'EST_POR_DOCENTE'] = est_por_docente
        input_data.at[0, 'N_NIVELES'] = datos['niveles']
        input_data.at[0, 'N_JORNADAS'] = datos['jornadas']
        input_data.at[0, 'ZONA_RURAL'] = 1 if datos['zona'] == 'rural' else 0
        input_data.at[0, 'BILINGUE_SI'] = 1 if datos['bilingue'] == 'si' else 0
        input_data.at[0, 'ETNOEDUC_SI'] = 1 if datos['etno'] == 'si' else 0

        probabilidad = modelo.predict_proba(input_data)[0, 1]
        return jsonify({'probabilidad': float(probabilidad)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)