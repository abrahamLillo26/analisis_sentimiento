from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# Carga modelo
classifier3 = pipeline("sentiment-analysis", model="citizenlab/twitter-xlm-roberta-base-sentiment-finetunned")
# Carga tokenizador
tokenizer = AutoTokenizer.from_pretrained("citizenlab/twitter-xlm-roberta-base-sentiment-finetunned")

# endpoint
@app.route('/analisis-sentimiento', methods=['POST'])
def analizar_sentimiento():
    # Obtener los datos JSON de la solicitud
    data = request.get_json()
    transcripciones = data.get('transcripciones')

    resultados = []
    for transcripcion in transcripciones:
        # Tokenizar transcripción
        tokens = tokenizer.tokenize(transcripcion)
        # número de tokens
        num_tokens = len(tokens)

        # Eliminación de tokens
        tokens_minimo = 500
        tokens_eliminar = num_tokens - tokens_minimo
        transcripcion_lista = transcripcion.split()
        nueva_lista = transcripcion_lista[0:tokens_eliminar]
        indice = len(nueva_lista)
        texto_final = " ".join(transcripcion_lista[indice:])

        # predicción de sentimiento
        resultado_modelo3 = classifier3(texto_final)
        resultados.append({'transcripcion': transcripcion, 'resultado': resultado_modelo3})

    return jsonify({'resultados': resultados})

if __name__ == '__main__':
    # Ejecuta aplicación Flask en puerto 5000
    #app.run(debug=True, port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000)