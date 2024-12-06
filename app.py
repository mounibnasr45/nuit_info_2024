from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__, static_folder='static')

# Charger le modèle entraîné
model_path = r"C:\Users\mouni\OneDrive\Bureau\nuitinfo_24\chatbot\results\checkpoint-990"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Définir le token de padding
tokenizer.pad_token = tokenizer.eos_token

# Créer une fonction pour générer des réponses
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'No message received'}), 400
    response = generate_response(user_input)
    return jsonify({'response': response})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
