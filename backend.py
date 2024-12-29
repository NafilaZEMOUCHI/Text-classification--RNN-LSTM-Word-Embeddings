import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PyPDF2 import PdfReader
from flask_cors import CORS
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import docx
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg pour ne pas ouvrir de fenêtre graphique
from flask import send_from_directory
import matplotlib.pyplot as plt  # Assurez-vous d'ajouter cette ligne



# Dictionnaire des classes
CLASS_LABELS = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sports",
    4: "technologie"
}

# Charger le modèle RNN
model = load_model('text_classifier_model.keras')

# Charger le tokenizer à partir du fichier pickle
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Définir les constantes
MAX_SEQUENCE_LENGTH = 1000  # Longueur maximale des séquences
VALIDATION_SPLIT = 0.2  # Fraction de validation pour diviser les données

app = Flask(__name__)
CORS(app)  

# Dossier où les fichiers seront stockés
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assurez-vous que le dossier existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Fonction pour extraire le texte d'un fichier DOCX
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Fonction pour nettoyer et préparer le texte
def clean_text(input_text):
    # Convertir le texte en minuscules
    input_text = input_text.lower()
    
    # Supprimer les balises HTML
    input_text = re.sub(r'<.*?>', '', input_text)
    
    # Supprimer les chiffres
    input_text = re.sub(r'\d+', '', input_text)
    
    # Supprimer la ponctuation
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les espaces supplémentaires
    input_text = re.sub(r'\s+', ' ', input_text).strip()
    
    return input_text

# Fonction pour lemmatiser et supprimer les stopwords
def process_text(input_text):
    # Initialiser lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Charger les stopwords de NLTK
    stop_words = set(stopwords.words('english'))
    
    # Tokenisation du texte
    words = input_text.split()
    
    # Filtrer les stopwords et appliquer la lemmatisation
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(processed_words)

# Fonction pour préparer le texte pour la prédiction
def prepare_input_text(input_text, tokenizer, max_sequence_length=1000):
    # Nettoyer le texte
    cleaned_text = clean_text(input_text)
    
    # Traiter le texte (lemmatisation et suppression des stopwords)
    processed_text = process_text(cleaned_text)
    
    # Tokenisation et padding
    sequences = tokenizer.texts_to_sequences([processed_text])
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return data


@app.route('/')
def home():
    return "API Flask de Classification de Textes est en cours d'exécution !"




@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text')

        if text:
            prepared_text = prepare_input_text(text, tokenizer)
            prediction = model.predict(prepared_text)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_probabilities = prediction[0].astype(float)
            
            predicted_category = CLASS_LABELS.get(predicted_class, "Inconnu")
            class_probabilities = {CLASS_LABELS.get(i, f"Classe {i}"): round(prob * 100, 2) 
                                   for i, prob in enumerate(predicted_probabilities)}
            
            # Créer l'histogramme
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(class_probabilities.keys(), class_probabilities.values(), color='skyblue')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probabilité (%)')
            ax.set_title('Probabilités des différentes classes')

            # Sauvegarder l'image dans un fichier dans le dossier static
            img_path = 'static/histogram.png'
            fig.savefig(img_path)
            plt.close(fig)

            return jsonify({
                'predicted_class': predicted_category,
                'class_probabilities': class_probabilities,
                'histogram_image_url': f'/static/histogram.png'  # URL relative
            })
        else:
            return jsonify({'error': 'Aucun texte fourni'}), 400
    except Exception as e:
        print(f"Erreur interne : {e}")
        return jsonify({'error': 'Erreur interne serveur'}), 500

     
  


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extraire le texte du fichier
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Format de fichier non pris en charge'}), 400

        # Classifier le texte extrait
        prepared_text = prepare_input_text(text, tokenizer)
        prediction = model.predict(prepared_text)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({'predicted_class': int(predicted_class)})  # Convert to Python int
    else:
        return jsonify({'error': 'Aucun fichier reçu'}), 400


@app.route('/classify-and-download', methods=['POST'])
def classify_and_download():
    file = request.files.get('file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extraire le texte du fichier
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            lines = text.splitlines()  # Diviser en lignes
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
            lines = text.splitlines()  # Diviser en lignes
        elif file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # Lire chaque ligne du fichier
        else:
            return jsonify({'error': 'Format de fichier non pris en charge'}), 400

        # Vérifier si le fichier contient du texte
        if not lines:
            return jsonify({'error': 'Fichier vide ou non valide'}), 400

        # Classifier chaque ligne
        csv_output = 'Texte,Classe Prédite\n'
        for line in lines:
            line = line.strip()  # Supprimer les espaces inutiles
            if line:  # Ignorer les lignes vides
                # Nettoyer le texte avant classification
                cleaned_line = clean_text(line)
                processed_line = process_text(cleaned_line)

                # Préparer le texte pour la prédiction
                prepared_text = prepare_input_text(processed_line, tokenizer)

                # Faire la prédiction
                prediction = model.predict(prepared_text)
                predicted_class = np.argmax(prediction, axis=1)[0]
                class_label = CLASS_LABELS.get(predicted_class, "Inconnu")

                # Ajouter la ligne nettoyée et la classe prédite au CSV
                csv_output += f'"{processed_line}","{class_label}"\n'

        # Supprimer le fichier téléchargé après traitement
        os.remove(file_path)

        # Créer une réponse CSV
        response = app.response_class(
            response=csv_output,
            status=200,
            mimetype='text/csv'
        )
        response.headers["Content-Disposition"] = "attachment; filename=resultats_classement.csv"
        return response
    else:
        return jsonify({'error': 'Aucun fichier reçu'}), 400



@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
