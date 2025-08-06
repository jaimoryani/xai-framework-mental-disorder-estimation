import os
import numpy as np
import pandas as pd
import librosa
import torch
import nltk
import shap
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)

# Load pre-trained models
depressed_model = tf.keras.models.load_model('multimodal_model_depressed.h5')
ptsd_model = tf.keras.models.load_model('multimodal_model_ptsd.h5')

AUDIO_DIM = 193
TEXT_DIM = 768

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Text Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(cleaned_tokens)


# Audio Preprocessing
def remove_silence(y, sr):
    yt, _ = librosa.effects.trim(y)
    return yt

def segment_audio(y, sr, segment_length=50):
    segment_samples = int(segment_length * sr)
    return [y[i:i + segment_samples] for i in range(0, len(y), segment_samples)]

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    y = remove_silence(y, sr)
    segments = segment_audio(y, sr)

    all_features = []

    for segment in segments:
        res = np.array([])

        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T, axis=0)
        res = np.hstack((res, mfccs))
        delta_mfcc = librosa.feature.delta(mfccs)
        res = np.hstack((res, delta_mfcc))
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        res = np.hstack((res, delta2_mfcc))

        stft = np.abs(librosa.stft(segment))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        res = np.hstack((res, chroma))

        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr).T, axis=0)
        res = np.hstack((res, mel_spectrogram))

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        res = np.hstack((res, contrast))

        tonnetz = np.mean(librosa.feature.tonnetz(y=segment, sr=sr).T, axis=0)
        res = np.hstack((res, tonnetz))

        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        res = np.hstack((res, [pitch]))

        all_features.append(res)

    return all_features


# Text Feature Extraction using BERT
def extract_text_features(transcript_file):
    transcript_df = pd.read_csv(transcript_file)
    participant_lines = transcript_df['Text']
    all_text = " ".join(participant_lines)
    cleaned_text = preprocess_text(all_text)

    inputs = tokenizer(cleaned_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding


def extract_features(audio_path, transcript_path):
    audio_features = extract_audio_features(audio_path)
    text_features = extract_text_features(transcript_path)
    return audio_features, text_features


# Model Prediction + SHAP Explainability
def predict_with_xai(model, audio_features, text_features):
    all_predictions = []
    explanations = []

    # Prepare SHAP explainer using KernelExplainer (for non-tree/non-linear models)
    def model_predict(input_combined):
        text_input = input_combined[:, :TEXT_DIM]
        audio_input = input_combined[:, TEXT_DIM:]
        return model.predict([text_input, audio_input])

    background = np.hstack([text_features.reshape(1, -1), audio_features[0].reshape(1, -1)])
    explainer = shap.KernelExplainer(model_predict, background)

    for row in audio_features:
        audio_input = row.reshape(1, -1)
        text_input = text_features.reshape(1, -1)
        input_combined = np.hstack([text_input, audio_input])

        prediction = model.predict([text_input, audio_input])[0][0]
        shap_values = explainer.shap_values(input_combined)

        all_predictions.append(prediction)
        explanations.append(shap_values.tolist())

    final_pred = np.mean(all_predictions)
    explanation_mean = np.mean(np.array(explanations), axis=0).tolist()

    return ("Positive" if final_pred > 0.5 else "Negative", float(final_pred), explanation_mean)


# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'transcript' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Please upload both transcript and audio files"}), 400

    transcript_file = request.files['transcript']
    audio_file = request.files['audio']

    transcript_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts')
    audio_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'audio')

    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    transcript_path = os.path.join(transcript_dir, secure_filename(transcript_file.filename))
    audio_path = os.path.join(audio_dir, secure_filename(audio_file.filename))

    transcript_file.save(transcript_path)
    audio_file.save(audio_path)

    audio_features, text_features = extract_features(audio_path, transcript_path)

    selected_models = request.form.getlist('models')
    results = {}

    if 'depressed' in selected_models:
        label, prob, explanation = predict_with_xai(depressed_model, audio_features, text_features)
        results['Depression'] = {"label": label, "probability": prob, "explanation": explanation}

    if 'ptsd' in selected_models:
        label, prob, explanation = predict_with_xai(ptsd_model, audio_features, text_features)
        results['PTSD'] = {"label": label, "probability": prob, "explanation": explanation}

    os.remove(transcript_path)
    os.remove(audio_path)

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
