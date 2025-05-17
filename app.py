from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
import joblib
import logging
from save_model import extract_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Class labels in the same order as training
class_labels = ['circle', 'square', 'rectangle', 'triangle']

# Load the model
try:
    logger.info("Loading model...")
    model = joblib.load('model.joblib')
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    """Preprocess image for model prediction."""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if not already
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # If RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize to expected size
        img_array = cv2.resize(img_array, (64, 64))
        
        # Extract features using the same function as training
        features = extract_features(img_array)
        
        return features
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.get_json()['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        features = preprocess_image(image)
        if features is None:
            return jsonify({'error': 'Failed to preprocess image'})
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = model.predict_proba(features)[0]
        
        # Log prediction details
        logger.info(f"Prediction array: {prediction}")
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        logger.info(f"Predicted class: {predicted_class} with confidence: {confidence}")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 