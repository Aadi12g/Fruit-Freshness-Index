import cv2
import numpy as np
import tensorflow as tf
from preprocess import clahe_enhancement

# Loading freshness classifier model
classifier_model = tf.keras.models.load_model('enhanced_fruit_freshness_classifier.h5')

def classify_freshness(roi):
    """Classify fruit freshness with preprocessing"""
    try:
        # Convert to RGB and normalize lighting
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_normalized = clahe_enhancement(roi_rgb)
        
        # Resize and preprocess
        roi_resized = cv2.resize(roi_normalized, (224, 224))
        roi_normalized = roi_resized / 255.0
        roi_normalized = roi_normalized.reshape(1, 224, 224, 3)
        
        # Predict
        predictions = classifier_model.predict(roi_normalized)
        return predictions[0][0] * 100  # Freshness index
    except Exception as e:
        print(f"Classification error: {e}")
        return 0  # Default freshness if error occurs


