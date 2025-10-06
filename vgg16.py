import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
import cv2
import os

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the top layers of the base model for fine-tuning
for layer in base_model.layers[:-4]:
    layer.trainable = False  # Freeze all layers except the last 4

# Custom feature extraction for edge detection and black spots analysis
def preprocess_custom_features(image):
    # Ensure the image is in the correct 8-bit grayscale format
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert to 8-bit unsigned integer type (if not already in that format)
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    
    # Apply edge detection (Canny)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    
    # Analyze for black spots (thresholding dark regions)
    _, black_spots = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Combine original grayscale image, edges, and black spots into a 3-channel image
    custom_feature_image = np.stack([gray_image, edges, black_spots], axis=-1)
    
    return custom_feature_image

# Preprocess dataset with custom features
def preprocess_image(image):
    # Resize image to 224x224
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Extract custom features
    custom_features = preprocess_custom_features(image_rgb)
    
    # Convert to float32 and normalize to range [0, 1]
    custom_features = custom_features.astype(np.float32) / 255.0
    
    return custom_features

# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification: fresh or rotten

# Define the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Image data generators for loading images
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,  # Apply the custom pre-processing function
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\EESHA\Downloads\fruit freshness indicator\dataset-20250225T193303Z-001\dataset', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    r'C:\Users\EESHA\Downloads\fruit freshness indicator\dataset-20250225T193303Z-001\dataset', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# CSV Logger to save training history
csv_logger = CSVLogger('training_log.csv', append=False)

# Train the model and save the training history
history = model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[csv_logger])

# Save the model
model.save('enhanced_fruit_freshness_classifier.h5')

# Save training history manually
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Save predictions
predictions = model.predict(validation_generator)

# Round predictions to get binary classification results
predicted_classes = np.round(predictions).astype(int)

# Save predictions to a CSV file
np.savetxt('predictions.csv', predicted_classes, delimiter=',', fmt='%d')


