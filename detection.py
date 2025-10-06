import tensorflow as tf
import numpy as np

# Loading the SSD MobileNet model
ssd_model = tf.saved_model.load(r'C:\Users\EESHA\Downloads\fruit freshness indicator\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model')

def detect_fruits(frame):
    """Detecting fruits in the frame using SSD MobileNet."""
    # Preprocessing the frame for SSD MobileNet input
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]  
    
    # Running object detection
    detections = ssd_model(input_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    scores = detections['detection_scores'][0].numpy()  # Confidence scores
    classes = detections['detection_classes'][0].numpy().astype(int)  # Class IDs
    
    return boxes, scores, classes





