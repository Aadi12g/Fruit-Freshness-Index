import os
import tensorflow as tf
import cv2
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the detection model
model_dir = r'C:\Users\EESHA\Downloads\maybe fruit\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
detection_model = tf.saved_model.load(str(model_dir))

# COCO class labels - we'll focus only on fruit classes
FRUIT_CLASSES = {
    52: 'apple', 53: 'orange', 54: 'banana', 55: 'broccoli', 56: 'carrot',
    # Add more fruit classes as needed
}

def run_inference(model, image):
    """Run object detection on the input image."""
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def normalize_brightness(image):
    """Normalize image brightness using CLAHE and gamma correction."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L channel back with a and b channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to RGB color space
    normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Additional gamma correction
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    gamma = 1.0
    
    if brightness < 50:    # Dark image
        gamma = 0.7
    elif brightness > 200: # Bright image
        gamma = 1.3
    
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        normalized = cv2.LUT(normalized, table)
    
    return normalized

def save_detections(image, detections, output_path):
    """Process and save detections focusing only on fruits."""
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    height, width, _ = image.shape
    detected = False

    # Fixed threshold for fruits (can be adjusted)
    threshold = 0.5
    box_color = (0, 255, 0)  # Green for fruits

    for i in range(len(scores)):
        # Only process if it's a fruit class and score is above threshold
        if scores[i] > threshold and classes[i] in FRUIT_CLASSES:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw bounding box
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), box_color, 2)
            
            # Add label with class name and confidence
            label = f"{FRUIT_CLASSES[classes[i]]}: {scores[i]:.2f}"
            cv2.putText(image, label, (int(left), int(top) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            detected = True

    if detected:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved detected image: {output_path}")
    else:
        print(f"No fruits detected in: {output_path}")

# Process all images in the dataset folder
dataset_path = r'C:\Users\EESHA\Downloads\maybe fruit\dataset-20250225T193303Z-001'
output_dir = r'C:\Users\EESHA\Downloads\maybe fruit\output_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_path = os.path.join(root, file)
            output_image_path = os.path.join(output_dir, file)
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue
                
                # Convert to RGB and normalize brightness
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = normalize_brightness(image_rgb)
                
                # Run detection and save results
                detections = run_inference(detection_model, image_rgb)
                save_detections(image_rgb, detections, output_image_path)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
