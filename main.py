import cv2
import numpy as np
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from detection import detect_fruits
from classifier import classify_freshness
from camera import get_webcam_feed
from utils import draw_label, draw_freshness_index
from preprocess import adjust_gamma, clahe_enhancement


def adjust_lighting(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    gamma = 0.5 if brightness < 50 else 1.5 if brightness > 200 else 1.0
    frame_normalized = adjust_gamma(frame_rgb, gamma=gamma)
    frame_enhanced = clahe_enhancement(frame_normalized)
    return frame_enhanced


# Appending new metrics to CSV
def save_metrics_to_csv(metrics_list):
    file_path = 'metrics.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Precision', 'Recall', 'F1 Score', 'Accuracy'])
        writer.writerow(metrics_list)


def main():
    y_true = []
    y_pred = []

    for frame in get_webcam_feed():
        frame_processed = adjust_lighting(frame)
        boxes, scores, classes = detect_fruits(frame_processed)

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                h, w, _ = frame.shape
                box = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
                startX, startY, endX, endY = box

                roi = frame_processed[startY:endY, startX:endX]
                freshness_index = classify_freshness(roi)

                # Simulated ground truth for testing
                true_label = 1 if freshness_index > 50 else 0
                pred_label = 1 if freshness_index > 50 else 0

                y_true.append(true_label)
                y_pred.append(pred_label)

                color = (0, 255, 0) if freshness_index > 50 else (0, 0, 255)
                draw_label(frame, "", box, color)
                draw_freshness_index(frame, f"Fresh: {freshness_index:.1f}%", box, color)

        resized_frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Fruit Freshness Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if y_true and y_pred:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        r1 = recall  

        save_metrics_to_csv([precision, recall, f1, accuracy, r1])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()








