# utils.py

import cv2

def draw_label(frame, label, box, color):
    """Draw bounding box and label on the frame."""
    startX, startY, endX, endY = box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    if label:
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_freshness_index(frame, freshness_text, box, color):
    """Draw the freshness index on the frame."""
    startX, startY, endX, endY = box
    # Position the text below the bounding box
    text_position = (startX, endY + 20)
    cv2.putText(frame, freshness_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
