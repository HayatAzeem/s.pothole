import cv2
import numpy as np
import supervision as sv
import logging
from config import MODEL_PREFIX

logger = logging.getLogger(__name__)

def process_frame(
    frame: np.ndarray, models: dict[str, tuple], thresholds: dict[str, float]
) -> tuple[np.ndarray, int]:
    annotated_frame = frame.copy()
    img_height, img_width = frame.shape[:2]
    img_area = img_width * img_height
    total_severity_raw = 0.0

    for model_name, (model, names_map, box_ann, label_ann) in models.items():
        try:
            results = model.predict(frame, conf=thresholds[model_name], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # --- SEVERITY CALCULATION ---
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                class_name = names_map.get(class_id, str(class_id)).lower()
                
                x1, y1, x2, y2 = detections.xyxy[i]
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area
                
                base_weight = 0
                if "pothole" in class_name:
                    base_weight = 8.0
                elif "crack-severe" in class_name:
                    base_weight = 6.0
                elif "crack" in class_name:
                    base_weight = 2.0
                
                total_severity_raw += base_weight * (1 + (area_ratio * 10))
            # ----------------------------

            labels = [
                f"{MODEL_PREFIX[model_name]}:{names_map.get(cls_id, str(cls_id))} {conf:.2f}"
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_ann.annotate(annotated_frame, detections)
            annotated_frame = label_ann.annotate(annotated_frame, detections, labels=labels)
        except Exception as e:
            logger.error(f"Error during prediction/annotation for {model_name}: {e}")
            cv2.putText(annotated_frame, f"Error processing {model_name}", (10, 30 + list(models.keys()).index(model_name) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- NORMALIZE TO 1-10 SCALE ---
    max_raw_score_threshold = 30.0 
    if total_severity_raw == 0:
        final_score = 1
    else:
        scaled = (total_severity_raw / max_raw_score_threshold) * 9 + 1
        final_score = min(10, max(1, round(scaled)))

    return annotated_frame, final_score
