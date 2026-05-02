# config.py

# --- Configuration ---
MODEL_PATHS = {
    "M1 (Model 1)": "./RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt",
    "M2 (Model 2)": "./YOLOv8_Small_2nd_Model.pt",
}

MODEL_PREFIX = {
    "M1 (Model 1)": "M1",
    "M2 (Model 2)": "M2",
}

DEFAULT_CONF = {"M1 (Model 1)": 0.35, "M2 (Model 2)": 0.40}

LIVE_FEED_TARGET_WIDTH = 640
