import os
import cv2
import base64
import logging
import streamlit as st
from ultralytics import YOLO
import supervision as sv

logger = logging.getLogger(__name__)

@st.cache_resource
def load_yolo_model(path: str):
    try:
        model = YOLO(path)
        logger.info(f"Successfully loaded model from {path}")
        return model, model.names
    except Exception as e:
        st.error(f"Error loading model at {path}: {e}")
        logger.error(f"Failed to load model at {path}", exc_info=e)
        return None, {}

def make_annotators(color: sv.Color):
    box_annotator = sv.BoxAnnotator(thickness=1, color=color)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.4,
        color=sv.Color.WHITE,
        text_color=sv.Color.BLACK,
        text_padding=2,
    )
    return box_annotator, label_annotator

def encode_image_to_base64(img_array, max_width=640, quality=80):
    """Resizes and compresses an image, then converts to base64."""
    h, w = img_array.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img_array = cv2.resize(img_array, (max_width, new_h))
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

def cleanup_previous_output():
    """Deletes the previously generated output file if it exists."""
    if st.session_state.output_file_path and os.path.exists(
        st.session_state.output_file_path
    ):
        try:
            os.remove(st.session_state.output_file_path)
            logger.info(
                f"Cleaned up previous output file: {st.session_state.output_file_path}"
            )
        except OSError as rm_err:
            logger.error(
                f"Error removing previous output file {st.session_state.output_file_path}: {rm_err}"
            )
    st.session_state.output_file_path = None
    st.session_state.output_file_name = None
    st.session_state.processing_complete = False
    st.session_state.processed_file_id = None
