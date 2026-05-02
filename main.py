import streamlit as st
import logging
import base64
from streamlit_local_storage import LocalStorage

from config import MODEL_PATHS, DEFAULT_CONF
from utils import load_yolo_model, make_annotators
from handlers import handle_image_input, handle_video_input, handle_live_camera
import supervision as sv

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "output_file_path" not in st.session_state:
    st.session_state.output_file_path = None
if "output_file_name" not in st.session_state:
    st.session_state.output_file_name = None

def main():
    st.set_page_config(layout="wide", page_title="Road Anomaly Detection")
    localS = LocalStorage()
    
    st.title("✨ Road Anomaly Detection with YOLOv8 🚗🚨")
    st.markdown("[Check On Github](https://github.com/HayatAzeem/s.pothole)")

    st.sidebar.header("⚙️ Configuration")
    st.sidebar.subheader("🧠 Models")
    use_m1 = st.sidebar.checkbox("M1 (RoadModel_yolov8m)", value=True, key="cb_m1")
    use_m2 = st.sidebar.checkbox("M2 (YOLOv8_Small_2nd_Model)", value=True, key="cb_m2")

    models_to_load = {}
    if use_m1:
        models_to_load["M1 (Model 1)"] = MODEL_PATHS["M1 (Model 1)"]
    if use_m2:
        models_to_load["M2 (Model 2)"] = MODEL_PATHS["M2 (Model 2)"]

    loaded_models = {}
    thresholds = {}
    model_load_failed = False

    if not models_to_load:
        st.sidebar.warning("Select at least one model checkbox to start.")
        st.info("👈 Please select models and configure input source in the sidebar.")
        st.stop()

    for name, path in models_to_load.items():
        model, names_map = load_yolo_model(path)
        if model and names_map:
            color = sv.Color.RED if name == "M1 (Model 1)" else sv.Color.BLUE
            box_ann, label_ann = make_annotators(color)
            loaded_models[name] = (model, names_map, box_ann, label_ann)
            thresholds[name] = st.sidebar.slider(
                f"{name} Confidence",
                0.1,
                1.0,
                DEFAULT_CONF[name],
                0.05,
                key=f"{name}_conf",
            )
        else:
            model_load_failed = True

    if model_load_failed:
        st.error("One or more models failed to load. Check logs and file paths.")
        st.stop()

    st.sidebar.subheader("🎬 Input Source")
    input_mode = st.sidebar.radio(
        "Select Input Type", ["Image", "Video", "Live Camera"], key="input_mode_radio"
    )

    if input_mode == "Image":
        image_placeholder = st.empty()
        handle_image_input(loaded_models, thresholds, image_placeholder, localS)
    elif input_mode == "Video":
        video_status_placeholder = st.empty()
        handle_video_input(loaded_models, thresholds, video_status_placeholder)
    elif input_mode == "Live Camera":
        handle_live_camera(loaded_models, thresholds)

    st.markdown("---")
    st.write("© 2025 Team 21")

    # Display recent images from local storage below copyright text
    st.subheader("History")
    
    history = localS.getItem("image_history")
    if history and isinstance(history, list) and len(history) > 0:
        clear_clicked = st.button("Clear History")
        if clear_clicked:
            localS.eraseItem("image_history")
            st.success("History cleared!")
        else:
            cols = st.columns(5)
            for i, item in enumerate(reversed(history)):
                col = cols[i % 5]
                try:
                    img_data = base64.b64decode(item["data"])
                    caption = f"{item['timestamp']} - Severity: {item.get('score', 'N/A')}/10"
                    col.image(img_data, caption=caption, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error decoding image from history: {e}")
    else:
        st.info("No history yet.")

if __name__ == "__main__":
    main()
