import os
import cv2
import tempfile
import numpy as np
import logging
import streamlit as st
import av
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from inference import process_frame
from utils import encode_image_to_base64, cleanup_previous_output
from config import LIVE_FEED_TARGET_WIDTH

logger = logging.getLogger(__name__)

def handle_image_input(models, thresholds, placeholder, localS):
    # Reset video processing state if switching to image mode
    if st.session_state.processed_file_id is not None:
        cleanup_previous_output()

    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_upload"
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not decode image. Please upload a valid image file.")
            placeholder.empty()
        else:
            with st.spinner("Processing image..."):
                processed_img, score = process_frame(img, models, thresholds)
            
            placeholder.image(processed_img, channels="BGR", use_container_width=True)
            
            # Display the severity score outside the image
            st.markdown(f"### 🚨 Damage Severity Score: **{score}/10**")
            
            # Save to local storage
            try:
                b64_img = encode_image_to_base64(processed_img)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history = localS.getItem("image_history")
                if not history or not isinstance(history, list):
                    history = []
                
                # Check if this exact image is already the last one (avoid duplicates on rerun)
                if not history or history[-1].get("data") != b64_img:
                    history.append({
                        "timestamp": timestamp,
                        "data": b64_img,
                        "score": score
                    })
                    # Keep only last 10 images
                    if len(history) > 10:
                        history = history[-10:]
                    localS.setItem("image_history", history)
            except Exception as e:
                logger.error(f"Error saving image to local storage: {e}")
            
            st.success("Image processing complete.")
    else:
        placeholder.info("Upload an image using the sidebar to start.")


def handle_video_input(models, thresholds, status_placeholder):
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
    )

    if uploaded_file:
        current_file_id = uploaded_file.file_id

        # Check if it's a new file upload
        if current_file_id != st.session_state.processed_file_id:
            logger.info(
                f"New video file uploaded (ID: {current_file_id}). Resetting state."
            )
            cleanup_previous_output()  # Clean up old output file before processing new one
            st.session_state.processed_file_id = current_file_id  # Set the new file ID

        # If processing is already complete for this file, just show download button
        if st.session_state.processing_complete and st.session_state.output_file_path:
            status_placeholder.success("✅ Video processing complete!")
            if os.path.exists(st.session_state.output_file_path):
                try:
                    with open(st.session_state.output_file_path, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name=st.session_state.output_file_name
                        or f"processed_{uploaded_file.name}",
                        mime="video/mp4",
                        key="download_btn_rerun",  # Added key for consistency
                    )
                    logger.info(
                        f"Download button shown again for already processed file: {st.session_state.output_file_path}"
                    )
                except Exception as e:
                    st.error(
                        f"Error reading previously processed video for download: {e}"
                    )
                    logger.error(
                        f"Error reading existing output file {st.session_state.output_file_path} for download",
                        exc_info=e,
                    )
            else:
                st.error("Previously processed file not found. Please upload again.")
                logger.warning(
                    f"Session state indicated processed file {st.session_state.output_file_path} but it was not found."
                )
                cleanup_previous_output()  # Reset state as the file is missing
            return  # Stop further execution in this function call

        # --- Start Processing for a new file or if not yet complete ---
        input_tmp_path = None
        output_video_path_current_run = None  # Use a temporary variable for this run
        cap = None
        writer = None
        processing_succeeded = False  # Flag to track success within try block

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as input_tmp_file:
                input_tmp_path = input_tmp_file.name
                input_tmp_file.write(uploaded_file.read())
            logger.info(f"Input video saved to temporary file: {input_tmp_path}")

            cap = cv2.VideoCapture(input_tmp_path)
            if not cap.isOpened():
                st.error("Error opening uploaded video file.")
                status_placeholder.empty()
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 30
                logger.warning("Could not read video FPS, defaulting to 30.")

            # Create a *new* temporary file for the output of this run
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as output_tmp_file:
                output_video_path_current_run = output_tmp_file.name
            logger.info(
                f"Output video for this run will be saved to: {output_video_path_current_run}"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_video_path_current_run, fourcc, fps, (width, height)
            )
            if not writer.isOpened():
                st.error(f"Error initializing video writer.")
                logger.error(
                    f"Failed to open VideoWriter for path: {output_video_path_current_run}"
                )
                if output_video_path_current_run and os.path.exists(
                    output_video_path_current_run
                ):
                    os.remove(
                        output_video_path_current_run
                    )  # Clean up failed output file
                output_video_path_current_run = None
                return

            prog_bar = st.progress(0, text="Processing video...")
            status_placeholder.info("Processing video, please wait...")
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, _ = process_frame(frame, models, thresholds)
                writer.write(out_frame)
                frame_idx += 1
                progress_percentage = (
                    frame_idx / total_frames if total_frames > 0 else 0
                )
                prog_text = f"Processing video... {frame_idx}/{total_frames if total_frames > 0 else '?'}"
                prog_bar.progress(min(progress_percentage, 1.0), text=prog_text)

            processing_succeeded = True  # Mark success only if loop completes
            prog_bar.progress(1.0, text="Processing complete.")
            logger.info("Video processing finished.")

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
            logger.error("Error during video processing loop", exc_info=e)
            status_placeholder.error("Processing failed.")
            if "prog_bar" in locals():
                prog_bar.empty()  # Ensure progress bar removed on error

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            logger.info("Video capture and writer resources released.")
            if input_tmp_path and os.path.exists(input_tmp_path):
                try:
                    os.remove(input_tmp_path)
                    logger.info(f"Removed input temp file: {input_tmp_path}")
                except OSError as rm_err:
                    logger.error(
                        f"Error removing input temp file {input_tmp_path}: {rm_err}"
                    )

        # --- Post-processing logic ---
        if (
            processing_succeeded
            and output_video_path_current_run
            and os.path.exists(output_video_path_current_run)
        ):
            # Store path and status in session state
            st.session_state.output_file_path = output_video_path_current_run
            st.session_state.output_file_name = f"processed_{uploaded_file.name}"
            st.session_state.processing_complete = True
            st.session_state.processed_file_id = (
                current_file_id  # Ensure ID is set on success
            )

            status_placeholder.success("✅ Video processing complete!")
            # Now display the download button for the first time
            try:
                with open(st.session_state.output_file_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name=st.session_state.output_file_name,
                    mime="video/mp4",
                    key="download_btn_first",  # Different key maybe? Helps debugging
                )
                logger.info(
                    f"Download button provided for newly processed file: {st.session_state.output_file_path}"
                )
            except Exception as e:
                st.error(f"Error reading processed video for download: {e}")
                logger.error(
                    f"Error reading output file {st.session_state.output_file_path} for download",
                    exc_info=e,
                )
                cleanup_previous_output()  # Reset state if download prep fails

        elif output_video_path_current_run and os.path.exists(
            output_video_path_current_run
        ):
            # Processing failed, clean up the output file created during this failed run
            logger.warning(
                f"Processing failed, cleaning up temporary output file: {output_video_path_current_run}"
            )
            try:
                os.remove(output_video_path_current_run)
            except OSError as rm_err:
                logger.error(
                    f"Error removing failed output temp file {output_video_path_current_run}: {rm_err}"
                )
            # Ensure session state is cleared if processing failed after a new file upload began
            if current_file_id == st.session_state.processed_file_id:
                cleanup_previous_output()

        if "prog_bar" in locals():
            prog_bar.empty()  # Final removal of progress bar

    else:
        # No file uploaded, ensure any previous state is cleared
        if st.session_state.processed_file_id is not None:
            cleanup_previous_output()
        status_placeholder.info("Upload a video using the sidebar to start.")


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, models, thresholds, target_width):
        self.models = models
        self.thresholds = thresholds
        self.target_width = target_width
        logger.info(
            f"YOLOVideoProcessor initialized. Target processing width: {self.target_width}"
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        original_height, original_width = img.shape[:2]
        img_resized = img
        if self.target_width is not None and original_width > self.target_width:
            aspect_ratio = original_height / original_width
            target_height = int(self.target_width * aspect_ratio)
            img_resized = cv2.resize(
                img, (self.target_width, target_height), interpolation=cv2.INTER_AREA
            )
        annotated_frame_resized, _ = process_frame(
            img_resized, self.models, self.thresholds
        )
        if img_resized is not img:
            final_frame = cv2.resize(
                annotated_frame_resized,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            final_frame = annotated_frame_resized
        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")


def handle_live_camera(models, thresholds):
    # Reset video processing state if switching to live mode
    if st.session_state.processed_file_id is not None:
        cleanup_previous_output()

    st.sidebar.info(
        "Click 'START' below to access your camera. "
        "Ensure camera permissions are granted in your browser."
    )
    media_constraints = {
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 30},
        },
        "audio": False,
    }

    def processor_factory():
        return YOLOVideoProcessor(
            models=models, thresholds=thresholds, target_width=LIVE_FEED_TARGET_WIDTH
        )

    webrtc_ctx = webrtc_streamer(
        key="live-camera-streamer",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=processor_factory,
        media_stream_constraints=media_constraints,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    if not webrtc_ctx.state.playing:
        st.info("Camera feed stopped or not started.")
