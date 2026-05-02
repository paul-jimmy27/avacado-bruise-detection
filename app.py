import streamlit as st
import numpy as np
import cv2
import spectral
import joblib
from collections import Counter
import tempfile
import zipfile
import os
import shutil

# -------------------------------
# CONFIG
# -------------------------------

TIME_MODEL_PATH  = r"./Models_time_newDataset1/xgb_time_classifier.pkl"
TIME_SCALER_PATH = r"./Models_time_newDataset1/xgb_time_scaler.pkl"
BASE_MODEL_DIR   = r"./NewDatasetModels1"

THRESH = 0.75
MORPH_KERNEL = 5
MIN_AREA = 4000

# -------------------------------
# SESSION STATE
# -------------------------------
if "done" not in st.session_state:
    st.session_state.done = False

if "final_image" not in st.session_state:
    st.session_state.final_image = None

if "majority_time" not in st.session_state:
    st.session_state.majority_time = None

# -------------------------------
# FINAL VIEW
# -------------------------------
if st.session_state.done:

    col1, col2 = st.columns([0.9, 0.1])

    with col1:
        st.write(f"🧠 Predicted Time Class: **T{st.session_state.majority_time}**")

    with col2:
        close = st.button("✖", key="close_btn")

    if close:
        st.session_state.done = False
        st.session_state.final_image = None
        st.session_state.majority_time = None
        st.rerun()

    img = st.session_state.final_image

    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)

    st.image(img)   

# -------------------------------
# UPLOAD + PROCESSING VIEW
# -------------------------------
else:

    st.title("🥑 Hyperspectral Bruise Detection")

    st.markdown("""
    #### 📁 Upload Requirements

    Please upload a **ZIP file** containing:

    - `.hdr` file (hyperspectral header)  
    - `.dat` / `.img` file  
    - RGB image (`.png` / `.jpg`)  

    ⚠️ HDR and RGB must match dimensions exactly
    """)

    zip_file = st.file_uploader("Upload ZIP file", type=["zip"])

    if zip_file:

        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # -------------------------------
        # FIND FILES
        # -------------------------------
        hdr_path = None
        data_path = None
        rgb_path = None

        for root, _, files in os.walk(tmpdir):
            for file in files:
                fpath = os.path.join(root, file)
                fname = file.lower()

                if fname.endswith(".hdr"):
                    hdr_path = fpath
                elif fname.endswith((".img", ".dat", ".raw")):
                    data_path = fpath
                elif fname.endswith((".png", ".jpg", ".jpeg")):
                    rgb_path = fpath

        if not hdr_path or not data_path or not rgb_path:
            st.error("Missing required files in ZIP")
            st.stop()
        st.success("✅ Files loaded successfully")
        # -------------------------------
        # LOAD HDR
        # -------------------------------
        img = spectral.open_image(hdr_path)
        cube = np.array(img.load(), dtype=np.float32)

        cube = np.rot90(cube, k=-1, axes=(0,1))

        H, W, B = cube.shape

        # -------------------------------
        # LOAD RGB
        # -------------------------------
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb_H, rgb_W = rgb.shape[:2]
        if (rgb_H, rgb_W) != (H, W):
            st.error("Dimension mismatch between HDR and RGB")
            st.stop()

        rgb_norm = rgb.astype(np.float32) / 255.0

        # -------------------------------
        # OTSU SEGMENTATION
        # -------------------------------
        gray = np.mean(cube, axis=2)
        gray = (gray - gray.min()) / (gray.max() - gray.min())
        gray_8 = (gray * 255).astype(np.uint8)

        _, mask = cv2.threshold(
            gray_8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        clean_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                continue

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if w/h > 3.0 or w/h < 0.3:
                continue

            clean_mask[labels == i] = 1

        mask = clean_mask

        # -------------------------------
        # FEATURES
        # -------------------------------
        coords = np.column_stack(np.where(mask == 1))
        spectra = cube[mask == 1]

        # -------------------------------
        # TIME MODEL
        # -------------------------------
        time_model  = joblib.load(TIME_MODEL_PATH)
        time_scaler = joblib.load(TIME_SCALER_PATH)

        spectra_time = time_scaler.transform(spectra)
        time_pred = time_model.predict(spectra_time)

        majority_time = Counter(time_pred).most_common(1)[0][0]
        st.session_state.majority_time = majority_time

        # -------------------------------
        # BRUISE MODEL
        # -------------------------------
        model_folder = os.path.join(
            BASE_MODEL_DIR,
            f"Models_xgb_t{majority_time}"
        )

        model  = joblib.load(os.path.join(model_folder, "xgb_model.pkl"))
        scaler = joblib.load(os.path.join(model_folder, "xgb_scaler.pkl"))

        spectra_scaled = scaler.transform(spectra)

        probs = model.predict_proba(spectra_scaled)[:, 1]

        prob_map = np.zeros((H, W), dtype=np.float32)

        for (y, x), p in zip(coords, probs):
            prob_map[y, x] = p

        pred_map = (prob_map > THRESH).astype(np.uint8)

        # -------------------------------
        # CLEAN BLOBS
        # -------------------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_map, 8)

        clean = np.zeros_like(pred_map)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 300:
                clean[labels == i] = 1

        pred_map = clean

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_CLOSE, kernel_big)
        pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_OPEN, kernel_small)

        pred_map = cv2.GaussianBlur(pred_map.astype(np.float32), (5,5), 0)
        pred_map = (pred_map > 0.3).astype(np.uint8)

        contours, _ = cv2.findContours(pred_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        smooth_map = np.zeros_like(pred_map)

        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            cv2.drawContours(smooth_map, [approx], -1, 1, -1)

        pred_map = smooth_map

        # -------------------------------
        # OVERLAY
        # -------------------------------
        overlay = rgb_norm.copy()
        overlay[pred_map == 1] = [1, 0, 0]

        alpha = 0.4
        final = (1 - alpha) * rgb_norm + alpha * overlay

        # -------------------------------
        # STORE + SWITCH VIEW
        # -------------------------------
        st.session_state.final_image = (final * 255).astype(np.uint8)
        st.session_state.done = True
        shutil.rmtree(tmpdir, ignore_errors=True)
        st.rerun()