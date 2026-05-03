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
import gc

# -------------------------------
# CONFIG
# -------------------------------

TIME_MODEL_PATH  = r"./Models_time_newDataset1/xgb_time_classifier.pkl"
TIME_SCALER_PATH = r"./Models_time_newDataset1/xgb_time_scaler.pkl"
BASE_MODEL_DIR   = r"./NewDatasetModels1"

THRESH = 0.75
MORPH_KERNEL = 5
MIN_AREA = 4000

time_descriptions = {
    -1:"""
At 0 minutes, the avocado is completely fresh with no signs of bruising or internal damage. The flesh is firm, smooth, and uniformly green in color. There is no cellular breakdown, so both texture and taste are at their best. It is suitable for export-quality markets and long storage. This stage represents the highest possible quality of the fruit.""",
    0: """
At 0 minutes immediately after bruising, the avocado may still look normal externally, but internal cell damage has already occurred. The impact ruptures cells and initiates enzymatic browning. No visible discoloration may appear yet, but structural weakening has begun. The fruit has already dropped from premium grade quality. Early handling is important to slow further deterioration.""",

    1: """
At 30 minutes, slight internal softening begins, though the outer skin may still appear unaffected. Enzymatic reactions increase, initiating minor browning beneath the surface. The affected area becomes less firm compared to the rest of the fruit. It remains suitable for fresh consumption and local markets. However, it is no longer ideal for export quality.""",

    2: """
At 1 hour, the bruised region begins to show faint brown discoloration internally. The flesh softens further, indicating progressing damage. Texture uniformity decreases, and the quality starts declining more noticeably. It is still safe to eat but should be consumed soon. Market value continues to reduce.""",

    3: """
At 3 hours, visible browning develops in the bruised area, making the damage more apparent. The texture becomes uneven with soft patches forming. Oxidation intensifies, slightly affecting taste and appearance. It is less suitable for fresh sale but acceptable for immediate consumption. Processing becomes a better option at this stage.""",

    4: """
At 6 hours, moderate bruising is clearly visible with darker brown regions inside the avocado. The flesh becomes softer and begins losing its original structure. Flavor may start to degrade due to continued oxidation. It is not suitable for premium or fresh markets. Immediate use for processing is recommended.""",

    5: """
At 12 hours, the avocado shows significant bruising with extensive internal browning. The texture turns noticeably mushy, especially around the damaged area. Flavor quality declines, and the fruit becomes less appealing for direct consumption. It is generally unsuitable for fresh sale. Limited processing use may still be possible if not spoiled.""",

    6: """
At 24 hours, severe bruising leads to widespread browning and breakdown of the flesh. The avocado loses firmness completely and becomes very soft. Taste and appearance deteriorate significantly. It is not suitable for fresh consumption or sale. Only minimal processing use may be possible if still safe.""",

    7: """
At 48 hours, the avocado is highly deteriorated with advanced internal breakdown. The flesh becomes very soft, dark, and may produce an unpleasant odor. Microbial activity may increase, making it unsafe for consumption. Nutritional and commercial value are lost. The fruit should be discarded to avoid health risks."""
}

# -------------------------------
# HELPER — force close mmap handles and delete folder
# spectral uses numpy memmap internally on Windows which
# keeps an OS-level file handle open even after del.
# We need to find and close all mmap objects before rmtree.
# -------------------------------

def force_delete(path):
    """
    Close any open mmap handles then delete the directory.
    Falls back to Windows 'rd' command if shutil.rmtree fails.
    """
    # Step 1: close all open mmap objects in current process
    for obj in gc.get_objects():
        try:
            if isinstance(obj, np.memmap):
                obj._mmap.close()
        except Exception:
            pass

    # Step 2: collect garbage to release remaining references
    gc.collect()

    # Step 3: try shutil first
    try:
        shutil.rmtree(path)
        return
    except Exception:
        pass

    # Step 4: Windows fallback — rd runs as separate process,
    # bypasses Python file handle tracking entirely
    os.system(f'rd /s /q "{path}"')


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

    st.markdown("""
        <style>
        div[data-testid="stButton"] button[kind="secondary"] {
            color: red;
            border: none;
            font-size: 23px;
            padding: 0.4em 0.8em;
            border-radius: 5px;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("Processed Result")

    with st.container(border=True):

        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("✖", key="close_btn"):
                st.session_state.done = False
                st.session_state.final_image = None
                st.session_state.majority_time = None
                st.rerun()

        img = st.session_state.final_image
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.image(img)

    st.header("Prediction Result")
    st.write(f"Predicted Time Class: T{st.session_state.majority_time}")

    with st.container(border=True):
        st.info(time_descriptions[st.session_state.majority_time])

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

        try:
            zip_path = os.path.join(tmpdir, "data.zip")

            with open(zip_path, "wb") as f:
                f.write(zip_file.getbuffer())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # -------------------------------
            # FIND FILES
            # -------------------------------
            hdr_path  = None
            data_path = None
            rgb_path  = None

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
            # Load into a plain numpy array immediately so spectral
            # releases its internal mmap as soon as possible.
            # -------------------------------
            hsi_img = spectral.open_image(hdr_path)
            cube    = np.array(hsi_img.load(), dtype=np.float32)

            # Explicitly close the mmap before anything else
            if hasattr(hsi_img, 'memmap'):
                try:
                    hsi_img.memmap.close()
                except Exception:
                    pass

            if hasattr(hsi_img, '_memmap'):
                try:
                    hsi_img._memmap.close()
                except Exception:
                    pass

            del hsi_img
            gc.collect()

            cube = np.rot90(cube, k=-1, axes=(0, 1))
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
            gray   = np.mean(cube, axis=2)
            gray   = (gray - gray.min()) / (gray.max() - gray.min())
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
            coords  = np.column_stack(np.where(mask == 1))
            spectra = cube[mask == 1]

            # Free the full cube — no longer needed
            del cube
            gc.collect()

            # -------------------------------
            # TIME MODEL
            # -------------------------------
            time_model  = joblib.load(TIME_MODEL_PATH)
            time_scaler = joblib.load(TIME_SCALER_PATH)

            spectra_time  = time_scaler.transform(spectra)
            time_pred     = time_model.predict(spectra_time)
            majority_time = Counter(time_pred).most_common(1)[0][0]

            st.session_state.majority_time = majority_time

            # -------------------------------
            # BRUISE MODEL
            # -------------------------------
            model_folder = os.path.join(BASE_MODEL_DIR, f"Models_xgb_t{majority_time}")

            model  = joblib.load(os.path.join(model_folder, "xgb_model.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "xgb_scaler.pkl"))

            spectra_scaled = scaler.transform(spectra)
            probs          = model.predict_proba(spectra_scaled)[:, 1]

            prob_map = np.zeros((H, W), dtype=np.float32)
            for (y, x), p in zip(coords, probs):
                prob_map[y, x] = p

            pred_map = (prob_map > THRESH).astype(np.uint8)
            # -------------------------------
            # BRUISE PIXEL CHECK OVERRIDE
            # -------------------------------
            bruise_pixel_count = np.sum(pred_map)

            if bruise_pixel_count < 750:
                majority_time = -1
                st.session_state.majority_time = majority_time

            # -------------------------------
            # CLEAN BLOBS
            # -------------------------------
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_map, 8)

            clean = np.zeros_like(pred_map)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 300:
                    clean[labels == i] = 1
            pred_map = clean

            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_CLOSE, kernel_big)
            pred_map = cv2.morphologyEx(pred_map, cv2.MORPH_OPEN,  kernel_small)

            pred_map = cv2.GaussianBlur(pred_map.astype(np.float32), (5, 5), 0)
            pred_map = (pred_map > 0.3).astype(np.uint8)

            contours, _ = cv2.findContours(pred_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            smooth_map = np.zeros_like(pred_map)
            for cnt in contours:
                if cv2.contourArea(cnt) < 300:
                    continue
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx  = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(smooth_map, [approx], -1, 1, -1)

            pred_map = smooth_map

            # -------------------------------
            # OVERLAY
            # -------------------------------
            rgb_norm_copy = rgb_norm.copy()
            rgb_norm_copy[pred_map == 1] = [1, 0, 0]

            alpha = 0.4
            final = (1 - alpha) * rgb_norm + alpha * rgb_norm_copy

            st.session_state.final_image = (final * 255).astype(np.uint8)
            st.session_state.done = True

        except Exception as e:
            st.error(f"Processing failed: {e}")

        finally:
            # force_delete closes mmap handles then deletes with
            # shutil.rmtree, falling back to Windows 'rd' if needed
            force_delete(tmpdir)

        if st.session_state.done:
            st.rerun()