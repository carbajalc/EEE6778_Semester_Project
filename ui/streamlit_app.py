# ============================================================
# PDF ACCESSIBILITY EVALUATOR - STREAMLIT APPLICATION
# ============================================================

import os
import sys
import json
import pandas as pd
import streamlit as st
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = "/content/drive/MyDrive/EEE6778_Semester_Project"
MODEL_DIR = os.path.join(PROJECT_DIR, "clip_training_results")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "streamlit_extracted_images")

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pdf_processor import process_uploaded_pdfs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD CLIP CONFIGURATION
# ============================================================

config_path = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

if not os.path.isfile(config_path):
    st.error(f"Model configuration not found: {config_path}")
    st.stop()

with open(config_path, "r") as f:
    config = json.load(f)

CLIP_MODEL_NAME = config["clip_model"]
CLIP_THRESHOLD = float(
    config["clip_similarity_threshold"]
)


# ============================================================
# LOAD CLIP MODEL ONLY WHEN NEEDED
# ============================================================

@st.cache_resource(show_spinner=False)
def load_clip_model(model_name):

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model = CLIPModel.from_pretrained(
        model_name
    ).to(DEVICE)

    model.eval()

    return processor, model




# ------------------------------------------------------------
# CLIP IMAGE-TEXT COSINE SIMILARITY
# ------------------------------------------------------------

def calculate_clip_similarity(
    image_path,
    alt_text,
    processor,
    model
):
    if alt_text is None or not str(alt_text).strip():
        return None

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    inputs = processor(
        text=[str(alt_text)],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        # Extract image and text features
        image_output = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        text_output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Handle Transformers versions that return
        # BaseModelOutputWithPooling instead of tensors.
        image_features = (
            image_output.pooler_output
            if hasattr(image_output, "pooler_output")
            else image_output
        )

        text_features = (
            text_output.pooler_output
            if hasattr(text_output, "pooler_output")
            else text_output
        )

        # Normalize feature vectors
        image_features = torch.nn.functional.normalize(
            image_features,
            p=2,
            dim=-1
        )

        text_features = torch.nn.functional.normalize(
            text_features,
            p=2,
            dim=-1
        )

        # Calculate cosine similarity
        similarity = (
            image_features @ text_features.T
        ).item()

    return float(similarity)



# ------------------------------------------------------------
# RESOLVE EXTRACTED IMAGE PATHS
# ------------------------------------------------------------

def resolve_image_path(path):
    path = str(path)

    candidates = [
        path,
        os.path.join(IMAGE_OUTPUT_DIR, path),
        os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(path)),
        os.path.join(IMAGE_OUTPUT_DIR, "images", os.path.basename(path))
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return path


# ------------------------------------------------------------
# EVALUATE EXTRACTED IMAGES
# ------------------------------------------------------------

def evaluate_extracted_images(image_df, processor, model, threshold):
    required_columns = [
        "document_id", "page_id", "figure_id", "image_path", "alt_text"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in image_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"PDF processor did not return required columns: {missing_columns}. "
            f"Available columns: {list(image_df.columns)}"
        )

    results = []

    for _, row in image_df.iterrows():
        image_path = resolve_image_path(row["image_path"])
        alt_text = row["alt_text"]

        similarity = None
        error_message = ""

        if pd.isna(alt_text) or not str(alt_text).strip():
            classification = "Missing alternative text"
        else:
            try:
                similarity = calculate_clip_similarity(
                    image_path=image_path,
                    alt_text=alt_text,
                    processor=processor,
                    model=model
                )

                classification = (
                    "Meaningful"
                    if similarity >= threshold
                    else "Potentially non-meaningful"
                )
            except Exception as exc:
                classification = "Evaluation error"
                error_message = str(exc)

        results.append({
            "document_id": row["document_id"],
            "page_id": row["page_id"],
            "figure_id": row["figure_id"],
            "image_path": image_path,
            "alt_text": alt_text,
            "similarity_score": similarity,
            "classification": classification,
            "error": error_message
        })

    return pd.DataFrame(results)


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="PDF Accessibility Evaluator",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Accessibility Evaluator")
st.write(
    "Upload PDF documents to evaluate semantic similarity between "
    "embedded images and their alternative text using your trained CLIP model."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CLIP Model", CLIP_MODEL_NAME.split("/")[-1])
with col2:
    st.metric("Similarity Threshold", f"{CLIP_THRESHOLD:.4f}")
with col3:
    st.metric("Processing Device", str(DEVICE).upper())

st.divider()

st.subheader("Upload PDF Documents")
uploaded_files = st.file_uploader(
    "Select one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.write(f"Uploaded documents: {len(uploaded_files)}")

    if st.button(
        "Evaluate PDF Accessibility",
        type="primary"
    ):

        # --------------------------------------------------------
        # LOAD CLIP MODEL
        # --------------------------------------------------------

        try:

            with st.spinner(
                "Loading CLIP model. This may take a moment..."
            ):

                processor, clip_model = load_clip_model(
                    CLIP_MODEL_NAME
                )

            st.success("CLIP model loaded successfully.")

        except Exception as exc:

            st.error(f"Could not load CLIP model: {exc}")
            st.exception(exc)
            st.stop()

        
        # --------------------------------------------------------
        # EXTRACT PDF IMAGES AND ALTERNATIVE TEXT
        # --------------------------------------------------------

        try:

            with st.spinner(
                "Extracting images and alternative text from PDFs..."
            ):

                # Streamlit UploadedFile objects are passed directly.
                # The updated PDF processor reads their bytes in memory.
                image_df, processing_log = process_uploaded_pdfs(
                    uploaded_files=uploaded_files,
                    output_image_dir=IMAGE_OUTPUT_DIR
                )

            # Display processing messages
            st.subheader("PDF Processing Log")

            for message in processing_log:
                st.write(message)

            if image_df is None or image_df.empty:

                st.warning(
                    "No embedded images were extracted."
                )

                st.stop()

            st.success(
                f"Successfully extracted {len(image_df)} images."
            )

        except Exception as exc:

            st.error(f"PDF processing failed: {exc}")
            st.exception(exc)
            st.stop()


        # --------------------------------------------------------
        # EVALUATE IMAGES USING CLIP
        # --------------------------------------------------------

        try:

            progress_bar = st.progress(0)

            status_text = st.empty()

            results = []

            for index, (_, row) in enumerate(
                image_df.iterrows()
            ):

                status_text.write(
                    f"Evaluating image {index + 1} "
                    f"of {len(image_df)}..."
                )

                single_image_df = pd.DataFrame([row])

                image_result = evaluate_extracted_images(
                    image_df=single_image_df,
                    processor=processor,
                    model=clip_model,
                    threshold=CLIP_THRESHOLD
                )

                results.append(image_result)

                progress_bar.progress(
                    (index + 1) / len(image_df)
                )

            results_df = pd.concat(
                results,
                ignore_index=True
            )

            st.session_state["evaluation_results"] = results_df

            status_text.success(
                "CLIP evaluation completed!"
            )

        except Exception as exc:

            st.error(f"CLIP evaluation failed: {exc}")
            st.exception(exc)
            st.stop()



        st.session_state["evaluation_results"] = results_df
        st.success("CLIP evaluation completed!")

if "evaluation_results" in st.session_state:
    results_df = st.session_state["evaluation_results"]

    st.divider()
    st.header("Evaluation Summary")

    total_images = len(results_df)
    meaningful = (results_df["classification"] == "Meaningful").sum()
    non_meaningful = (
        results_df["classification"] == "Potentially non-meaningful"
    ).sum()
    missing_alt = (
        results_df["classification"] == "Missing alternative text"
    ).sum()
    errors = (results_df["classification"] == "Evaluation error").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Images", int(total_images))
    col2.metric("Meaningful", int(meaningful))
    col3.metric("Potentially Non-meaningful", int(non_meaningful))
    col4.metric("Missing Alt Text", int(missing_alt))
    col5.metric("Evaluation Errors", int(errors))

    st.caption(
        "CLIP similarity is a semantic similarity measure, not a probability "
        "or a standalone determination of WCAG compliance."
    )

    st.subheader("Image Evaluation Results")
    display_df = results_df.drop(
        columns=["image_path", "error"]
    ).copy()
    display_df["similarity_score"] = display_df["similarity_score"].apply(
        lambda value: round(value, 4) if pd.notna(value) else None
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Individual Image Analysis")
    for _, row in results_df.iterrows():
        with st.expander(
            f"Figure {row['figure_id']} | Page {row['page_id']} | "
            f"{row['classification']}"
        ):
            image_col, details_col = st.columns([1, 2])

            with image_col:
                if os.path.isfile(row["image_path"]):
                    st.image(
                        row["image_path"],
                        caption=f"Figure {row['figure_id']}",
                        use_container_width=True
                    )
                else:
                    st.warning(f"Image file not found: {row['image_path']}")

            with details_col:
                st.write("**Document ID:**", row["document_id"])
                st.write("**Page:**", row["page_id"])
                st.write("**Figure ID:**", row["figure_id"])
                st.write("**Alternative Text:**")

                if pd.notna(row["alt_text"]) and str(row["alt_text"]).strip():
                    st.write(row["alt_text"])
                else:
                    st.warning("No alternative text found.")

                if pd.notna(row["similarity_score"]):
                    st.metric(
                        "Cosine Similarity",
                        f"{row['similarity_score']:.4f}"
                    )
                else:
                    st.write("**Cosine Similarity:** Not available")

                if row["classification"] == "Meaningful":
                    st.success("Meaningful")
                elif row["classification"] == "Potentially non-meaningful":
                    st.warning("Potentially non-meaningful")
                elif row["classification"] == "Missing alternative text":
                    st.error("Missing alternative text")
                else:
                    st.error(f"Evaluation error: {row['error']}")

                st.caption(
                    f"Classification threshold: {CLIP_THRESHOLD:.4f}"
                )

    st.divider()
    st.subheader("Download Evaluation Results")
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="pdf_accessibility_evaluation.csv",
        mime="text/csv"
    )
