
# ============================================================
# PDF ACCESSIBILITY EVALUATOR
# UPDATED CONTRASTIVE CLIP MODEL
# INSTRUCTOR REVIEW VERSION
# ============================================================

import os
import sys
import json
import io
import html
import hashlib
import importlib.util

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/EEE6778_Semester_Project"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "clip_contrastive_model"
)

PDF_PROCESSOR_PATH = os.path.join(
    PROJECT_DIR,
    "pdf_processor.py"
)

IMAGE_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "updated_instructor_extracted_images"
)

os.makedirs(
    IMAGE_OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

REVIEW_STATUSES = [
    "Not reviewed",
    "Approved",
    "Revised",
    "Decorative",
    "Needs follow-up"
]


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Accessibility Evaluator",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Accessibility Evaluator")

st.caption(
    "Instructor review application using the newly trained "
    "contrastive CLIP model."
)

st.write(
    """
    Upload course PDF documents to identify embedded images,
    inspect their existing alternative text, and review
    image-text similarity results.

    Instructors can approve existing alternative text,
    revise descriptions, identify decorative images, and
    document follow-up actions.
    """
)

st.divider()


# ============================================================
# 3. VERIFY MODEL AND PROCESSOR FILES
# ============================================================

config_path = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

if not os.path.isfile(config_path):
    st.error(
        f"CLIP model configuration was not found: {config_path}"
    )
    st.stop()

if not os.path.isfile(
    os.path.join(MODEL_DIR, "config.json")
):
    st.error(
        f"Hugging Face CLIP model files were not found: {MODEL_DIR}"
    )
    st.stop()

if not os.path.isfile(PDF_PROCESSOR_PATH):
    st.error(
        f"PDF processor was not found: {PDF_PROCESSOR_PATH}"
    )
    st.stop()

with open(config_path, "r") as f:
    CONFIG = json.load(f)


# The current training configuration uses model_name and
# similarity_threshold. Fallbacks support older configurations.

CLIP_MODEL_NAME = (
    CONFIG.get("model_name")
    or CONFIG.get("base_model")
    or CONFIG.get("clip_model")
)

threshold_value = CONFIG.get(
    "similarity_threshold",
    CONFIG.get("clip_similarity_threshold")
)

if not CLIP_MODEL_NAME:
    st.error(
        "The model configuration does not contain "
        "model_name, base_model, or clip_model."
    )
    st.stop()

if threshold_value is None:
    st.error(
        "The model configuration does not contain "
        "similarity_threshold or clip_similarity_threshold."
    )
    st.stop()

CLIP_THRESHOLD = float(threshold_value)


# ============================================================
# 4. IMPORT THE EXISTING PDF PROCESSOR
# ============================================================

# Do not modify pdf_processor.py.
# Its process_uploaded_pdfs() function expects Streamlit
# UploadedFile objects, not filesystem paths.

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

spec = importlib.util.spec_from_file_location(
    "project_pdf_processor",
    PDF_PROCESSOR_PATH
)

pdf_processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_processor)

process_uploaded_pdfs = (
    pdf_processor.process_uploaded_pdfs
)


# ============================================================
# 5. LOAD THE NEW CONTRASTIVE CLIP MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_clip_model(model_dir, device_name):

    processor = CLIPProcessor.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model = CLIPModel.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model = model.to(device_name)
    model.eval()

    return processor, model


# ============================================================
# 6. CALCULATE IMAGE-TEXT COSINE SIMILARITY
# ============================================================

def calculate_clip_similarity(
    image_path,
    alt_text,
    processor,
    model
):

    if alt_text is None:
        return None

    alt_text = str(alt_text).strip()

    if not alt_text:
        return None

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file was not found: {image_path}"
        )

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    inputs = processor(
        text=[alt_text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.inference_mode():

        # Image encoder and projection layer.
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )

        image_features = model.visual_projection(
            vision_outputs.pooler_output
        )

        # Text encoder and projection layer.
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        text_features = model.text_projection(
            text_outputs.pooler_output
        )

        # Normalize embeddings.
        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1
        )

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1
        )

        # Cosine similarity.
        similarity = torch.sum(
            image_features * text_features,
            dim=-1
        ).item()

    return float(similarity)


# ============================================================
# 7. RESOLVE IMAGE FILE PATHS
# ============================================================

def resolve_image_path(image_path):

    if image_path is None:
        return ""

    if pd.isna(image_path):
        return ""

    image_path = str(image_path)

    candidates = [
        image_path,
        os.path.join(IMAGE_OUTPUT_DIR, image_path),
        os.path.join(
            IMAGE_OUTPUT_DIR,
            os.path.basename(image_path)
        ),
        os.path.join(
            IMAGE_OUTPUT_DIR,
            "images",
            os.path.basename(image_path)
        )
    ]

    for candidate in candidates:

        if os.path.isfile(candidate):
            return candidate

    return image_path


# ============================================================
# 8. EVALUATE EXTRACTED IMAGES
# ============================================================

def evaluate_extracted_images(
    image_df,
    processor,
    model,
    threshold
):

    required_columns = [
        "document_id",
        "page_id",
        "figure_id",
        "image_path",
        "alt_text"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in image_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The PDF processor did not return the required "
            f"columns: {missing_columns}"
        )

    results = []

    for _, row in image_df.iterrows():

        image_path = resolve_image_path(
            row["image_path"]
        )

        alt_text = row["alt_text"]

        similarity = None
        error_message = ""

        if (
            pd.isna(alt_text)
            or not str(alt_text).strip()
        ):

            classification = "Missing alternative text"

        else:

            try:

                similarity = calculate_clip_similarity(
                    image_path=image_path,
                    alt_text=alt_text,
                    processor=processor,
                    model=model
                )

                if similarity >= threshold:
                    classification = "Meaningful"
                else:
                    classification = "Potentially non-meaningful"

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
# 9. GENERATE A PDF REVIEW REPORT
# ============================================================

def generate_pdf_report(review_df):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.4 * inch,
        leftMargin=0.4 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=17,
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10
    )

    story = []

    story.append(
        Paragraph(
            "PDF Accessibility Instructor Review Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Generated from the updated contrastive CLIP "
            "PDF accessibility evaluator.",
            body_style
        )
    )

    story.append(Spacer(1, 12))

    total = len(review_df)

    approved = (
        review_df["review_status"] == "Approved"
    ).sum()

    revised = (
        review_df["review_status"] == "Revised"
    ).sum()

    decorative = (
        review_df["review_status"] == "Decorative"
    ).sum()

    follow_up = (
        review_df["review_status"] == "Needs follow-up"
    ).sum()

    not_reviewed = (
        review_df["review_status"] == "Not reviewed"
    ).sum()

    summary_data = [
        ["Metric", "Count"],
        ["Total images", str(total)],
        ["Approved", str(approved)],
        ["Revised", str(revised)],
        ["Decorative", str(decorative)],
        ["Needs follow-up", str(follow_up)],
        ["Not reviewed", str(not_reviewed)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[2.5 * inch, 1.0 * inch]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(summary_table)
    story.append(Spacer(1, 16))

    story.append(
        Paragraph(
            "Image-Level Review Details",
            styles["Heading2"]
        )
    )

    table_data = [[
        "Document",
        "Page",
        "Figure",
        "CLIP Score",
        "Model Finding",
        "Review Status",
        "Instructor Alt Text",
        "Notes"
    ]]

    for _, row in review_df.iterrows():

        score = row["similarity_score"]

        score_text = (
            f"{score:.4f}"
            if pd.notna(score)
            else "N/A"
        )

        values = [
            row["document_id"],
            row["page_id"],
            row["figure_id"],
            score_text,
            row["classification"],
            row["review_status"],
            row["instructor_alt_text"],
            row["review_notes"]
        ]

        table_data.append([
            Paragraph(
                html.escape(str(value or "")),
                body_style
            )
            for value in values
        ])

    detail_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            1.15 * inch,
            0.4 * inch,
            0.65 * inch,
            0.65 * inch,
            1.05 * inch,
            0.85 * inch,
            2.0 * inch,
            1.5 * inch
        ]
    )

    detail_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4)
        ])
    )

    story.append(detail_table)

    story.append(Spacer(1, 14))

    story.append(
        Paragraph(
            "Interpretation note: CLIP cosine similarity is an "
            "image-text alignment measure. It is not a probability, "
            "a standalone accessibility score, or proof of WCAG "
            "or PDF/UA compliance. Instructor review is required.",
            body_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 10. APPLICATION HEADER AND MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "CLIP Model",
        os.path.basename(MODEL_DIR)
    )

with col2:
    st.metric(
        "Similarity Threshold",
        f"{CLIP_THRESHOLD:.6f}"
    )

with col3:
    st.metric(
        "Processing Device",
        str(DEVICE).upper()
    )

st.caption(
    "The similarity threshold is loaded from the saved "
    "contrastive CLIP model configuration."
)

st.divider()


# ============================================================
# 11. PDF UPLOAD
# ============================================================

st.header("1. Upload Course PDFs")

uploaded_files = st.file_uploader(
    "Select one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    key="instructor_pdf_uploader"
)

if uploaded_files:

    st.write(
        f"Uploaded documents: {len(uploaded_files)}"
    )

    for uploaded_file in uploaded_files:
        st.write(f"• {uploaded_file.name}")

    st.info(
        "Click the evaluation button to extract embedded "
        "images and evaluate their alternative text."
    )

    evaluate_button = st.button(
        "Extract Images and Evaluate Accessibility",
        type="primary",
        key="instructor_evaluate_button"
    )

    if evaluate_button:

        # ----------------------------------------------------
        # Load CLIP model.
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Loading the updated contrastive CLIP model..."
            ):

                processor, clip_model = load_clip_model(
                    MODEL_DIR,
                    str(DEVICE)
                )

            st.success(
                "The updated CLIP model loaded successfully."
            )

        except Exception as exc:

            st.error(
                f"Could not load the CLIP model: {exc}"
            )

            st.exception(exc)
            st.stop()

        # ----------------------------------------------------
        # Extract images and alt text.
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Extracting images and alternative text from PDFs..."
            ):

                # IMPORTANT:
                # Pass the original UploadedFile objects.
                # Do not pass file paths or call process_pdf().
                image_df, processing_log = (
                    process_uploaded_pdfs(
                        uploaded_files=uploaded_files,
                        output_image_dir=IMAGE_OUTPUT_DIR
                    )
                )

            st.subheader("PDF Processing Log")

            for message in processing_log:

                if str(message).startswith("Error"):
                    st.error(message)
                else:
                    st.success(message)

            if image_df is None or image_df.empty:

                st.warning(
                    "No embedded images were extracted. "
                    "Check the PDF processing log."
                )

                st.stop()

            st.success(
                f"Successfully extracted {len(image_df)} images."
            )

        except Exception as exc:

            st.error(
                f"PDF processing failed: {exc}"
            )

            st.exception(exc)
            st.stop()

        # ----------------------------------------------------
        # Evaluate extracted images.
        # ----------------------------------------------------

        try:

            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_images = len(image_df)

            for index, (_, row) in enumerate(
                image_df.iterrows()
            ):

                status_text.write(
                    f"Evaluating image {index + 1} "
                    f"of {total_images}..."
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
                    (index + 1) / total_images
                )

            results_df = pd.concat(
                results,
                ignore_index=True
            )

            # Create a stable row identifier for review widgets.
            results_df["review_id"] = [
                hashlib.md5(
                    (
                        str(row["document_id"])
                        + "|"
                        + str(row["page_id"])
                        + "|"
                        + str(row["figure_id"])
                        + "|"
                        + str(index)
                    ).encode("utf-8")
                ).hexdigest()[:12]
                for index, row in results_df.iterrows()
            ]

            st.session_state[
                "instructor_evaluation_results"
            ] = results_df

            # Reset review state for the newly processed batch.
            st.session_state[
                "instructor_review_state"
            ] = {}

            status_text.success(
                "CLIP evaluation completed successfully."
            )

        except Exception as exc:

            st.error(
                f"CLIP evaluation failed: {exc}"
            )

            st.exception(exc)
            st.stop()


# ============================================================
# 12. LOAD RESULTS FROM SESSION STATE
# ============================================================

if "instructor_evaluation_results" in st.session_state:

    results_df = st.session_state[
        "instructor_evaluation_results"
    ].copy()

    if "instructor_review_state" not in st.session_state:
        st.session_state[
            "instructor_review_state"
        ] = {}

    review_state = st.session_state[
        "instructor_review_state"
    ]

    st.divider()

    st.header("2. Evaluation Dashboard")

    total_images = len(results_df)

    meaningful_count = (
        results_df["classification"] == "Meaningful"
    ).sum()

    flagged_count = (
        results_df["classification"]
        == "Potentially non-meaningful"
    ).sum()

    missing_count = (
        results_df["classification"]
        == "Missing alternative text"
    ).sum()

    error_count = (
        results_df["classification"]
        == "Evaluation error"
    ).sum()

    # --------------------------------------------------------
    # Initialize review state for each image.
    # --------------------------------------------------------

    for _, row in results_df.iterrows():

        review_id = row["review_id"]

        if review_id not in review_state:

            original_alt = row["alt_text"]

            if pd.isna(original_alt):
                original_alt = ""
            else:
                original_alt = str(original_alt)

            review_state[review_id] = {
                "instructor_alt_text": original_alt,
                "review_status": "Not reviewed",
                "review_notes": "",
                "decorative": False
            }

    # --------------------------------------------------------
    # Dashboard metrics.
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Images",
        int(total_images)
    )

    col2.metric(
        "Meaningful",
        int(meaningful_count)
    )

    col3.metric(
        "Potentially Non-meaningful",
        int(flagged_count)
    )

    col4.metric(
        "Missing Alt Text",
        int(missing_count)
    )

    col5.metric(
        "Evaluation Errors",
        int(error_count)
    )

    st.caption(
        "A model finding is a review prompt, not a final "
        "accessibility determination."
    )

    # --------------------------------------------------------
    # Document-level summary.
    # --------------------------------------------------------

    st.subheader("Document Summary")

    document_summary = (
        results_df.groupby("document_id")
        .agg(
            total_images=("figure_id", "count"),
            meaningful=(
                "classification",
                lambda x: (x == "Meaningful").sum()
            ),
            potentially_non_meaningful=(
                "classification",
                lambda x: (
                    x == "Potentially non-meaningful"
                ).sum()
            ),
            missing_alt_text=(
                "classification",
                lambda x: (
                    x == "Missing alternative text"
                ).sum()
            )
        )
        .reset_index()
    )

    st.dataframe(
        document_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Prepare review status counts.
    # --------------------------------------------------------

    def create_review_dataframe():

        reviewed_records = []

        for _, row in results_df.iterrows():

            review_id = row["review_id"]
            state = review_state[review_id]

            record = row.to_dict()

            record["instructor_alt_text"] = (
                state["instructor_alt_text"]
            )

            record["review_status"] = (
                state["review_status"]
            )

            record["review_notes"] = (
                state["review_notes"]
            )

            record["decorative"] = (
                state["decorative"]
            )

            reviewed_records.append(record)

        return pd.DataFrame(reviewed_records)

    reviewed_df = create_review_dataframe()

    reviewed_count = (
        reviewed_df["review_status"] != "Not reviewed"
    ).sum()

    completion = (
        reviewed_count / total_images * 100
        if total_images > 0
        else 0
    )

    st.subheader("Instructor Review Progress")

    progress_col1, progress_col2 = st.columns([3, 1])

    with progress_col1:
        st.progress(completion / 100)

    with progress_col2:
        st.metric(
            "Review Completion",
            f"{completion:.1f}%"
        )

    # --------------------------------------------------------
    # Review status metrics.
    # --------------------------------------------------------

    reviewed_df = create_review_dataframe()

    status_counts = (
        reviewed_df["review_status"]
        .value_counts()
        .reindex(REVIEW_STATUSES, fill_value=0)
    )

    status_cols = st.columns(5)

    for index, status in enumerate(REVIEW_STATUSES):

        status_cols[index].metric(
            status,
            int(status_counts[status])
        )

    st.divider()


    # ========================================================
    # 13. FILTER AND SORT IMAGE RESULTS
    # ========================================================

    st.header("3. Instructor Image Review")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        classification_options = [
            "All",
            "Meaningful",
            "Potentially non-meaningful",
            "Missing alternative text",
            "Evaluation error"
        ]

        selected_classification = st.selectbox(
            "Filter by model finding",
            classification_options,
            key="instructor_classification_filter"
        )

    with filter_col2:

        review_options = [
            "All"
        ] + REVIEW_STATUSES

        selected_review_status = st.selectbox(
            "Filter by review status",
            review_options,
            key="instructor_review_filter"
        )

    with filter_col3:

        sort_option = st.selectbox(
            "Sort images",
            [
                "Document order",
                "Lowest similarity first",
                "Highest similarity first"
            ],
            key="instructor_sort_option"
        )

    display_df = create_review_dataframe()

    if selected_classification != "All":

        display_df = display_df[
            display_df["classification"]
            == selected_classification
        ]

    if selected_review_status != "All":

        display_df = display_df[
            display_df["review_status"]
            == selected_review_status
        ]

    if sort_option == "Lowest similarity first":

        display_df = display_df.sort_values(
            "similarity_score",
            ascending=True,
            na_position="first"
        )

    elif sort_option == "Highest similarity first":

        display_df = display_df.sort_values(
            "similarity_score",
            ascending=False,
            na_position="last"
        )

    st.write(
        f"Displaying {len(display_df)} image(s)."
    )

    # ========================================================
    # 14. INDIVIDUAL INSTRUCTOR REVIEW
    # ========================================================

    for _, row in display_df.iterrows():

        review_id = row["review_id"]
        state = review_state[review_id]

        document_id = row["document_id"]
        page_id = row["page_id"]
        figure_id = row["figure_id"]

        expander_title = (
            f"Document: {document_id} | "
            f"Page: {page_id} | "
            f"Figure: {figure_id} | "
            f"{row['classification']}"
        )

        with st.expander(
            expander_title,
            expanded=False
        ):

            image_col, details_col = st.columns([1, 2])

            with image_col:

                image_path = row["image_path"]

                if (
                    image_path
                    and os.path.isfile(image_path)
                ):

                    st.image(
                        image_path,
                        caption=f"Figure {figure_id}",
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "The extracted image file could not be found."
                    )

            with details_col:

                st.write(
                    "**Document ID:**",
                    document_id
                )

                st.write(
                    "**Page:**",
                    page_id
                )

                st.write(
                    "**Figure ID:**",
                    figure_id
                )

                st.write("**Original Alternative Text**")

                original_alt = row["alt_text"]

                if (
                    pd.notna(original_alt)
                    and str(original_alt).strip()
                ):

                    st.info(str(original_alt))

                else:

                    st.warning(
                        "No alternative text was found."
                    )

                score = row["similarity_score"]

                if pd.notna(score):

                    st.metric(
                        "CLIP Cosine Similarity",
                        f"{score:.6f}"
                    )

                else:

                    st.write(
                        "**CLIP Cosine Similarity:** Not available"
                    )

                if row["classification"] == "Meaningful":

                    st.success(
                        "Model finding: Meaningful"
                    )

                elif row["classification"] == "Potentially non-meaningful":

                    st.warning(
                        "Model finding: Potentially non-meaningful"
                    )

                elif row["classification"] == "Missing alternative text":

                    st.error(
                        "Model finding: Missing alternative text"
                    )

                else:

                    st.error(
                        f"Model evaluation error: {row['error']}"
                    )

            st.divider()

            st.markdown("**Instructor Review**")

            # ------------------------------------------------
            # Editable alternative text.
            # ------------------------------------------------

            new_alt_text = st.text_area(
                "Instructor-approved alternative text",
                value=state["instructor_alt_text"],
                key=f"alt_text_{review_id}",
                height=110,
                help=(
                    "Edit the alternative text when the original "
                    "description is incomplete, inaccurate, or unclear."
                )
            )

            state["instructor_alt_text"] = new_alt_text

            # ------------------------------------------------
            # Decorative image selection.
            # ------------------------------------------------

            decorative = st.checkbox(
                "This image is decorative and does not require descriptive alternative text.",
                value=state["decorative"],
                key=f"decorative_{review_id}"
            )

            state["decorative"] = decorative

            # ------------------------------------------------
            # Review status.
            # ------------------------------------------------

            selected_status = st.selectbox(
                "Review decision",
                REVIEW_STATUSES,
                index=REVIEW_STATUSES.index(
                    state["review_status"]
                ),
                key=f"status_{review_id}"
            )

            # Decorative selection takes precedence.
            if decorative:

                selected_status = "Decorative"

                st.caption(
                    "This image is marked decorative. "
                    "Confirm that it conveys no information "
                    "or functionality needed by the learner."
                )

            elif selected_status == "Decorative":

                decorative = True
                state["decorative"] = True

            state["review_status"] = selected_status

            # ------------------------------------------------
            # Review notes.
            # ------------------------------------------------

            review_notes = st.text_area(
                "Instructor review notes",
                value=state["review_notes"],
                key=f"notes_{review_id}",
                height=80,
                help=(
                    "Record why the description was approved, "
                    "revised, or flagged for follow-up."
                )
            )

            state["review_notes"] = review_notes

            # ------------------------------------------------
            # Validation messages.
            # ------------------------------------------------

            if (
                selected_status in ["Approved", "Revised"]
                and not str(new_alt_text).strip()
            ):

                st.warning(
                    "An approved or revised image must have "
                    "alternative text. If the image is decorative, "
                    "select the decorative checkbox instead."
                )

            if (
                selected_status == "Revised"
                and str(new_alt_text).strip()
                == str(original_alt or "").strip()
            ):

                st.info(
                    "The instructor alternative text currently "
                    "matches the original text."
                )

            if selected_status == "Needs follow-up":

                st.warning(
                    "This image requires additional review."
                )

            review_state[review_id] = state

    # ========================================================
    # 15. FINALIZE REVIEW DATA
    # ========================================================

    reviewed_df = create_review_dataframe()

    # --------------------------------------------------------
    # Validate instructor decisions.
    # --------------------------------------------------------

    invalid_approvals = reviewed_df[
        (
            reviewed_df["review_status"].isin(
                ["Approved", "Revised"]
            )
        )
        &
        (
            reviewed_df["instructor_alt_text"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )
    ]

    st.divider()

    st.header("4. Review Summary")

    total_reviewed = (
        reviewed_df["review_status"] != "Not reviewed"
    ).sum()

    approved_count = (
        reviewed_df["review_status"] == "Approved"
    ).sum()

    revised_count = (
        reviewed_df["review_status"] == "Revised"
    ).sum()

    decorative_count = (
        reviewed_df["review_status"] == "Decorative"
    ).sum()

    follow_up_count = (
        reviewed_df["review_status"] == "Needs follow-up"
    ).sum()

    not_reviewed_count = (
        reviewed_df["review_status"] == "Not reviewed"
    ).sum()

    summary_cols = st.columns(5)

    summary_cols[0].metric(
        "Approved",
        int(approved_count)
    )

    summary_cols[1].metric(
        "Revised",
        int(revised_count)
    )

    summary_cols[2].metric(
        "Decorative",
        int(decorative_count)
    )

    summary_cols[3].metric(
        "Needs Follow-up",
        int(follow_up_count)
    )

    summary_cols[4].metric(
        "Not Reviewed",
        int(not_reviewed_count)
    )

    if len(invalid_approvals) > 0:

        st.warning(
            f"{len(invalid_approvals)} image(s) are marked "
            "Approved or Revised but have blank instructor "
            "alternative text. Update these decisions before "
            "exporting the review."
        )

    # --------------------------------------------------------
    # Results table.
    # --------------------------------------------------------

    st.subheader("Instructor Review Results")

    table_columns = [
        "document_id",
        "page_id",
        "figure_id",
        "classification",
        "similarity_score",
        "review_status",
        "instructor_alt_text",
        "review_notes"
    ]

    st.dataframe(
        reviewed_df[table_columns],
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # 16. EXPORT CSV
    # ========================================================

    st.divider()

    st.header("5. Export Instructor Review")

    export_df = reviewed_df.drop(
        columns=["image_path", "review_id"],
        errors="ignore"
    ).copy()

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Instructor Review as CSV",
        data=csv_data,
        file_name="instructor_pdf_accessibility_review.csv",
        mime="text/csv",
        key="instructor_csv_download"
    )

    # ========================================================
    # 17. EXPORT PDF REPORT
    # ========================================================

    if len(invalid_approvals) > 0:

        st.warning(
            "Resolve blank alternative text for Approved or "
            "Revised items before generating the final PDF report."
        )

    else:

        try:

            pdf_data = generate_pdf_report(
                reviewed_df
            )

            st.download_button(
                label="Download Instructor Review as PDF",
                data=pdf_data,
                file_name="instructor_pdf_accessibility_review.pdf",
                mime="application/pdf",
                key="instructor_pdf_download"
            )

        except Exception as exc:

            st.error(
                f"Could not generate the PDF report: {exc}"
            )


# ============================================================
# 18. MODEL INTERPRETATION AND LIMITATIONS
# ============================================================

st.divider()

with st.expander(
    "About the CLIP model and interpretation of results"
):

    st.markdown(
        f"""
        **Model:** `{CLIP_MODEL_NAME}`

        **Similarity threshold:** `{CLIP_THRESHOLD:.6f}`

        The application uses the newly trained contrastive CLIP
        model to calculate cosine similarity between an extracted
        PDF image and its existing alternative text.

        The image and text are encoded using the model's visual
        and text encoders and their corresponding projection layers.
        The resulting embeddings are L2-normalized before cosine
        similarity is calculated.

        **Important limitations:**

        - Cosine similarity is not a probability.
        - A high score does not guarantee that alternative text
          is accurate, complete, or appropriate in context.
        - A low score does not automatically mean that alternative
          text is inaccessible.
        - The model does not independently determine WCAG
          compliance or PDF/UA conformance.
        - Instructors should review the actual image and its
          surrounding document context before approving text.
        - The original CLIP similarity score is not recalculated
          when an instructor edits alternative text.

        The model should therefore be used as a screening and
        prioritization tool, with instructor judgment providing
        the final review decision.
        """
    )

st.caption(
    "Instructor review decisions are maintained in the current "
    "Streamlit session. Download the CSV or PDF report to preserve "
    "the review results outside the current session."
)
