# PDF Accessibility Assessment Using LayoutLMv3

## Project Description

This project aims to develop a deep learning-based system that evaluates the accessibility of PDF documents using document understanding and multimodal machine learning techniques. The primary goal is to help instructors identify potential accessibility issues in course materials, specifically PDF documents.

The project uses **Microsoft LayoutLMv3**, a pretrained multimodal transformer that processes text, visual information, and spatial document layouts. By combining document layout recognition with accessibility-specific evaluation, the system aims to identify potential issues and provide criterion-level assessments.

The initial assessment focuses on three accessibility criteria:

1. Alternative Text Quality
2. Color Contrast
3. Table Structure

The system will produce accessibility classifications, confidence scores, and specific document identification of violations. The overall accessibility score will be calculated from the criterion-level results.

**Note:** This project is intended as an automated preliminary accessibility assessment tool. Its predictions do not constitute official WCAG compliance certification or replace a comprehensive accessibility review.

## Installation and Setup

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone (https://github.com/carbajalc/EEE6778_Semester_Project.git)
cd EEE6778_Semester_Project
```

### 2. Install Python Dependencies

The project uses Python and several machine learning and PDF-processing libraries.

Install the required dependencies:

```bash
pip install torch torchvision
pip install transformers datasets
pip install pymupdf
pip install pandas numpy matplotlib seaborn
pip install pillow scikit-learn
pip install ipywidgets
```

If a `requirements.txt` file is provided, install the dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Environment Requirements

Recommended environment:

- Python 3.10 or later
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook or Google Colab
- GPU acceleration (recommended for model training)

A GPU is recommended when fine-tuning LayoutLMv3 because of its computational requirements. The notebook can also be executed using a CPU, although training may take considerably longer.

## How to Run the Notebook

The primary development and experimentation environment for this project is Jupyter Notebook or Google Colab.

### Option 1: Google Colab

1. Download or clone the repository.
2. Upload the notebook (`.ipynb`) to Google Colab.
3. Connect to a runtime. A GPU runtime is recommended.
4. Install the required dependencies if they are not already available.
5. Download or upload the datasets described below.
6. Update the dataset directory paths in the notebook.
7. Execute the notebook cells in order.

### Option 2: Jupyter Notebook

Launch Jupyter Notebook from the project directory:

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

### Expected Workflow

The notebook follows the general workflow below:

1. **Dataset Preparation:** Load and inspect the DocLayNet and PDF Accessibility Benchmark datasets.
2. **PDF Processing:** Extract text, images, document metadata, and structural information from PDF files.
3. **Preprocessing:** Prepare document images, text tokens, and bounding boxes for LayoutLMv3.
4. **Model Development:** Load the pretrained LayoutLMv3 model and prepare it for document layout recognition and accessibility-related classification.
5. **Model Training:** Train or fine-tune the model using the available annotations and accessibility labels.
6. **Evaluation:** Evaluate model performance using classification metrics such as accuracy, precision, recall, and F1-score.
7. **Accessibility Assessment:** Generate criterion-level predictions and confidence scores for PDF documents.

The exact execution steps may vary depending on the notebook and model implementation.

## Dataset Information

This project uses two primary datasets.

### 1. DocLayNet

DocLayNet is a large-scale document layout annotation dataset developed by IBM Research. It contains approximately 80,000 manually annotated document pages across multiple document types.

The dataset includes 11 document layout categories:

- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title

**Purpose in this project:**

DocLayNet provides training data for recognizing and classifying document layout elements. This information can help LayoutLMv3 identify important document components, including headings, images, paragraphs, and tables.

**Dataset sources:**

- [DocLayNet GitHub Repository](https://github.com/DS4SD/DocLayNet)
- [DocLayNet on Hugging Face](https://huggingface.co/datasets/docling-project/DocLayNet)

### 2. PDF Accessibility Benchmark (2025)

The PDF Accessibility Benchmark is a dataset containing 125 PDF documents with accessibility annotations across seven criteria.

The dataset provides four possible annotation labels:

- **Passed:** The document satisfies the evaluated accessibility criterion.
- **Failed:** The document does not satisfy the evaluated accessibility criterion.
- **Not Present:** The evaluated feature is absent from the document.
- **Cannot Tell:** There is insufficient information to determine whether the criterion is satisfied.

The seven evaluated criteria are:

1. Alternative Text Quality
2. Color Contrast
3. Font Readability
4. Functional Hyperlinks
5. Logical Reading Order
6. Semantic Tagging
7. Table Structure

**Purpose in this project:**

This dataset provides accessibility-specific annotations that can be used to develop and evaluate the system's ability to identify potential accessibility issues in PDF documents. However, not all criteria and annotation labels may be used in the project.

**Dataset source:**

- [PDF Accessibility Benchmark GitHub Repository](https://github.com/Anukriti12/PDF-Accessibility-Benchmark)
- [Research Paper (arXiv)](https://arxiv.org/abs/2509.18965)

### Dataset Preprocessing

Before training, the datasets require preprocessing to ensure compatibility with the model.

The preprocessing workflow may include:

- Extracting text and images from PDF documents.
- Rendering PDF pages into images.
- Extracting document layout information and bounding boxes.
- Mapping annotations to the appropriate layout categories.
- Converting accessibility labels into numerical representations.
- Handling missing values and uncertain annotations.
- Splitting documents into training, validation, and testing sets.

Data augmentation may also be explored to address the limited number of accessibility-labeled documents. All augmented versions of an original PDF should remain in the same dataset split to prevent data leakage.

**Important:** DocLayNet provides document layout annotations, not accessibility labels. The PDF Accessibility Benchmark provides accessibility-specific annotations. These datasets serve different purposes within the project.

## Project Structure

The repository is organized around the following components:

```text
EEE6778_Semester_Project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── DocLayNet/
│   └── PDF-Accessibility-Benchmark/
│
├── documents/
│   └── (All written documents submitted)
│
├── notebooks/
│   └── (All coding notebooks)
|
└── results/
    └── (Evaluation results and visualizations)
|
├── src/
│   └── (Any helper functions)
│
└── ui/
    └── (User interface code)
```

*Note: This is a suggested directory structure. Update it to match the actual organization of the repository.*

## Author

**Christina Carbajal**

M.S. Applied Data Science  
University of Florida

GitHub: [YOUR-GITHUB-USERNAME](https://github.com/YOUR-GITHUB-USERNAME)  
Email: YOUR-EMAIL-ADDRESS

---

## Acknowledgments

This project uses the following resources:

- IBM Research for the DocLayNet dataset.
- The authors of the PDF Accessibility Benchmark (2025).
- Microsoft for the pretrained LayoutLMv3 architecture and Hugging Face model resources.

LayoutLMv3 documentation:  
https://huggingface.co/microsoft/layoutlmv3-base
