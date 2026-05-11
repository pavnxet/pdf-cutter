# Bilingual Exam Cropper AI Agent

> Project specification for Claude Code / AI coding agents to build a self-learning PDF cropping system for Hindi-English bilingual exam papers.

## 1. Project Overview

**Problem:** Examination boards scan bilingual papers where one page contains Hindi (top or left) and English translation (bottom or right). Manual cropping is slow and inconsistent.

**Solution:** Build an AI agent that learns from human-drawn boxes. User uploads a scanned PDF/image, draws two boxes (hindi, english) once, the system trains a YOLOv8 object detection model via Roboflow, and future pages are auto-cropped. The agent improves via active learning.

**Target User:** Non-developer in India (exam center staff). Must work with drag-drop, no CLI.

## 2. Core Goals

1. Learn layout by example, not by fixed coordinates
2. Support both Top-Bottom and Left-Right splits (separate models)
3. Input: PDF or JPG/PNG scan (200-600 DPI)
4. Output: `document_hindi.pdf` and `document_english.pdf`
5. Feedback loop: correct a bad crop → retrain in <15 mins
6. Runs locally or on free cloud (Colab / Hugging Face)

## 3. Architecture

```
[User Upload PDF] 
    → pdf2image (300 DPI) 
    → [Inference Service]
        → Roboflow YOLOv8 model (primary)
        → Fallback: PaddleOCR language detection
    → Crop boxes with 5px padding
    → Merge crops → 2 PDFs
    → [Review UI] → user corrects → send to Roboflow → retrain
```

**Key Components:**
- **Frontend:** Streamlit web app (drag-drop, box editor)
- **Backend:** FastAPI
- **Model Training:** Roboflow API (no-code training)
- **OCR Fallback:** PaddleOCR (supports 100+ languages including Hindi/Devanagari)
- **Storage:** Local `/data` folder with structure below

## 4. Tech Stack

| Layer | Tool | Why |
| --- | --- | --- |
| Detection | YOLOv8s via Roboflow | 1-click training, free tier, 95% accuracy with 25 images |
| OCR Fallback | PaddleOCR 2.9 | Detects Hindi vs English script without training |
| PDF handling | pdf2image, PyMuPDF, img2pdf | Convert and rebuild PDFs |
| UI | Streamlit | Non-developer friendly |
| API | FastAPI | For Claude Code to call |
| Hosting | Hugging Face Spaces or local | Free |

## 5. Data Model

**Classes:** 
- `hindi` (label 0)
- `english` (label 1)

**Annotation format:** COCO JSON or YOLO txt (Roboflow exports both)

**Training set requirement:** Minimum 20 annotated pages per layout type. Ideal 50.

## 6. Directory Structure (for Claude Code)

```
bilingual-cropper/
├── app.py                 # Streamlit UI
├── api.py                 # FastAPI backend
├── core/
│   ├── pdf_processor.py   # PDF → images → PDF
│   ├── infer.py           # Roboflow inference + fallback
│   ├── cropper.py         # Crop with padding
│   └── trainer.py         # Upload corrections to Roboflow
├── models/
│   └── .gitkeep           # YOLO weights cache
├── data/
│   ├── uploads/
│   ├── crops_hindi/
│   └── crops_english/
├── requirements.txt
└── README.md
```

## 7. Implementation Details for Claude

### 7.1 PDF Processing
```python
# pdf_processor.py
def pdf_to_images(pdf_path, dpi=300):
    from pdf2image import convert_from_path
    return convert_from_path(pdf_path, dpi=dpi)

def images_to_pdf(images, output_path):
    import img2pdf
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert([img.filename for img in images]))
```

### 7.2 Inference (Primary)
- Use Roboflow Inference SDK
- API_KEY from env `ROBOFLOW_API_KEY`
- Model ID: `exam-cropper/3` (versioned)
- Confidence threshold: 0.5
- Return boxes sorted by y-coordinate (top-bottom) or x-coordinate (left-right)

### 7.3 Fallback (No Training)
If model confidence <0.4, use PaddleOCR:
1. Run `ocr.ocr(image, cls=True)`
2. For each text box, detect script: Devanagari Unicode range → hindi, Latin → english
3. Cluster boxes into two large regions using DBSCAN
4. Return bounding boxes

PaddleOCR supports Hindi and English natively, enabling layout detection without labeled data.

### 7.4 Active Learning Loop
1. User reviews crops in Streamlit
2. If wrong, click "Correct" → opens built-in box editor (streamlit-drawable-canvas)
3. Save corrected annotation as YOLO format
4. `trainer.py` uploads to Roboflow via API: `project.upload(image_path, annotation_path)`
5. Trigger training: `project.train()` → webhook notifies when done

## 8. Streamlit UI Flow

1. **Upload** – drag PDF
2. **Choose Layout** – "Top-Bottom" or "Left-Right"
3. **Process** – shows progress bar, displays first page with predicted boxes overlay
4. **Review** – side-by-side Hindi / English crops
5. **Correct** – draw new boxes if needed → "Save & Retrain"
6. **Download** – two PDFs

## 9. API Endpoints (FastAPI)

- `POST /process` – input: PDF file, layout; output: job_id
- `GET /status/{job_id}` – returns progress
- `GET /download/{job_id}/{lang}` – hindi or english PDF
- `POST /feedback` – upload corrected boxes

## 10. Training Instructions for Agent

When user provides 20+ examples:
1. Create Roboflow project via API if not exists
2. Upload images + annotations
3. Generate dataset version with augmentations: rotate ±5°, brightness ±15%, blur 1px
4. Train YOLOv8s for 50 epochs
5. Deploy model, store model_id in `.env`

Roboflow automates object detection training, making it ideal for non-developers.

## 11. Prompts for Claude Code

Use these exact prompts during development:

**Bootstrap:**
> "Create the full project structure above. Implement pdf_processor.py with pdf2image at 300 DPI. Use Python 3.11."

**Inference:**
> "Implement infer.py that calls Roboflow Inference API, falls back to PaddleOCR Hindi/English detection if confidence low. Return dict with boxes."

**UI:**
> "Build Streamlit app.py with file_uploader, show image with boxes using st.image and PIL draw, allow correction with streamlit-drawable-canvas."

## 12. Acceptance Criteria

- [ ] Upload 10-page bilingual PDF → get 2 PDFs in <60 seconds
- [ ] Accuracy >90% on unseen scans after 25 training samples
- [ ] User can correct 1 page → model retrains automatically
- [ ] Works offline after model download (optional)
- [ ] No command line needed

## 13. Future Enhancements

- Auto-detect layout (top-bottom vs left-right) using aspect ratio
- Batch processing for 100+ PDFs
- Export crops as searchable PDF using Tesseract Hindi+English
- Mobile app via PWA

## 14. Environment Variables

```
ROBOFLOW_API_KEY=rf_...
ROBOFLOW_PROJECT=exam-cropper
ROBOFLOW_MODEL_VERSION=3
PADDLEOCR_LANGS=hi,en
```

## 15. Notes for Non-Developer Deployment

1. Install: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. First run: paste Roboflow API key in sidebar
4. Upload 20 samples → click "Train Initial Model"

This spec is designed so Claude Code can generate working code in one pass, with clear separation between UI, inference, and training.
