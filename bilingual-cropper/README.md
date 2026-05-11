# Bilingual Exam Cropper - Ready Made

Full working code for Hindi-English PDF splitting using OpenRouter vision models.

## Quick Start
```bash
docker build -t exam-cropper .
docker run -p 8501:8501 -v $(pwd)/data:/app/data exam-cropper
```
Open http://localhost:8501

## Features
- ✅ OpenRouter integration (owl-alpha, Claude 3.5, GPT-4o, Gemini)
- ✅ Few-shot learning (saves corrections locally)
- ✅ PaddleOCR + DBSCAN fallback
- ✅ Geometric split as last resort
- ✅ 5px padding with edge clipping
- ✅ Auto-cleanup, keeps learning memory
- ✅ Full error handling

## Get API Key
1. Go to openrouter.ai
2. Create account, add $5 credit
3. Copy API key to sidebar
