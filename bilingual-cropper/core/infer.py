import os, base64, json
import numpy as np
from openai import OpenAI
from PIL import Image
from sklearn.cluster import DBSCAN

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # w, h

def scale_boxes(data, w, h):
    """Convert 0-1000 normalized to absolute pixels"""
    def scale(box):
        x1, y1, x2, y2 = box
        return [
            int(x1 * w / 1000),
            int(y1 * h / 1000),
            int(x2 * w / 1000),
            int(y2 * h / 1000)
        ]
    if "hindi" in data:
        data["hindi"] = scale(data["hindi"])
    if "english" in data:
        data["english"] = scale(data["english"])
    return data

def run_openrouter(image_path, model, api_key, examples=[], layout="Top-Bottom"):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    system_prompt = "You are an expert in Indian bilingual exam papers. Find the Hindi (Devanagari script) region and English region. Return ONLY JSON: {"hindi":[x1,y1,x2,y2],"english":[x1,y1,x2,y2],"confidence":0.0-1.0}. Use 0-1000 normalized coordinates. Hindi is usually on top or left."
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for ex in examples[-3:]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex['image_b64']}"}},
                {"type": "text", "text": f"Example: {json.dumps(ex['boxes'])}"}
            ]
        })
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Layout: {layout}. Analyze and return boxes."}
        ]
    })
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.1
        )
        data = json.loads(resp.choices[0].message.content)
        w, h = get_image_size(image_path)
        data = scale_boxes(data, w, h)
        data["confidence"] = float(data.get("confidence", 0.85))
        return data, "openrouter"
    except Exception as e:
        return None, f"openrouter_error: {str(e)[:100]}"

def paddle_fallback(image_path, layout):
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=False, show_log=False)
        result = ocr.ocr(image_path, cls=False)
        
        if not result or not result[0]:
            return None
            
        boxes = []
        for line in result[0]:
            pts = line[0]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        
        if len(boxes) < 4:
            return None
            
        # Cluster by position
        centers = np.array([[ (b[0]+b[2])/2, (b[1]+b[3])/2 ] for b in boxes])
        coord = centers[:,1].reshape(-1,1) if layout == "Top-Bottom" else centers[:,0].reshape(-1,1)
        
        clustering = DBSCAN(eps=80, min_samples=2).fit(coord)
        labels = clustering.labels_
        
        unique_labels = [l for l in set(labels) if l != -1]
        if len(unique_labels) < 2:
            return None
        
        # Get two largest clusters
        clusters = []
        for lbl in unique_labels:
            cluster_boxes = [boxes[i] for i, l in enumerate(labels) if l == lbl]
            x1 = min(b[0] for b in cluster_boxes)
            y1 = min(b[1] for b in cluster_boxes)
            x2 = max(b[2] for b in cluster_boxes)
            y2 = max(b[3] for b in cluster_boxes)
            clusters.append([x1, y1, x2, y2, len(cluster_boxes)])
        
        clusters.sort(key=lambda x: x[4], reverse=True)
        clusters = clusters[:2]
        
        # Sort by position (top then bottom, or left then right)
        if layout == "Top-Bottom":
            clusters.sort(key=lambda x: x[1])
        else:
            clusters.sort(key=lambda x: x[0])
        
        return {
            "hindi": clusters[0][:4],
            "english": clusters[1][:4],
            "confidence": 0.65
        }
    except Exception:
        return None

def geometric_split(image_path, layout):
    w, h = get_image_size(image_path)
    if layout == "Top-Bottom":
        return {"hindi": [0, 0, w, h//2], "english": [0, h//2, w, h], "confidence": 0.3}
    else:
        return {"hindi": [0, 0, w//2, h], "english": [w//2, 0, w, h], "confidence": 0.3}

def run_inference(image_path, model, api_key, examples=[], layout="Top-Bottom"):
    # 1. OpenRouter
    data, source = run_openrouter(image_path, model, api_key, examples, layout)
    if data and "hindi" in data and "english" in data:
        data["source"] = source
        return data
    
    # 2. PaddleOCR fallback
    paddle_data = paddle_fallback(image_path, layout)
    if paddle_data:
        paddle_data["source"] = "paddleocr"
        return paddle_data
    
    # 3. Geometric
    geo = geometric_split(image_path, layout)
    geo["source"] = "geometric"
    return geo
