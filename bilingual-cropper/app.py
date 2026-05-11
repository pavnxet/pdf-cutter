import streamlit as st
import os, shutil, traceback
from core.pdf_processor import pdf_to_images, images_to_pdf
from core.infer import run_inference
from core.cropper import crop_with_padding
from core.few_shot import save_correction, get_latest_examples
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Bilingual Exam Cropper", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Bilingual Exam Cropper - OpenRouter Ready")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", 
                           value=os.getenv("OPENROUTER_API_KEY", ""),
                           type="password", 
                           help="Get from openrouter.ai/keys")
    
    default_model = os.getenv("OPENROUTER_MODEL", "openrouter/owl-alpha")
    model_options = [
        "openrouter/owl-alpha",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-1.5-pro"
    ]
    if default_model not in model_options:
        model_options.insert(0, default_model)
        
    model = st.selectbox("Vision Model", 
                        model_options,
                        index=model_options.index(default_model) if default_model in model_options else 0,
                        help="owl-alpha is fastest and cheapest")
    layout = st.radio("Page Layout", ["Top-Bottom", "Left-Right"], help="How is your exam split?")
    
    st.divider()
    examples = get_latest_examples(10)
    st.metric("Learned Examples", len(examples))
    if st.button("Clear Learning Memory"):
        if os.path.exists("data/examples/examples.json"):
            os.remove("data/examples/examples.json")
        st.success("Memory cleared!")

st.markdown("Upload a bilingual Hindi-English exam PDF. The AI will auto-crop into two separate PDFs.")

uploaded = st.file_uploader("Choose PDF", type=["pdf"], help="Scanned exam papers work best at 300 DPI")

if uploaded:
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar")
        st.stop()
    
    os.makedirs("data/uploads", exist_ok=True)
    pdf_path = f"data/uploads/{uploaded.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())
    
    col1, col2 = st.columns([1,3])
    with col1:
        process_btn = st.button("🚀 Process PDF", type="primary", use_container_width=True)
    with col2:
        st.caption(f"File: {uploaded.name} ({uploaded.size // 1024} KB)")
    
    if process_btn:
        try:
            with st.status("Processing...", expanded=True) as status:
                st.write("📄 Converting PDF to images...")
                images = pdf_to_images(pdf_path, dpi=300)
                st.write(f"✓ Converted {len(images)} pages")
                
                examples = get_latest_examples(3)
                st.write(f"🧠 Using {len(examples)} learned examples")
                
                results = []
                progress = st.progress(0, text="Analyzing pages...")
                
                for i, img_path in enumerate(images):
                    st.write(f"Analyzing page {i+1}...")
                    try:
                        data = run_inference(img_path, model, api_key, examples, layout)
                        results.append({"path": img_path, "data": data, "page": i+1})
                    except Exception as e:
                        st.error(f"Page {i+1} failed: {str(e)}")
                        # Fallback to geometric
                        from core.infer import geometric_split
                        data = geometric_split(img_path, layout)
                        data["source"] = "error_fallback"
                        results.append({"path": img_path, "data": data, "page": i+1})
                    
                    progress.progress((i+1)/len(images), text=f"Page {i+1}/{len(images)}")
                
                st.session_state["results"] = results
                st.session_state["images"] = images
                st.session_state["pdf_name"] = uploaded.name
                status.update(label="✅ Processing complete!", state="complete")
        
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.code(traceback.format_exc())

if "results" in st.session_state:
    st.divider()
    st.subheader("📊 Review Results")
    
    # Build review table
    table_data = []
    for r in st.session_state["results"]:
        d = r["data"]
        conf = d.get("confidence", 0.5)
        source = d.get("source", "unknown")
        
        if source == "openrouter" and conf > 0.75:
            status = "✅ Good"
        elif source == "openrouter":
            status = "⚠️ Review"
        elif source == "paddleocr":
            status = "🔧 Auto-detected"
        else:
            status = "✂️ Geometric split"
        
        table_data.append({
            "Page": r["page"],
            "Hindi Box": f"{d['hindi'][2]-d['hindi'][0]}x{d['hindi'][3]-d['hindi'][1]}",
            "English Box": f"{d['english'][2]-d['english'][0]}x{d['english'][3]-d['english'][1]}",
            "Confidence": f"{conf:.2f}",
            "Method": source,
            "Status": status
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Correction interface
    st.divider()
    st.subheader("✏️ Correct a Page (teaches the AI)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        page_idx = st.selectbox(
            "Select page to correct",
            range(len(st.session_state["results"])),
            format_func=lambda x: f"Page {x+1} - {table_data[x]['Status']}"
        )
        st.caption("Draw RED box for Hindi, then BLUE box for English")
        st.caption("Draw exactly 2 rectangles")
        
        if st.button("💾 Save Correction", type="secondary"):
            if "canvas_data" in st.session_state:
                canvas_data = st.session_state["canvas_data"]
                if canvas_data and len(canvas_data.get("objects", [])) >= 2:
                    objs = canvas_data["objects"][:2]
                    img_path = st.session_state["images"][page_idx]
                    img = Image.open(img_path)
                    scale_x = img.width / 800  # canvas is scaled
                    scale_y = img.height / 600
                    
                    hindi_box = [
                        int(objs[0]["left"] * scale_x),
                        int(objs[0]["top"] * scale_y),
                        int((objs[0]["left"] + objs[0]["width"]) * scale_x),
                        int((objs[0]["top"] + objs[0]["height"]) * scale_y)
                    ]
                    english_box = [
                        int(objs[1]["left"] * scale_x),
                        int(objs[1]["top"] * scale_y),
                        int((objs[1]["left"] + objs[1]["width"]) * scale_x),
                        int((objs[1]["top"] + objs[1]["height"]) * scale_y)
                    ]
                    
                    count = save_correction(img_path, hindi_box, english_box)
                    st.success(f"✅ Saved! AI now has {count} examples. Re-process to use it.")
                    st.balloons()
    
    with col2:
        img_path = st.session_state["images"][page_idx]
        img = Image.open(img_path)
        # Resize for canvas
        canvas_height = 600
        canvas_width = int(img.width * canvas_height / img.height)
        if canvas_width > 800:
            canvas_width = 800
            canvas_height = int(img.height * 800 / img.width)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=img.resize((canvas_width, canvas_height)),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="canvas"
        )
        if canvas_result.json_data:
            st.session_state["canvas_data"] = canvas_result.json_data
    
    # Generate outputs
    st.divider()
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔨 Generate PDFs", type="primary", use_container_width=True):
            with st.spinner("Cropping pages..."):
                hindi_crops = []
                english_crops = []
                
                os.makedirs("data/crops_hindi", exist_ok=True)
                os.makedirs("data/crops_english", exist_ok=True)
                
                for r in st.session_state["results"]:
                    d = r["data"]
                    img_path = r["path"]
                    page_num = r["page"]
                    
                    try:
                        h_crop = crop_with_padding(img_path, d["hindi"], padding=5)
                        e_crop = crop_with_padding(img_path, d["english"], padding=5)
                        
                        h_path = f"data/crops_hindi/page_{page_num:03d}.png"
                        e_path = f"data/crops_english/page_{page_num:03d}.png"
                        
                        h_crop.save(h_path, "PNG")
                        e_crop.save(e_path, "PNG")
                        
                        hindi_crops.append(h_path)
                        english_crops.append(e_path)
                    except Exception as e:
                        st.error(f"Failed to crop page {page_num}: {e}")
                
                if hindi_crops:
                    images_to_pdf(hindi_crops, "data/hindi_output.pdf")
                    images_to_pdf(english_crops, "data/english_output.pdf")
                    st.session_state["ready"] = True
                    st.success(f"✅ Created {len(hindi_crops)} pages each")
    
    with col2:
        if st.session_state.get("ready"):
            with open("data/hindi_output.pdf", "rb") as f:
                st.download_button(
                    "📘 Download Hindi PDF",
                    f,
                    file_name=f"{st.session_state['pdf_name'].replace('.pdf','')}_hindi.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with col3:
        if st.session_state.get("ready"):
            with open("data/english_output.pdf", "rb") as f:
                st.download_button(
                    "📗 Download English PDF",
                    f,
                    file_name=f"{st.session_state['pdf_name'].replace('.pdf','')}_english.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    st.divider()
    if st.button("🗑️ Clear Session & Delete Files", help="Removes uploaded PDF and crops (keeps learned examples)"):
        for folder in ["data/uploads", "data/crops_hindi", "data/crops_english"]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        for f in ["data/hindi_output.pdf", "data/english_output.pdf"]:
            if os.path.exists(f):
                os.remove(f)
        st.session_state.clear()
        st.rerun()

# Footer
st.divider()
st.caption("Bilingual Exam Cropper v1.0 | Powered by OpenRouter | Made for non-developers")
