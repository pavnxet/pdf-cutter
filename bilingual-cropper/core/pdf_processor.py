import os
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

def check_poppler():
    try:
        convert_from_path
        return True
    except Exception:
        return False

def pdf_to_images(pdf_path, dpi=300, output_folder="data/uploads"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"Poppler not found. Install poppler-utils. Error: {e}")
    image_paths = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"{base_name}_page_{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths

def images_to_pdf(image_paths, output_path):
    if not image_paths:
        raise ValueError("No images to convert")
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(image_paths))
    return output_path
