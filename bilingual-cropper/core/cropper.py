from PIL import Image

def crop_with_padding(image_path, box, padding=5):
    img = Image.open(image_path)
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))
    x2 = min(w, int(x2 + padding))
    y2 = min(h, int(y2 + padding))
    return img.crop((x1, y1, x2, y2))
