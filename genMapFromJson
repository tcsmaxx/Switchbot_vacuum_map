import json
import base64
import lz4.block
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter
import base64 as b64

# ----------------------------
# Einstellungen
# ----------------------------
SCALE = 4
ORIENTATION_OPS = ["flip_tb", "rot180"]

def normalize_payload(obj: dict) -> dict:
    if not isinstance(obj, dict): return {}
    if "data" in obj and isinstance(obj["data"], dict):
        return obj["data"]
    return obj

def b64_name(s: str) -> str:
    try:
        return b64.b64decode(s).decode("utf-8", errors="ignore").strip()
    except Exception:
        return s or ""

def clamp_pts(pts, w, h):
    # Filtert None-Werte aus, falls to_px fehlschlägt
    safe_pts = [p for p in pts if p is not None]
    return [(max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))) for x, y in safe_pts]

def poly_centroid(pts):
    if len(pts) < 3: return pts[0] if pts else (0, 0)
    x, y = np.array([p[0] for p in pts], dtype=np.float64), np.array([p[1] for p in pts], dtype=np.float64)
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    a = (x * y2 - x2 * y)
    A = a.sum() / 2.0
    if abs(A) < 1e-6: return int(x.mean()), int(y.mean())
    cx = ((x + x2) * a).sum() / (6.0 * A)
    cy = ((y + y2) * a).sum() / (6.0 * A)
    return int(cx), int(cy)

def wall_outline_thin(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    up, dn = np.roll(m, -1, axis=0), np.roll(m, 1, axis=0)
    lf, rt = np.roll(m, -1, axis=1), np.roll(m, 1, axis=1)
    return m & (~(m & up & dn & lf & rt))

def get_font(size: int):
    for fn in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "tahoma.ttf"):
        try: return ImageFont.truetype(fn, size)
        except: pass
    return ImageFont.load_default()

def apply_orientation_to_image(img: Image.Image, ops):
    for op in ops:
        if op == "flip_tb": img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif op == "flip_lr": img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == "rot180": img = img.transpose(Image.ROTATE_180)
    return img

def transform_point(x, y, w, h, ops):
    for op in ops:
        if op == "flip_tb": y = (h - 1) - y
        elif op == "flip_lr": x = (w - 1) - x
        elif op == "rot180": x, y = (w - 1) - x, (h - 1) - y
    return x, y

def draw_pill_transparent(draw, cx, cy, text, canvas_w, scale):
    font_size = int(5.5 * scale)
    pad_x, pad_y = int(4.0 * scale), int(2.5 * scale)
    radius = int(4.0 * scale)
    
    pill_fill = (45, 45, 45, 140)
    text_fill = (255, 255, 255, 245)
    
    font = get_font(font_size)
    
    # Text-Dimensionen  berechnen
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # Pill-Größe basierend auf Textmaß
    pill_w = tw + 2 * pad_x
    pill_h = th + 2 * pad_y
    
    rx0, ry0 = cx - pill_w // 2, cy - pill_h // 2
    rx1, ry1 = rx0 + pill_w, ry0 + pill_h
    
    # 1. Die Pill zeichnen
    draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=radius, fill=pill_fill)
    
    # 2. Den Text zeichnen  mit Anker "mm" - cx und cy direkt als Zentrum für den Anker
    draw.text((cx, cy), text, font=font, fill=text_fill, anchor="mm")

def draw_path_soft(base_rgba, pts, width_px=2, color=(255, 255, 255, 210), aa=3, blur=0.8):
    if not pts or len(pts) < 2: return base_rgba
    W, H = base_rgba.size
    hi = Image.new("RGBA", (W * aa, H * aa), (0, 0, 0, 0))
    ImageDraw.Draw(hi).line([(int(x * aa), int(y * aa)) for x, y in pts], fill=color, width=max(1, int(width_px * aa)))
    if blur > 0: hi = hi.filter(ImageFilter.GaussianBlur(radius=blur * aa / 3.0))
    return Image.alpha_composite(base_rgba, hi.resize((W, H), Image.LANCZOS))

def build_map_like_app(json_path="daten.json", out_path="map_like_app.png", scale=SCALE, ops=ORIENTATION_OPS):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = normalize_payload(json.load(f))
    except Exception as e:
        print(f"Fehler: {e}")
        return

    map_str = data.get("map", "")
    width = int(data.get("width", 281))
    res, x_min, y_min = float(data.get("resolution", 0.05)), float(data.get("x_min", -9.1)), float(data.get("y_min", -0.3))

    # Definition von to_px
    def to_px(x_mm, y_mm):
        px = int(((x_mm / 1000.0) - x_min) / res)
        py = int(((y_mm / 1000.0) - y_min) / res)
        return px, py

    # Dekomprimierung
    raw_bytes = base64.b64decode(map_str)
    dec = lz4.block.decompress(raw_bytes, uncompressed_size=width * 1500)
    height = len(dec) // width
    grid = np.frombuffer(dec[:width*height], dtype=np.uint8).reshape((height, width))

    # Base Layers
    base = Image.new("RGBA", (width, height), (252, 252, 252, 255))
    unk_layer = np.zeros((height, width, 4), dtype=np.uint8)
    unk_layer[grid == 127] = (245, 246, 248, 255)
    base = Image.alpha_composite(base, Image.fromarray(unk_layer, "RGBA"))

    # Zimmer
    room_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    floor_img = Image.fromarray(((grid == 255).astype(np.uint8) * 255), mode="L")
    ROOM_COLORS = [(125,195,245,200), (160,175,245,200), (255,215,110,200), (170,230,215,200), (240,165,140,200)]
    
    rooms = data.get("smartArea", {}).get("value", [])
    for i, room in enumerate(rooms):
        pts = clamp_pts([to_px(v[0], v[1]) for v in room.get("vertexs", [])], width, height)
        if len(pts) > 2:
            rm = Image.new("L", (width, height), 0)
            ImageDraw.Draw(rm).polygon(pts, fill=255)
            room_layer = Image.composite(Image.new("RGBA", (width, height), ROOM_COLORS[i % len(ROOM_COLORS)]), room_layer, ImageChops.multiply(rm, floor_img))
    base = Image.alpha_composite(base, room_layer)

    # Wände
    wall_img = np.zeros((height, width, 4), dtype=np.uint8)
    wall_img[wall_outline_thin(grid == 0)] = (40, 45, 60, 255)
    base = Image.alpha_composite(base, Image.fromarray(wall_img, "RGBA"))

    # Forbidden
    forbid_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fd = ImageDraw.Draw(forbid_overlay)
    for area in data.get("area", []):
        pts = clamp_pts([to_px(v[0], v[1]) for v in area.get("vertexs", [])], width, height)
        if len(pts) >= 3 and area.get("active") == "forbid":
            fd.polygon(pts, fill=(235, 80, 80, 60), outline=(170, 40, 40, 200))
        elif len(pts) >= 2:
            fd.line(pts, fill=(170, 40, 40, 200), width=1)
    base = Image.alpha_composite(base, forbid_overlay)

    # Transformation & Scale
    base = apply_orientation_to_image(base, ops)
    map_scaled = base.resize((width * scale, height * scale), Image.NEAREST)

    # Pfad
    pos_array = data.get("posArray", [])
    if pos_array:
        pts_path = []
        for p in pos_array:
            tx, ty = transform_point(*to_px(p[0], p[1]), width, height, ops)
            pts_path.append((tx * scale, ty * scale))
        map_scaled = draw_path_soft(map_scaled, pts_path)

    # Top Layer (Pills & Charger)
    top_layer = Image.new("RGBA", map_scaled.size, (0, 0, 0, 0))
    top_draw = ImageDraw.Draw(top_layer)

    for room in rooms:
        name = b64_name(room.get("name", ""))
        if name:
            cx, cy = poly_centroid(clamp_pts([to_px(v[0], v[1]) for v in room.get("vertexs", [])], width, height))
            tx, ty = transform_point(cx, cy, width, height, ops)
            draw_pill_transparent(top_draw, tx * scale, ty * scale, name, map_scaled.size[0], scale)

    cp = data.get("chargeHandlePos")
    if cp:
        tx, ty = transform_point(*to_px(cp[0], cp[1]), width, height, ops)
        r = 6 * scale
        top_draw.ellipse([tx*scale-r, ty*scale-r, tx*scale+r, ty*scale+r], fill=(160, 32, 240, 200), outline=(255,255,255,255), width=int(1.5*scale))

    Image.alpha_composite(map_scaled, top_layer).save(out_path)
    print(f"Datei gespeichert: {out_path}")
    Image.open(out_path).show()

if __name__ == "__main__":
    build_map_like_app()
