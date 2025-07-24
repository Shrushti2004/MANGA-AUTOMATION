from PIL import Image, ImageDraw, ImageFont
import os
from lib.layout.layout import MangaLayout, Speaker
from math import atan2, cos, sin, hypot
import random
from math import ceil

FONTPATH = "fonts/NotoSansCJK-Regular.ttc"


def controlnetres2pil(res):
    canvas = (res.canvas_width, res.canvas_height)
    w, h = canvas
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    default_palette = [
        ( 31, 119, 180),  # blue
        (255, 127,  14),  # orange
        ( 44, 160,  44),  # green
        (214,  39,  40),  # red
        (148, 103, 189),  # purple
        (140,  86,  75),  # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189,  34),  # olive
        ( 23, 190, 207),  # cyan
    ]

    def get_color(idx):
        if idx < len(default_palette):
            return default_palette[idx]
        random.seed(idx)
        return tuple(random.randint(0, 255) for _ in range(3))

    attr_names = [
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
    ]

    radius = 4
    for person_idx, person in enumerate(res.people):
        col = get_color(person_idx)
        for attr in attr_names:
            pts = getattr(person, attr, None)
            if not pts:
                continue
            for x, y in pts:
                if x == 0 and y == 0:
                    continue
                x, y = x * w, y * h  # 正規化→画面座標
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(bbox, fill=col)

    return img

def draw_bubble(img, bbox, type, point):
    base_width = 40
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    k = 2 ** 0.5
    a, b = (w / 2) * k, (h / 2) * k
    ellipse_box = (cx - a, cy - b, cx + a, cy + b)
    draw = ImageDraw.Draw(img)
    draw.ellipse(ellipse_box, fill="white", outline="black", width=2)
    tx, ty = point
    theta = atan2((ty - cy) / b, (tx - cx) / a)
    bx, by = cx + a * cos(theta), cy + b * sin(theta)
    tx_tan, ty_tan = -a * sin(theta), b * cos(theta)
    t_len = hypot(tx_tan, ty_tan)
    tx_tan, ty_tan = tx_tan / t_len, ty_tan / t_len
    half = base_width / 2
    lx, ly = bx + tx_tan * half, by + ty_tan * half
    rx, ry = bx - tx_tan * half, by - ty_tan * half

    nx, ny = (bx - cx), (by - cy)
    n_len = hypot(nx, ny)
    nx, ny = nx / n_len, ny / n_len
    lx, ly = lx - nx * 2, ly - ny * 2
    rx, ry = rx - nx * 2, ry - ny * 2
    draw.polygon([(lx, ly), (rx, ry), point], fill="white", outline="black", width=2)

def draw_vertical_text(img, text, bbox, type, point):
    FONTSIZE = 10
    draw = ImageDraw.Draw(img)
    if type == "dialogue":
        draw_bubble(img, bbox, type, point)
    else:
        gap = 2
        new_bbox = [bbox[0] - gap, bbox[1] - gap, bbox[2] + gap, bbox[3] + gap]
        draw.rectangle(new_bbox, fill="white", outline="black", width=2)

    font = ImageFont.truetype(FONTPATH, FONTSIZE)
    x1, y1, x2, y2 = bbox
    box_w, box_h   = x2 - x1, y2 - y1
    line_gap = 0
    col_gap = 2
    char_bbox = draw.textbbox((0, 0), "あ", font=font,
                              direction="ttb", features=["vert", "vrt2"])
    char_w  = char_bbox[2] - char_bbox[0]
    char_h  = char_bbox[3] - char_bbox[1]

    max_rows = max(1, (box_h + line_gap) // (char_h + line_gap))

    total_cols = ceil(len(text) / max_rows)
    needed_w   = total_cols * char_w + (total_cols - 1) * col_gap
    if needed_w > box_w:
        raise ValueError("フォントサイズが大き過ぎて bbox 内に収まりません")

    start_x = x2 - char_w
    step_x  = -(char_w + col_gap)

    start_y = y1

    col = 0
    for i, ch in enumerate(text):
        row = i % max_rows
        # 列が変わるとき
        if i and row == 0:
            col += 1

        x = start_x + step_x * col
        y = start_y + row * (char_h + line_gap)

        draw.text((x, y), ch,
                  fill="black",
                  font=font,
                  direction="ttb",
                  features=["vert", "vrt2"])
    return

def draw_speech(pil_image, text, text_info, type, point):
    TEXT_THRESHOLD = 10
    for text_info_single in text_info:
        text_len = len(text)
        if text_len - text_info_single["length"] > TEXT_THRESHOLD:
            draw_vertical_text(pil_image, text[:text_info_single["length"]], text_info_single["bbox"], type, point)
            text = text[text_info_single["length"]:]
        else:
            draw_vertical_text(pil_image, text, text_info_single["bbox"], type, point)
    return



def generate_name(controlnet_result, base_layout, scored_layouts, panel, name_path):
    width, height = controlnet_result.canvas_width, controlnet_result.canvas_height
    pil_image_original = controlnetres2pil(controlnet_result)
    total_monologue = ""
    seen_idx = []
    for ref_layout, score, pairs in scored_layouts[-1:]:
        pil_image = pil_image_original.copy()
        for ele in panel:
            if ele["type"] == "description":
                continue
            if ele["type"] == "monologue":
                total_monologue += ele["content"]
                continue
            if ele["type"] == "dialogue":
                for idx, layout_ele in enumerate(base_layout.elements):
                    if type(layout_ele) == Speaker:
                        if idx not in seen_idx:
                            seen_idx.append(idx)
                            for pair in pairs:
                                if pair[0] == idx:
                                    ref_elem = ref_layout.elements[pair[1]]
                                    point_x = (layout_ele.bbox[0] + layout_ele.bbox[2]) / 2
                                    point_y = layout_ele.bbox[1]
                                    draw_speech(pil_image, ele["content"], ref_elem.text_info, "dialogue" , (point_x, point_y))
                                    # draw_speaker(pil_image, layout_ele.bbox, ele["speaker"])
        draw_speech(pil_image, total_monologue, ref_layout.unrelated_text_bbox, "monologue", (0, 0))
        pil_image.save(name_path)

    