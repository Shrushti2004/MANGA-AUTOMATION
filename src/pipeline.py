import argparse
from datetime import datetime
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from lib.layout.layout import generate_layout, similar_layouts
from lib.script.divide import divide_script, ele2panels, richfy_panel, refine_elements
from lib.image.image import generate_images
from lib.image.controlnet import detect_human, check_open, controlnet2bboxes
from lib.name.name import generate_name

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline script for processing')
    parser.add_argument('--script_path', help='Path to the script file')
    parser.add_argument('--output_path', help='Path for output')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    if not check_open():
        raise ValueError("ControlNet is not running")
    date = datetime.now().strftime("%Y%m%d_%H%M")
    image_base_dir = os.path.join(args.output_path, date, "images")
    if not os.path.exists(image_base_dir):
        os.makedirs(image_base_dir)
    name_base_dir = os.path.join(args.output_path, date, "name")
    if not os.path.exists(name_base_dir):
        os.makedirs(name_base_dir)

    client = OpenAI()
    elements = divide_script(client, args.script_path, args.output_path)
    elements = refine_elements(elements, args.output_path)
    panels = ele2panels(client, elements, args.output_path)
    richfied_panels = richfy_panel(client, panels, args.output_path)
    created_date, image_dir = generate_images(client, richfied_panels, args.output_path)
    date_dirs = [d for d in os.listdir(args.output_path) if os.path.isdir(os.path.join(args.output_path, d))]
    date_dirs.sort(key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
    image_dir = os.path.join(args.output_path, date_dirs[-1], "images")
    name_dir = os.path.join(args.output_path, date_dirs[-1], "name")
    controlnet_result_dict = detect_human(image_dir)
    name_panel_dir = os.path.join(name_dir, f"panel{idx}")
    os.makedirs(name_panel_dir, exist_ok=True)
    for i, controlnet_result in enumerate(controlnet_results):
        width, height = controlnet_result.canvas_width, controlnet_result.canvas_height
        bboxes = controlnet2bboxes(controlnet_result)
        layout = generate_layout(bboxes, richfied_panels[idx], width, height)
        scored_layouts = similar_layouts(layout)
        name_path = os.path.join(name_dir, f"name{i}.png")
        generate_name(controlnet_result, layout, scored_layouts, richfied_panels[idx], name_path)

if __name__ == "__main__":
    main()