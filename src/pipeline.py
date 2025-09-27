import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from lib.layout.layout import generate_layout, similar_layouts
from lib.script.divide import divide_script, ele2panels, refine_elements
from lib.image.image import generate_image, generate_image_prompts, enhance_prompts, generate_image_with_sd
from lib.image.controlnet import detect_human, check_open, controlnet2bboxes, run_controlnet_openpose
from lib.name.name import generate_name, generate_animepose_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline script for processing')
    parser.add_argument('--script_path', help='Path to the script file')
    parser.add_argument('--output_path', help='Path for output')
    parser.add_argument('--resume_latest',
                        action='store_true', help='Debug mode')
    args = parser.parse_args()
    return args


def main():
    NUM_REFERENCES = 5
    args = parse_args()
    resume_latest = args.resume_latest
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    if not check_open():
        raise ValueError("ControlNet is not running")
    if resume_latest:
        date_dirs = [d for d in os.listdir(args.output_path) if os.path.isdir(
            os.path.join(args.output_path, d))]
        date_dirs.sort(key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
        date = date_dirs[-1]
        base_dir = os.path.join(args.output_path, date)
        image_base_dir = os.path.join(base_dir, "images")
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join(args.output_path, date)
        image_base_dir = os.path.join(base_dir, "images")
        if not os.path.exists(image_base_dir):
            os.makedirs(image_base_dir)

    client = OpenAI()
    print("Dividing script...")
    elements = divide_script(client, args.script_path, base_dir)
    print("Refining elements...")
    elements = refine_elements(elements, base_dir)
    print("Detected Speakers:")
    speakers = set()
    for element in elements:
        speakers.add(element["speaker"])
    speakers = list(speakers - {""})
    print(speakers)
    print("Converting elements to panels...")
    panels = ele2panels(client, elements, base_dir)
    print("Generating image prompts...")
    prompts = generate_image_prompts(client, panels, speakers, base_dir)
    print("Enhancing prompts...")
    prompts = enhance_prompts(client, prompts, base_dir)
    num_images = 3
    for i, prompt in tqdm(enumerate(prompts), desc="Generating names"):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)
        for j in range(num_images):
            image_path = os.path.join(panel_dir, f"{j:02d}.png")
            # generate_image(client, prompt, image_path)
            generate_image_with_sd(prompt, image_path)
            if not os.path.exists(image_path):
                continue
            anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
            generate_animepose_image(image_path, prompt, anime_image_path)
            openpose_result = run_controlnet_openpose(
                image_path, anime_image_path)
            width, height = openpose_result.canvas_width, openpose_result.canvas_height
            bboxes = controlnet2bboxes(openpose_result)
            layout = generate_layout(bboxes, panels[i], width, height)
            if layout is None:
                continue
            scored_layouts = similar_layouts(layout)
            for idx_ref_layout, scored_layout in enumerate(scored_layouts[:NUM_REFERENCES]):
                print(scored_layout[0].image_path)
                save_path = os.path.join(
                    panel_dir, f"{j:02d}_name_{idx_ref_layout:1d}.png")
                generate_name(openpose_result, layout,
                              scored_layout, panels[i], save_path)


if __name__ == "__main__":
    main()
