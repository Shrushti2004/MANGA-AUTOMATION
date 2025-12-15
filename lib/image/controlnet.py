from typing import List, Optional, Tuple
import io
from tqdm import tqdm
import os
import base64
import json
import requests
from PIL import Image

class People:
    def __init__(self):
        self.pose_keypoints_2d: Optional[List[Tuple[float, float]]] = None
        self.face_keypoints_2d: Optional[List[Tuple[float, float]]] = None
        self.hand_left_keypoints_2d: Optional[List[Tuple[float, float]]] = None
        self.hand_right_keypoints_2d: Optional[List[Tuple[float, float]]] = None

    def __str__(self):
        return f"pose_keypoints_2d={self.pose_keypoints_2d}\n" \
               f"face_keypoints_2d={self.face_keypoints_2d}\n" \
               f"hand_left_keypoints_2d={self.hand_left_keypoints_2d}\n" \
               f"hand_right_keypoints_2d={self.hand_right_keypoints_2d}"

class ControlNetResult:
    def __init__(self, json_response, base_image_path=""):
        self.canvas_height: int = None
        self.canvas_width: int = None
        self.image: Image = None
        self.people: List[People] = []
        self.base_image_path = base_image_path
        self._parse_response(json_response)

    def _parse_response(self, json_response):
        self.canvas_height = json_response["poses"][0]["canvas_height"]
        self.canvas_width = json_response["poses"][0]["canvas_width"]

        image_base64 = json_response["images"][0]
        image_bytes = io.BytesIO(base64.b64decode(image_base64))
        self.image = Image.open(image_bytes)

        def parse_keypoints(keypoints):
            if keypoints is None:
                return []
            return [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3)]

        num_people = len(json_response["poses"][0]["people"])
        for i in range(num_people):
            person = People()
            p = json_response["poses"][0]["people"][i]
            person.pose_keypoints_2d = parse_keypoints(p.get("pose_keypoints_2d"))
            person.face_keypoints_2d = parse_keypoints(p.get("face_keypoints_2d"))
            person.hand_left_keypoints_2d = parse_keypoints(p.get("hand_left_keypoints_2d"))
            person.hand_right_keypoints_2d = parse_keypoints(p.get("hand_right_keypoints_2d"))
            self.people.append(person)

def controlnet2bboxes(controlnet_results: ControlNetResult):
    width, height = controlnet_results.canvas_width, controlnet_results.canvas_height
    raw_bboxes = []

    for person in controlnet_results.people:
        if not person.pose_keypoints_2d:
            continue
        x1, y1 = float('inf'), float('inf')
        x2, y2 = 0, 0
        for keypoint in person.pose_keypoints_2d:
            if keypoint[0] == 0 and keypoint[1] == 0:
                continue
            x1 = min(x1, keypoint[0])
            y1 = min(y1, keypoint[1])
            x2 = max(x2, keypoint[0])
            y2 = max(y2, keypoint[1])
        raw_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    # Clean invalid boxes
    cleaned = []
    for b in raw_bboxes:
        x1, y1, x2, y2 = b
        if x1 == float('inf') or y1 == float('inf'):
            continue
        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        cleaned.append(b)

    # Fallback
    if not cleaned:
        print("[WARN] No valid bboxes detected – using fallback bbox")
        cleaned = [[width // 3, height // 3, width // 2, height // 2]]

    return cleaned

def detect_human(image_dir):
    results_dict = {}
    for panel_name in tqdm(os.listdir(image_dir), desc="Processing Panels for detect human"):
        panel_dir = os.path.join(image_dir, panel_name)
        results = []
        for filename in os.listdir(panel_dir):
            filepath = os.path.join(panel_dir, filename)
            with open(filepath, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            payload = {"controlnet_module": "openpose_full", "controlnet_input_images": [img_data]}
            response = requests.post("http://127.0.0.1:7860/controlnet/detect", json=payload).json()
            contrlnet_res = ControlNetResult(response, filepath)
            results.append(contrlnet_res)
        results_dict[panel_name] = results
    return results_dict

def run_controlnet_openpose(image_path, controlnetres_image_path=None, output_path=None):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    payload = {"controlnet_module": "openpose_full", "controlnet_input_images": [img_data]}
    response = requests.post("http://127.0.0.1:7860/controlnet/detect", json=payload).json()
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "response.json"), "w") as f:
            json.dump(response, f, indent=2)
    return ControlNetResult(response, controlnetres_image_path)

def generate_with_controlnet_openpose(pose_image_path, prompt, save_path, model="tAnimeV4Pruned_v20", width=512, height=512):
    negative_prompt = (
        "nsfw, (photorealistic:1.5), (color:1.5), (shading:1.4), (smooth:1.4), 3d, render,non-overlapping "
    )
    with open(pose_image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "override_settings": {"sd_model_checkpoint": model},
        "sampler_name": "Euler a",
        "scheduler": "Automatic",
        "steps": 20,
        "cfg_scale": 4,
        "width": width,
        "height": height,
        "seed": -1,
        "alwayson_scripts": {
            "controlnet": {
                "args": [{
                    "image": img_data,
                    "module": "scribble_pidinet",
                    "model": "control_v11p_sd15_scribble",
                    "enabled": True,
                    "weight": 1,
                    "guidance_end": 0.8,
                }]
            }
        }
    }
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload).json()
    image_base64 = response["images"][0]
    image_bytes = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    image.save(save_path)

def check_open():
    url = "http://127.0.0.1:7860"
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
