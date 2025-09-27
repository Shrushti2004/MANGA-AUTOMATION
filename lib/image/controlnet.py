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
        self.pose_keypoints_2d :Optional[List[Tuple[float, float]]]= None
        self.face_keypoints_2d:Optional[List[Tuple[float, float]]] = None
        self.hand_left_keypoints_2d:Optional[List[Tuple[float, float]]] = None
        self.hand_right_keypoints_2d:Optional[List[Tuple[float, float]]] = None

    def __str__(self):
        return f"pose_keypoints_2d={self.pose_keypoints_2d}\n" \
                f"face_keypoints_2d={self.face_keypoints_2d}\n" \
                f"hand_left_keyopints_2d={self.hand_left_keypoints_2d}\n"  \
                f"hand_right_keyoints_2d={self.hand_right_keypoints_2d})"


class ControlNetResult:
    def __init__(self, json_response, base_image_path=""):
        self.canvas_height:int = None
        self.canvas_width:int = None
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
            if keypoints == None:
                return None
            return [
                (keypoints[i], keypoints[i + 1])
                for i in range(0, len(keypoints), 3)
            ]

        num_people = len(json_response["poses"][0]["people"])
        for i in range(num_people):
            person = People()
            person.pose_keypoints_2d = parse_keypoints(json_response["poses"][0]["people"][i]["pose_keypoints_2d"])
            person.face_keypoints_2d = parse_keypoints(json_response["poses"][0]["people"][i]["face_keypoints_2d"])
            person.hand_left_keyopints_2d = parse_keypoints(json_response["poses"][0]["people"][i]["hand_left_keypoints_2d"])
            person.hand_right_keyoints_2d = parse_keypoints(json_response["poses"][0]["people"][i]["hand_right_keypoints_2d"])
            self.people.append(person)

def controlnet2bboxes(controlnet_results: ControlNetResult):
    width, height = controlnet_results.canvas_width, controlnet_results.canvas_height
    bboxes = []
    for person in controlnet_results.people:
        bbox = [max(width, height)] * 4
        for keypoint in person.pose_keypoints_2d:
            x, y = keypoint
            bbox[0] = min(bbox[0], x * width)
            bbox[1] = min(bbox[1], y * height)
            bbox[2] = max(bbox[2], x * width)
            bbox[3] = max(bbox[3], y * height)
        bboxes.append(bbox)
    bboxes = sorted(bboxes, key=lambda x: (x[0] + x[2]) / 2, reverse=True)
    return bboxes


def detect_human(image_dir):
    dict = {}
    for panel_name in tqdm(os.listdir(image_dir), desc="Processing Panels for detect human"):
        panel_dir = os.path.join(image_dir, panel_name)
        results = []
        for filename in os.listdir(panel_dir):
            filepath = os.path.join(panel_dir, filename)
            with open(filepath, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "controlnet_module": "openpose_full",
                "controlnet_input_images": [img_data],
            }

            response = requests.post("http://127.0.0.1:7860/controlnet/detect", json=payload).json()
            with open("response.txt", "w") as f:
                json.dump(response, f, indent=2)

            contrlnet_res = ControlNetResult(response, filepath)
            results.append(contrlnet_res)
        dict[panel_name] = results
    return dict

def run_controlnet_openpose(image_path, controlnetres_image_path=None,output_path=None):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "controlnet_module": "openpose_full",
        "controlnet_input_images": [img_data],
    }

    response = requests.post("http://127.0.0.1:7860/controlnet/detect", json=payload).json()
    if output_path is not None:
        with open(os.path.join(output_path, "response.json"), "w") as f:
            json.dump(response, f, indent=2)

    return ControlNetResult(response, controlnetres_image_path)


def generate_with_controlnet_openpose(pose_image_path, prompt, save_path, model="T-anime-v4"):
    with open(pose_image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "prompt": prompt,
        "override_settings" : {
            "sd_model_checkpoint" : model,
        },
        "width": 512,
        "height": 512,
        "alwayson_scripts": {
            "controlnet": {
                "args": [{
                    "image": img_data,
                    "module": "openpose",
                    "model": "control_v11p_sd15_openpose",
                    "enabled": True,
                    "control_mode": "ControlNet is more important",
                }]
            }
        }
    }
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload).json()
    image_base64 = response["images"][0]
    image_bytes = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    image.save(save_path)
    return


def check_open():
    url = "http://127.0.0.1:7860"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False
