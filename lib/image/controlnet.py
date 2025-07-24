from typing import List, Optional, Tuple
import io
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
    def __init__(self, json_response):
        self.canvas_height:int = None
        self.canvas_width:int = None
        self.image: Image = None
        self.people: List[People] = []
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
    if image_dir is None:
        raise ValueError("image_dir is not set")

    dict = {}

    for panel_dir in os.listdir(image_dir):
        panel_dir = os.path.join(image_dir, panel_dir)
        results = []
        for filepath in os.listdir(panel_dir):
            filepath = os.path.join(panel_dir, filepath)
            with open(filepath, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "controlnet_module": "openpose_full",
                "controlnet_input_images": [img_data],
            }

            response = requests.post("http://127.0.0.1:7860/controlnet/detect", json=payload).json()
            with open("response.txt", "w") as f:
                json.dump(response, f, indent=2)

            contrlnet_res = ControlNetResult(response)
            results.append(contrlnet_res)
        dict[panel_dir] = results
    return dict


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
