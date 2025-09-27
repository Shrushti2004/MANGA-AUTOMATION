import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from lib.layout.score import calc_similarity

class Element:
    def __init__(self, bbox: List[int]):
        self.bbox = bbox

    def get_area(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

class Speaker(Element):
    def __init__(self, bbox: List[int], text_length: int, text_info=None):
        super().__init__(bbox)
        self.text_length = text_length
        self.text_info = text_info
    
    def __repr__(self):
        return f'Speaker(bbox: {self.bbox}, text_length: {self.text_length})'

class NonSpeaker(Element):
    def __init__(self, bbox: List[int]):
        super().__init__(bbox)

    def __repr__(self):
        return f'NonSpeaker(bbox: {self.bbox})'

class MangaLayout:
    def __init__(self, image_path: str, width: int, height: int, elements: List[Element], unrelated_text_length: int, unrelated_text_bbox: List[Dict[str, int]]):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.elements = elements
        self.unrelated_text_length = unrelated_text_length
        self.unrelated_text_bbox = unrelated_text_bbox

    def adjust(self, base_width: int, base_height: int):
        original_width = self.width
        original_height = self.height
        target_aspect_ratio = base_width / base_height
        original_aspect_ratio = original_width / original_height
        
        if original_aspect_ratio > target_aspect_ratio:
            new_width = original_width
            new_height = int(original_width / target_aspect_ratio)
        else:
            new_height = original_height
            new_width = int(original_height * target_aspect_ratio)
        
        width_scale_1 = new_width / original_width
        height_scale_1 = new_height / original_height
        
        final_width_scale = base_width / new_width
        final_height_scale = base_height / new_height
        
        total_width_scale = width_scale_1 * final_width_scale
        total_height_scale = height_scale_1 * final_height_scale
        
        
        self.width = base_width
        self.height = base_height
        
        for element in self.elements:
            element.bbox = [
                int(element.bbox[0] * total_width_scale), 
                int(element.bbox[1] * total_height_scale),
                int(element.bbox[2] * total_width_scale),
                int(element.bbox[3] * total_height_scale)
            ]
            if type(element) == Speaker:
                for text_info in element.text_info:
                    text_info["bbox"] = [
                        int(text_info["bbox"][0] * total_width_scale),
                        int(text_info["bbox"][1] * total_height_scale),
                        int(text_info["bbox"][2] * total_width_scale),
                        int(text_info["bbox"][3] * total_height_scale)
                    ]
        for text_info in self.unrelated_text_bbox:
            text_info["bbox"] = [
                int(text_info["bbox"][0] * total_width_scale),
                int(text_info["bbox"][1] * total_height_scale),
                int(text_info["bbox"][2] * total_width_scale),
                int(text_info["bbox"][3] * total_height_scale)
            ]

    def plot_data(self, ax):
        if self.image_path is not None and not self.image_path == "":
            image_path = os.path.join(os.getcwd(), os.path.normpath(self.image_path))
            img = Image.open(image_path)
            img = img.resize((self.width, self.height))
            ax.imshow(img)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for element in self.elements:
            x1, y1, x2, y2 = element.bbox
            if type(element) == Speaker:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            elif type(element) == NonSpeaker:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2))

    def __repr__(self):
        return f'''
        MangaLayout:
            image_path: {self.image_path}
            width: {self.width}
            height: {self.height}
            elements: {self.elements}
            unrelated_text_length: {self.unrelated_text_length}
        '''

def _generate_layout_from_metadata(metadata: dict):
    image_path = metadata["image_path"]
    width = metadata["width"]
    height = metadata["height"]
    speaker_objects = metadata["speaker_objects"]
    non_speaker_objects = metadata["non_speaker_objects"]
    unrelated_text_length = metadata["unrelated_text_length"]
    unrelated_text_bbox = metadata["unrelated_text_bbox"]
    elements = []
    for speaker_object in speaker_objects:
        elements.append(Speaker(speaker_object["bbox"], speaker_object["text_length"], speaker_object["text_info"]))
    for non_speaker_object in non_speaker_objects:
        elements.append(NonSpeaker(non_speaker_object["bbox"]))
    mangalayout = MangaLayout(image_path, width, height, elements, unrelated_text_length, unrelated_text_bbox)
    return mangalayout

def from_condition(annfile: str, num_speakers: int, num_non_speakers: int, base_text_length: int, text_length_threshold: int, base_width: int, base_height: int, aspect_ratio_threshold: float,  adjust: bool):
    '''
    話者の数の条件に一致するLayoutオブジェクトのリストを生成する

    Args:
        annfile: アノテーションファイルのパス
        num_speakers: 話者数
        num_non_speakers: 非話者数

    Returns:
        Layout: レイアウト
    '''

    if not os.path.exists(annfile):
        raise FileNotFoundError(f"Annotation file not found: {annfile}")

    if not os.path.isfile(annfile):
        raise IsADirectoryError(f"Annotation file is a directory: {annfile}")

    with open(annfile, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    key = f"{num_speakers}_{num_non_speakers}"

    if not key in ann:
        raise ValueError(f"Annotation file does not contain key: {key}")

    layouts = []
    base_aspect_ratio = base_width / base_height
    for metadata in ann[key]:
        unrelated_text_length = metadata['unrelated_text_length']
        if unrelated_text_length < base_text_length - text_length_threshold or unrelated_text_length > base_text_length + text_length_threshold:
            continue
        width, height = metadata['width'], metadata['height']
        aspect_ratio = width / height
        if aspect_ratio < base_aspect_ratio - aspect_ratio_threshold or aspect_ratio > base_aspect_ratio + aspect_ratio_threshold:
            continue
        layout = _generate_layout_from_metadata(metadata)
        if adjust:
            layout.adjust(base_width, base_height)
        layouts.append(layout)

    return layouts

def is_valid_layout(bboxes: List[List[int]], panel):
    speaker_count = 0
    for element in panel:
        if element["type"] == "dialogue":
            speaker_count += 1
    return speaker_count <= len(bboxes)

def generate_layout(bboxes: List[List[int]], panel, width, height):
    if not is_valid_layout(bboxes, panel):
        print("invalid layout")
        return None
    manga_elements = []
    unrelated_text_length = 0
    bbox_pointer = 0
    bbox_len = len(bboxes)
    for element in panel:
        if element["type"] == "dialogue":
            speaker = Speaker(bboxes[bbox_pointer], len(element["content"]))
            manga_elements.append(speaker)
            bbox_pointer += 1
        elif element["type"] == "monologue":
            unrelated_text_length += len(element["content"])

    while bbox_pointer < bbox_len:
        manga_elements.append(NonSpeaker(bboxes[bbox_pointer]))
        bbox_pointer += 1

    manga_layout = MangaLayout("", width, height, manga_elements, unrelated_text_length, None)
    return manga_layout

def similar_layouts(layout: MangaLayout, text_length_threshold=5, aspect_ratio_threshold=0.3, annfile="./curated_dataset/database.json"):
    num_speakers = 0
    num_non_speakers = 0
    unrelated_text_length = layout.unrelated_text_length
    for element in layout.elements:
        if type(element) == Speaker:
            num_speakers += 1
        elif type(element) == NonSpeaker:
            num_non_speakers += 1
    filtered_layouts = from_condition(annfile, num_speakers, num_non_speakers, unrelated_text_length, text_length_threshold, layout.width, layout.height, aspect_ratio_threshold, True)

    layout_scores = []
    for filtered_layout in filtered_layouts:
        score, pairs = calc_similarity(layout, filtered_layout, 0.4)
        layout_scores.append((filtered_layout, score, pairs))
    
    layout_scores.sort(key=lambda x: x[1], reverse=True)
    return layout_scores


# test function for from_num_speakers
if __name__ == "__main__":
    annfile = "./curated_dataset/database.json"
    num_speakers = 0
    num_non_speakers = 1
    base_text_length = 80
    text_length_threshold = 20
    aspect_ratio_threshold = 0.4
    base_width = 100
    base_height = 100
    layouts = from_condition(annfile, num_speakers, num_non_speakers, base_text_length, text_length_threshold, base_width, base_height, aspect_ratio_threshold, True)
    print(f"Found {len(layouts)} layouts")