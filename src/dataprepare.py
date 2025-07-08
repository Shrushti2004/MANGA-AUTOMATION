import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def _analyze_metadata(manga_path: str, metadata: dict):
    image_path = os.path.join(manga_path, f"{metadata['id']}.png")
    width, height = metadata['frame_width'], metadata['frame_height']
    text_objects = metadata['text_objects']
    body_objects = metadata['body_objects']
    relations = metadata['relations']
    speaker_and_text_length = {}
    related_text = set()
    for relation in relations:
        related_body_id = relation['body_id']
        related_text_id = relation['text_id']
        related_text.add(related_text_id)
        related_text_content = ""
        for text_object in text_objects:
            if text_object['id'] == related_text_id:
                related_text_content = text_object['text']
                break
        speaker_and_text_length[related_body_id] = speaker_and_text_length.get(related_body_id, 0) + len(related_text_content)

    speaker_objects = []
    non_speaker_objects = []
    unrelated_text_length = 0
    for body_object in body_objects:
        body_id = body_object['id']
        body_bbox = body_object['bbox']
        if body_id in speaker_and_text_length:
            speaker_objects.append({"bbox" : body_bbox, "text_length" : speaker_and_text_length[body_id]})
        else:
            non_speaker_objects.append({"bbox" : body_bbox})

    for text_object in text_objects:
        if text_object['id'] not in related_text:
            unrelated_text_length += len(text_object['text'])
    
    return len(speaker_objects), len(non_speaker_objects), {
        "image_path": image_path,
        "width": width,
        "height": height,
        "speaker_objects": speaker_objects,
        "non_speaker_objects": non_speaker_objects,
        "unrelated_text_length": unrelated_text_length
    }




def main():
    DATASET_PATH = "./curated_dataset"
    
    result = {}
    for manga_dir in tqdm(os.listdir(DATASET_PATH), desc="Processing manga directories"):
        manga_path = os.path.join(DATASET_PATH, manga_dir)
        if not os.path.isdir(manga_path):
            continue
            
        annotation_file = os.path.join(manga_path, 'annotation.json')
        with open(annotation_file, 'r', encoding='utf-8') as f:
            metadatum = json.load(f)
        
        for metadata in metadatum:
            num_speakers, num_non_speakers, content = _analyze_metadata(manga_path, metadata)
            key = f"{num_speakers}_{num_non_speakers}"
            if key not in result:
                result[key] = []
            result[key].append(content)

    output_path = os.path.join(DATASET_PATH, "database.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Database saved to {output_path}")
    

if __name__ == "__main__":
    main()