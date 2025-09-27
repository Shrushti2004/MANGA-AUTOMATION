import os
import io
from PIL import Image
import base64
import json
import time
import requests
from tqdm import tqdm, trange
from datetime import datetime
from lib.image.prompt import generate_prompt_prompt, enhancement_prompt

def enhance_prompts(client, prompts, output_path):
    if os.path.exists(os.path.join(output_path, "enhanced_image_prompts.json")):
        with open(os.path.join(output_path, "enhanced_image_prompts.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    new_prompts = []
    for prompt in tqdm(prompts, desc="Enhancing prompts"):
        messages = [
            {"role": "system", "content": enhancement_prompt},
            {"role": "user", "content": prompt},
        ]
        max_retry = 3
        for attempt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o", messages=messages, temperature=0.0, store=False
                )
                result = response.choices[0].message.content
                new_prompts.append(result)
                break
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                if attempt < max_retry:
                    time.sleep(1)
                    print(f"Retrying... ({attempt + 1}/{max_retry})")
                else:
                    raise Exception("Failed to enhance prompt")
            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_retry:
                    time.sleep(1)
                    print(f"Retrying... ({attempt + 1}/{max_retry})")
                else:
                    raise Exception("Failed to enhance prompt")
    with open(os.path.join(output_path, "enhanced_image_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(new_prompts, f, ensure_ascii=False, indent=4)
    return new_prompts

def generate_image_prompts(client, panels, speakers, output_path, max_retry=3):
    if os.path.exists(os.path.join(output_path, "image_prompts.json")):
        with open(os.path.join(output_path, "image_prompts.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    messages = [
        {"role": "system", "content": "What is the Japanese Roman Name of the following names? MUST Answer in the format of list [name1, name2, ...]. DO NOT change the order."},
        {"role": "user", "content": json.dumps(speakers)},
    ]
    roman_names = []
    print("Changing letters to romanization...")
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0, store=False
            )
            result = response.choices[0].message.content
            roman_names = json.loads(result)
            break
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to generate romanization")
    print(f"Romanization: {roman_names}")
    prompts = []
    for i in range(len(panels)):
        descriptions = [desc for desc in panels[i] if desc["type"] == "description"]
        monologues = [monologue for monologue in panels[i] if monologue["type"] == "monologue"]
        dialogues = [dialogue for dialogue in panels[i] if dialogue["type"] == "dialogue"]
        additional_guideines = f"**Change the letters {speakers} to the following romanization {roman_names}. You MUST assume that {speakers} are the human names**"
        messages = [
            {"role": "system", "content": generate_prompt_prompt.format(descriptions=descriptions, monologues=monologues, dialogues=dialogues, additional_guideines=additional_guideines)},
        ]
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.0, store=False
        )
        result = response.choices[0].message.content
        prompts.append(result)
        with open(os.path.join(output_path, "image_prompts.json"), "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)
    return prompts

def generate_image(client, prompt, image_path,num_retry=5):
    for attempt in range(num_retry):
        try:
            item = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
            ).data[0]
            image_url = item.url
            img_response = requests.get(image_url, stream=True)
            if img_response.status_code == 200:
                with open(image_path, 'wb') as out_file:
                    for chunk in img_response.iter_content(1024):
                        out_file.write(chunk)
                break
        except Exception as e:
            if attempt < num_retry:
                time.sleep(1)
                print(f"Error: {e}")
                print(f"Retrying... ({attempt + 1}/{num_retry})")
            else:
                raise Exception("Failed to generate images")
    return

def generate_image_with_sd(prompt, image_path):
    model="T-anime-v4"
    payload = {
        "prompt": prompt,
        "override_settings" : {
            "sd_model_checkpoint" : model,
        },
        "width": 512,
        "height": 512,
    }
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload).json()
    image_base64 = response["images"][0]
    image_bytes = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    image.save(image_path)
    return
