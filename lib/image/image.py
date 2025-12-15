import os
import io
from PIL import Image
import base64
import json
import time
import requests
from tqdm import tqdm
from lib.image.prompt import generate_prompt_prompt, enhancement_prompt

# -------------------------------
# Enhance prompts using GPT-4o
# -------------------------------
def enhance_prompts(client, prompts, output_path):
    output_file = os.path.join(output_path, "enhanced_image_prompts.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
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
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0
                )
                result = response.choices[0].message.content
                new_prompts.append(result)
                break
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        else:
            raise Exception("Failed to enhance prompt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_prompts, f, ensure_ascii=False, indent=4)
    return new_prompts

# -------------------------------
# Generate image prompts from panels
# -------------------------------
def generate_image_prompts(client, panels, speakers, output_path, max_retry=3):
    output_file = os.path.join(output_path, "image_prompts.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Step 1: Romanize speakers
    messages = [
        {"role": "system", "content": "Convert these names to Japanese Romanization. Return as [name1, name2,...]"},
        {"role": "user", "content": json.dumps(speakers)},
    ]
    roman_names = []
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0
            )
            roman_names = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            print(f"Retry {attempt+1}/{max_retry} failed: {e}")
            time.sleep(1)
    print(f"Romanization: {roman_names}")

    # Step 2: Generate prompts for each panel
    prompts = []
    for i, panel in enumerate(panels):
        descriptions = [d for d in panel if d["type"] == "description"]
        monologues = [m for m in panel if m["type"] == "monologue"]
        dialogues = [d for d in panel if d["type"] == "dialogue"]
        additional_guidelines = f"Change letters {speakers} to romanization {roman_names}."

        messages = [
            {"role": "system", "content": generate_prompt_prompt.format(
                descriptions=descriptions, 
                monologues=monologues, 
                dialogues=dialogues, 
                additional_guideines=additional_guidelines
            )}
        ]
        for attempt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0
                )
                prompts.append(response.choices[0].message.content)
                break
            except Exception as e:
                print(f"Retry {attempt+1}/{max_retry} failed: {e}")
                time.sleep(1)
        else:
            raise Exception("Failed to generate image prompts")

        # Save after each panel
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)
    return prompts

# -------------------------------
# Generate image using DALL·E 3
# -------------------------------
def generate_image(client, prompt, image_path, num_retry=5):
    for attempt in range(num_retry):
        try:
            result = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard"
            ).data[0]
            image_url = result.url
            img_response = requests.get(image_url, stream=True)
            if img_response.status_code == 200:
                with open(image_path, "wb") as f:
                    for chunk in img_response.iter_content(1024):
                        f.write(chunk)
                break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    else:
        raise Exception("Failed to generate image")

# -------------------------------
# Generate image with local Stable Diffusion
# -------------------------------
def generate_image_with_sd(prompt, image_path, width=512, height=512):
    model = "tAnimeV4Pruned_v20"
    payload = {
        "prompt": prompt,
        "override_settings": {"sd_model_checkpoint": model},
        "width": width,
        "height": height,
    }
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload).json()
    image_base64 = response["images"][0]
    image_bytes = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    image.save(image_path)
