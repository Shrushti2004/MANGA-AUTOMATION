import os
import json
import time
import requests
from tqdm import tqdm, trange
from datetime import datetime
from lib.image.prompt import enhancement_prompt

def enhance_prompt(client, panel, max_retry=3):

    messages = [
        {"role": "system", "content": enhancement_prompt},
        {"role": "user", "content": json.dumps(panel)},
    ]
    max_retry = 3
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0, store=False
            )
            result = response.choices[0].message.content
            return result
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

def generate_images(client, panels, out_path, num_images=5,num_retry=3):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    image_base_dir = os.path.join(out_path, date, "images")
    if not os.path.exists(image_base_dir):
        os.makedirs(image_base_dir)

    for panel_idx, panel in enumerate(panels):
        panel_dir = os.path.join(image_base_dir, f"panel{panel_idx}")
        os.makedirs(panel_dir, exist_ok=True)
        prompt = enhance_prompt(client, panel)
        print(f"Prompt: {prompt}")

        for i in trange(num_images, desc="Generating images"):
            filename = f"{i:02d}.png"
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
                        output_file_path = os.path.join(panel_dir, filename)
                        with open(output_file_path, 'wb') as out_file:
                            for chunk in img_response.iter_content(1024):
                                out_file.write(chunk)
                        break
                except Exception as e:
                    if attempt < num_retry:
                        time.sleep(1)
                        print(f"Retrying... ({attempt + 1}/{num_retry})")
                    else:
                        raise Exception("Failed to generate images")
    return date, image_base_dir
