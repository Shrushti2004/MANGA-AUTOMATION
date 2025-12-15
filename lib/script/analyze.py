import os
import json
import time
from lib.script.prompt import storyboard_prompt

def analyze_storyboard(client, panels, output_path, max_retry=3):

    metadata_path = os.path.join(output_path, "panel_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            print("Loaded existing storyboard metadata.")
            return json.load(f)

    messages = [
        {"role": "system", "content": storyboard_prompt},
        {"role": "user", "content": json.dumps(panels, ensure_ascii=False)},
    ]

    print("Analyzing storyboard structure...")

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0
            )

            result_text = response.choices[0].message.content

            # Remove markdown wrappers like ```json
            if result_text.startswith("```"):
                result_text = result_text.strip("`").replace("json\n", "").strip()

            metadata = json.loads(result_text)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            return metadata

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)

    raise Exception("Failed to generate storyboard metadata")
