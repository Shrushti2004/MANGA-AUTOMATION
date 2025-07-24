import os
import json
import time
from openai import OpenAI
from lib.script.prompt import division_prompt, input_example, output_example, panel_example_input, panel_example_output, panel_prompt, richfy_prompt

def _extract_inside_parenthesis(str):
    if str.startswith("「") and str.endswith("」"):
        return str[1:-1]
    elif str.startswith("『") and str.endswith("』"):
        return str[1:-1]
    else:
        return _extract_inside_parenthesis(str[1:])

def refine_elements(elements, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.exists(os.path.join(output_path, "elements_refined.json")):
        with open(os.path.join(output_path, "elements_refined.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    
    for element in elements:
        if element["type"] == "monologue" or element["type"] == "dialogue":
            element["content"] = _extract_inside_parenthesis(element["content"])

    with open(os.path.join(output_path, "elements_refined.json"), "w", encoding="utf-8") as f:
        json.dump(elements, f, ensure_ascii=False, indent=4)

    return elements




def richfy_panel(client, panels,output_path, max_retry=3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.exists(os.path.join(output_path, "panel_richfy.json")):
        with open(os.path.join(output_path, "panel_richfy.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    messages = [
        {"role": "system", "content": richfy_prompt},
        {"role": "user", "content": json.dumps(panels)},
    ]

    for attempt in range(max_retry):
        try: 
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0, store=False
            )
            result = response.choices[0].message.content
            print(result)
            result = json.loads(result)
            with open(os.path.join(output_path, "panel_richfy.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            return result
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to richfy")
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to richfy")



    

def ele2panels(client, elements, output_path, max_retry=3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.exists(os.path.join(output_path, "panel.json")):
        with open(os.path.join(output_path, "panel.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    
    messages = [
        {"role": "system", "content": panel_prompt},
        {"role": "user", "content": panel_example_input},
        {"role": "assistant", "content": panel_example_output},
        {"role": "user", "content": json.dumps(elements)},
    ]

    for attempt in range(max_retry):
        try: 
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0, store=False
            )
            result = response.choices[0].message.content
            result = json.loads(result)
            with open(os.path.join(output_path, "panel.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            return result
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to elements2panels")
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to elements2panels")






def divide_script(client,script_path, output_path, max_retry=3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    base_name = os.path.basename(script_path)

    divided_script_path = os.path.join(output_path, base_name)
    if os.path.exists(divided_script_path):
        with open(divided_script_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    messages = [
        {"role": "system", "content": division_prompt},
        {"role": "user", "content": input_example},
        {"role": "assistant", "content": output_example},
        {"role": "user", "content": content},
    ]

    for attempt in range(max_retry):
        try: 
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0, store=False
            )
            result = response.choices[0].message.content
            result = json.loads(result)
            with open(divided_script_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            return result
        except json.JSONDecodeError as e:
            print(f"Error: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to divide script")
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retry:
                time.sleep(1)
                print(f"Retrying... ({attempt + 1}/{max_retry})")
            else:
                raise Exception("Failed to divide script")
