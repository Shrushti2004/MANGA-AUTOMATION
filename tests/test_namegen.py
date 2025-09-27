from pandas.core.window import ewm
import pytest
import json
import os
from dotenv import load_dotenv
from openai import OpenAI



@pytest.mark.llm
def test_divide_script():
    from lib.script.divide import divide_script
    load_dotenv()
    client = OpenAI()
    script_path = "./examples/DVwife1.txt"
    output_path = "./tests/output/divide_script"
    divide_script(client, script_path, output_path)

def test_refine_elements():
    from lib.script.divide import refine_elements
    load_dotenv()
    client = OpenAI()
    divide_script_path = "./tests/input/divide_script.json"
    output_path = "./tests/output/refine_elements"
    with open(divide_script_path, "r") as f:
        elements = json.load(f)
    refine_elements(elements, output_path)

@pytest.mark.llm
def test_ele2panels():
    from lib.script.divide import ele2panels
    load_dotenv()
    client = OpenAI()
    refine_elements_path = "./tests/input/refine_elements.json"
    with open(refine_elements_path, "r") as f:
        elements = json.load(f)
    ele2panels(client, elements, "./tests/output/ele2panels")

@pytest.mark.llm
def test_generate_image_prompts():
    from lib.image.image import generate_image_prompts
    load_dotenv()
    client = OpenAI()
    ele2panels_pata = "./tests/input/ele2panels.json"
    with open(ele2panels_path, "r") as f:
        panels = json.load(f)
    generate_image_prompts(client, panels, "./tests/output/image_prompts")

@pytest.mark.llm
def test_enhance_prompts():
    from lib.image.image import enhance_prompts
    load_dotenv()
    client = OpenAI()
    image_prompts_path = "./tests/input/image_prompts.json"
    with open(image_prompts_path, "r") as f:
        image_prompts = json.load(f)
    enhance_prompts(client, image_prompts, "./tests/output/enhanced_image_prompts")

@pytest.mark.llm
def test_generate_image():
    from lib.image.image import generate_image
    load_dotenv()
    client = OpenAI()
    image_prompts_path = "./tests/input/enhanced_image_prompts.json"
    with open(image_prompts_path, "r") as f:
        image_prompts = json.load(f)
    prompt = image_prompts[0]
    generate_image(client, prompt, "./tests/output/image_per_panel/test.png")

def test_run_controlnet_openpose():
    from lib.image.controlnet import run_controlnet_openpose
    run_controlnet_openpose("./tests/input/image.png", "./tests/output/controlnet_openpose")

def test_generate_with_controlnet_openpose():
    from lib.image.controlnet import generate_with_controlnet_openpose
    generate_with_controlnet_openpose("./tests/input/image.png", "a beautiful woman in a red dress", "./tests/output/controlnet_openpose_image/test.png")

def test_generate_animepose_image():
    from lib.name.name import generate_animepose_image
    from matplotlib import pyplot as plt
    import random
    image_prompts_path = "./tests/input/image_prompts.json"
    with open(image_prompts_path, "r") as f:
        image_prompts = json.load(f)
    image_dir = "./tests/output/images/ver1"
    panel_dir_names = os.listdir(image_dir)
    random_dir_name = random.choice(panel_dir_names)
    random_dir_idx = int(random_dir_name.split("l")[1])
    input_image_path = os.path.join(image_dir, random_dir_name, "00.png")
    prompt = image_prompts[random_dir_idx]
    generate_animepose_image(input_image_path, prompt, "./tests/output/animepose_image/test.png")

def test_compare_original_and_animepose():
    from lib.name.name import generate_animepose_image
    from matplotlib import pyplot as plt
    from PIL import Image
    import random
    import glob
    
    image_prompts_path = "./tests/input/image_prompts.json"
    image_dir = "./tests/output/images/ver1"
    output_dir = "./tests/output/animepose_comparison"
    
    with open(image_prompts_path, "r") as f:
        image_prompts = json.load(f)
    
    panel_dir_names = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    panel_dir_names.sort()
    
    print(f"処理するパネル数: {len(panel_dir_names)}")
    
    for panel_dir in panel_dir_names:
        panel_path = os.path.join(image_dir, panel_dir)
        
        image_files = glob.glob(os.path.join(panel_path, "*.png"))
        if not image_files:
            print(f"警告: {panel_dir}に画像ファイルが見つかりません")
            continue
        
        selected_image_path = random.choice(image_files)
        panel_idx = int(panel_dir.replace("panel", ""))
        
        if panel_idx >= len(image_prompts):
            print(f"警告: {panel_dir}に対応するプロンプトがありません")
            continue
        
        prompt = image_prompts[panel_idx]
        
        original_image = Image.open(selected_image_path)
        
        animepose_output_path = os.path.join(output_dir, f"{panel_dir}_animepose.png")
        generate_animepose_image(selected_image_path, prompt, animepose_output_path)
        
        animepose_image = Image.open(animepose_output_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_image)
        ax1.set_title(f"元の画像: {os.path.basename(selected_image_path)}")
        ax1.axis('off')
        
        ax2.imshow(animepose_image)
        ax2.set_title(f"AnimePose結果: {panel_dir}")
        ax2.axis('off')
        
        fig.suptitle(f"比較結果: {panel_dir}", fontsize=16)
        
        save_path = os.path.join(output_dir, f"{panel_dir}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"保存完了: {save_path}")
        
    
    print("比較処理が完了しました")


def test_draw_vertical_text():
    from lib.name.name import draw_vertical_text
    from PIL import Image, ImageDraw, ImageFont
    FONTPATH = "fonts/NotoSansCJK-Regular.ttc"
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    bbox = [400, 100, 500, 400]
    text = "こんにちは、私は大学生です。明日からは学校に行きたいと思っています。でも今日は元気がないです。だからひとまず朝ご飯をいただきます。美味しそう！！！"
    draw_vertical_text(img, text, bbox, "dialogue")
    img.show()