from datetime import datetime
import sys
import os
from PIL import Image
import glob
import random

def combine_images_vertically(image_paths, output_path):
    """複数の画像を縦に繋げて1つの画像にする"""
    if not image_paths:
        return None
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"画像の読み込みエラー {path}: {e}")
            continue
    
    if not images:
        return None
    
    # 最大幅を取得
    max_width = max(img.width for img in images)
    
    # 縦に繋げた画像の高さを計算
    total_height = sum(img.height for img in images)
    
    # 新しい画像を作成
    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
    # 画像を縦に配置
    y_offset = 0
    for img in images:
        # 中央揃えで配置
        x_offset = (max_width - img.width) // 2
        combined_image.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # 保存
    combined_image.save(output_path)
    print(f"保存完了: {output_path}")
    return combined_image

def process_panels_in_groups(image_dir, output_dir):
    """5個ずつの連続するパネルグループを作って、それぞれのグループから1枚ずつランダムに選んで縦に繋げる"""
    all_panels = []
    for d in sorted(os.listdir(image_dir), key=lambda x: int(x.replace("panel", ""))):
        images = [i for i in os.listdir(os.path.join(image_dir, d)) if len(i.split("_")) == 3]
        images = sorted(images, key=lambda x: int(x.split("_")[0]))
        if len(images) != 0:
            image = images[0]
            all_panels.append(os.path.join(image_dir,d, image))
    print(all_panels)
    
    print(f"利用可能なパネル: {all_panels}")
    print(f"総パネル数: {len(all_panels)}")
    
    output_dir = os.path.join(output_dir, "vertically_connected")
    os.makedirs(output_dir, exist_ok=True)
    
    group_size = 3
    panel_groups = []
    
    for i in range(0, len(all_panels), group_size):
        group = all_panels[i:i+group_size]
        panel_groups.append(group)
    
    
    for group_idx, panel_group in enumerate(panel_groups):
        start_panel = panel_group[0]
        end_panel = panel_group[-1]
        output_filename = f"group_{group_idx}.png"
        output_filepath = os.path.join(output_dir, output_filename)
        combine_images_vertically(panel_group, output_filepath)

if __name__ == "__main__":
    args = sys.argv
    process_panels_in_groups(args[1], args[2])

