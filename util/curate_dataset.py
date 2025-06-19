import os
import sys
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from typing import List
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from tqdm import tqdm


MANGA109_ROOT = "./dataset/Manga109"
MANGA109_DIALOG_ROOT = "./dataset/Manga109Dialog"
ANN_ROOT = os.path.join(MANGA109_ROOT, "annotations")
IMG_ROOT = os.path.join(MANGA109_ROOT,"images")


class MangaObject:
    def __init__(self, text=None, text_bbox=None, face_bbox=None, body_bbox=None):
        self.text = text
        self.text_bbox = text_bbox
        self.face_bbox = face_bbox
        self.body_bbox = body_bbox
    
    def __str__(self):
        parts = []
        if self.text:
            parts.append(f"Text: '{self.text}'")
        if self.text_bbox:
            parts.append(f"Text_bbox: {self.text_bbox}")
        if self.face_bbox:
            parts.append(f"Face_bbox: {self.face_bbox}")
        if self.body_bbox:
            parts.append(f"Body_bbox: {self.body_bbox}")
        
        if not parts:
            return "MangaObject(empty)"
        
        return f"MangaObject({', '.join(parts)})"

class FrameObject:
    def __init__(self, frame_bbox, objects=None):
        self.frame_bbox = frame_bbox
        self.objects = objects if objects is not None else []


def calc_bbox_intersection(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    width = max(0, xi2 - xi1)
    height = max(0, yi2 - yi1)

    return width * height

def associate_frame(bbox, frame_dict):
    max_frame_id = None
    cur_max = 0
    for frame_id, frame_obj in frame_dict.items():
        frame_bbox = frame_obj.frame_bbox
        intersection = calc_bbox_intersection(bbox, frame_bbox)
        if intersection > cur_max:
            cur_max = intersection
            max_frame_id = frame_id
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if max_frame_id is None or cur_max / area < 0.9:
        return None
    return max_frame_id

def find_speaker(book, page_idx, text_id):
    ann_file = os.path.join(MANGA109_DIALOG_ROOT, f"{book}.xml")
    if not os.path.exists(ann_file):
        print(f"annotation file for {book} does not exist")
        return []
    root = ET.parse(ann_file).getroot()
    page_elem = root.find(f"./pages/page[@index='{page_idx}']")
    if page_elem is None:
        return None
    s2t_elem = page_elem.find(f"./*[@text_id='{text_id}']")
    if s2t_elem is None:
        return None
    return s2t_elem.attrib["speaker_id"]


def visualize_panels(book, all_pages_result, save_dir="visualized"):
    """matplotlibを使ってgrid上にパネルを配置し、各ページごとに1つの画像にまとめる"""
    
    # ページごとにパネルをグループ化
    page_panels = {}
    for page_idx, frame_dict in all_pages_result.items():
        if not frame_dict:
            continue
        
        # 画像ファイルのパス
        img_file = os.path.join("dataset/Manga109/images", book, f"{page_idx:03d}.jpg")
        if not os.path.exists(img_file):
            continue
        
        # 画像を読み込み
        img = Image.open(img_file)
        
        # 各フレーム（パネル）を処理
        panels = []
        for frame_id, frame_obj in frame_dict.items():
            # フレームのbboxで画像を切り抜き
            x1, y1, x2, y2 = frame_obj.frame_bbox
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # 切り抜いた画像に描画
            draw = ImageDraw.Draw(cropped_img)
            
            # 各オブジェクトを描画
            for obj in frame_obj.objects:
                # テキストボックスの中心座標を計算
                text_center = None
                if obj.text_bbox:
                    tx1, ty1, tx2, ty2 = obj.text_bbox
                    tx1, ty1 = tx1 - x1, ty1 - y1
                    tx2, ty2 = tx2 - x1, ty2 - y1
                    draw.rectangle([tx1, ty1, tx2, ty2], outline="blue", width=2)
                    text_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                
                # 顔のbboxを描画（緑色）
                face_center = None
                if obj.face_bbox:
                    fx1, fy1, fx2, fy2 = obj.face_bbox
                    fx1, fy1 = fx1 - x1, fy1 - y1
                    fx2, fy2 = fx2 - x1, fy2 - y1
                    draw.rectangle([fx1, fy1, fx2, fy2], outline="green", width=2)
                    face_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
                
                # 体のbboxを描画（黄色）
                body_center = None
                if obj.body_bbox:
                    bx1, by1, bx2, by2 = obj.body_bbox
                    bx1, by1 = bx1 - x1, by1 - y1
                    bx2, by2 = bx2 - x1, by2 - y1
                    draw.rectangle([bx1, by1, bx2, by2], outline="yellow", width=2)
                    body_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                
                # textとface/bodyを直線で結ぶ
                if text_center:
                    if face_center:
                        draw.line([text_center, face_center], fill="purple", width=2)
                    if body_center:
                        draw.line([text_center, body_center], fill="purple", width=2)
            
            # PIL画像をnumpy配列に変換
            img_array = np.array(cropped_img)
            panels.append((frame_id, img_array))
        
        # フレームID順にソート
        panels.sort(key=lambda x: x[0])
        page_panels[page_idx] = panels
    
    # 各ページごとにグリッド画像を作成
    for page_idx, panels in page_panels.items():
        if not panels:
            continue
        
        # パネル数に基づいてグリッドサイズを決定
        num_panels = len(panels)
        grid_cols = min(8, num_panels)  # 最大8列
        grid_rows = (num_panels + grid_cols - 1) // grid_cols  # 必要な行数を計算
        
        # 図を作成
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 5 * grid_rows))
        fig.suptitle(f'{book} - Page {page_idx} ({num_panels} panels)', fontsize=16)
        
        # 1行の場合はaxesを配列として扱う
        if grid_rows == 1:
            axes = [axes]
        
        # 各パネルを配置（右上から左下へ）
        panel_idx = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                # 右上から左下への順序に変更
                actual_col = grid_cols - 1 - col
                ax = axes[row][actual_col]
                
                if panel_idx < len(panels):
                    frame_id, img_array = panels[panel_idx]
                    ax.imshow(img_array)
                    ax.set_title(f'Frame {frame_id}', fontsize=10)
                    ax.axis('off')
                    panel_idx += 1
                else:
                    ax.axis('off')
        
        # 余分なサブプロットを非表示
        for j in range(panel_idx, grid_rows * grid_cols):
            row = j // grid_cols
            col = j % grid_cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        
        # 保存
        os.makedirs(save_dir, exist_ok=True)
        book_dir = os.path.join(save_dir, book)
        os.makedirs(book_dir, exist_ok=True)
        save_path = os.path.join(book_dir, f"{page_idx:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ページ {page_idx} のグリッド画像を保存しました: {save_path}")

def generate_curated_dataset(book, all_pages_result, saveroot):
    """各フレームをcropして保存し、アノテーション情報をJSON形式で保存する"""
    
    # 保存先ディレクトリを作成
    book_dir = os.path.join(saveroot, book)
    os.makedirs(book_dir, exist_ok=True)
    
    # アノテーション情報を格納するリスト
    annotations = []
    
    # 一意のIDを割り当てるためのカウンター
    global_id_counter = 0
    
    # 各ページを処理
    for page_idx, frame_dict in all_pages_result.items():
        if not frame_dict:
            continue
        
        # 画像ファイルのパス
        img_file = os.path.join("dataset/Manga109/images", book, f"{page_idx:03d}.jpg")
        if not os.path.exists(img_file):
            continue
        
        # 画像を読み込み
        img = Image.open(img_file)
        
        # 各フレーム（パネル）を処理
        for frame_id, frame_obj in frame_dict.items():
            # フレームのbboxで画像を切り抜き
            x1, y1, x2, y2 = frame_obj.frame_bbox
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # 一意のIDを割り当て
            frame_global_id = f"{book}_{page_idx:03d}_{frame_id}_{global_id_counter:06d}"
            global_id_counter += 1
            
            # 画像を保存
            save_path = os.path.join(book_dir, f"{frame_global_id}.png")
            cropped_img.save(save_path)
            
            # アノテーション情報を作成
            annotation = {
                "id": frame_global_id,
                "original_page": page_idx,
                "original_frame_id": frame_id,
                "frame_width": x2 - x1,
                "frame_height": y2 - y1,
                "text_objects": [],
                "face_objects": [],
                "body_objects": [],
                "relations": []
            }
            
            # 各オブジェクトを処理
            text_objects = []
            face_objects = []
            body_objects = []
            
            for obj in frame_obj.objects:
                # テキストオブジェクトの処理
                if obj.text_bbox:
                    # crop後の座標系に変換
                    tx1, ty1, tx2, ty2 = obj.text_bbox
                    tx1, ty1 = tx1 - x1, ty1 - y1
                    tx2, ty2 = tx2 - x1, ty2 - y1
                    
                    text_id = f"{frame_global_id}_text_{len(text_objects):03d}"
                    text_obj = {
                        "id": text_id,
                        "bbox": [tx1, ty1, tx2, ty2],
                        "text": obj.text
                    }
                    text_objects.append(text_obj)
                    annotation["text_objects"].append(text_obj)
                
                # 顔オブジェクトの処理
                if obj.face_bbox:
                    # crop後の座標系に変換
                    fx1, fy1, fx2, fy2 = obj.face_bbox
                    fx1, fy1 = fx1 - x1, fy1 - y1
                    fx2, fy2 = fx2 - x1, fy2 - y1
                    
                    face_id = f"{frame_global_id}_face_{len(face_objects):03d}"
                    face_obj = {
                        "id": face_id,
                        "bbox": [fx1, fy1, fx2, fy2]
                    }
                    face_objects.append(face_obj)
                    annotation["face_objects"].append(face_obj)
                
                # 体オブジェクトの処理
                if obj.body_bbox:
                    # crop後の座標系に変換
                    bx1, by1, bx2, by2 = obj.body_bbox
                    bx1, by1 = bx1 - x1, by1 - y1
                    bx2, by2 = bx2 - x1, by2 - y1
                    
                    body_id = f"{frame_global_id}_body_{len(body_objects):03d}"
                    body_obj = {
                        "id": body_id,
                        "bbox": [bx1, by1, bx2, by2]
                    }
                    body_objects.append(body_obj)
                    annotation["body_objects"].append(body_obj)
                
                # 関係性の処理
                if obj.text_bbox and obj.face_bbox:
                    # textとfaceの関係
                    text_id = f"{frame_global_id}_text_{len(text_objects)-1:03d}"
                    face_id = f"{frame_global_id}_face_{len(face_objects)-1:03d}"
                    relation = {
                        "type": "text_to_face",
                        "text_id": text_id,
                        "face_id": face_id
                    }
                    annotation["relations"].append(relation)
                
                if obj.text_bbox and obj.body_bbox:
                    # textとbodyの関係
                    text_id = f"{frame_global_id}_text_{len(text_objects)-1:03d}"
                    body_id = f"{frame_global_id}_body_{len(body_objects)-1:03d}"
                    relation = {
                        "type": "text_to_body",
                        "text_id": text_id,
                        "body_id": body_id
                    }
                    annotation["relations"].append(relation)
            
            annotations.append(annotation)
    
    # アノテーションファイルを保存
    annotation_file = os.path.join(book_dir, "annotation.json")
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"キュレーションデータセットを生成しました: {book_dir}")
    print(f"保存された画像数: {len(annotations)}")
    print(f"アノテーションファイル: {annotation_file}")

def curate_book(book, saveroot, debug=False):
    # 既にannotation.jsonが存在する場合はスキップ
    book_dir = os.path.join(saveroot, book)
    annotation_file = os.path.join(book_dir, "annotation.json")
    if os.path.exists(annotation_file):
        print(f"スキップ: {book} のannotation.jsonが既に存在します")
        return None
    
    ann_file = os.path.join(ANN_ROOT, f"{book}.xml")
    img_dir = os.path.join(IMG_ROOT,book)
    out_dir = os.path.join(saveroot, book)
    os.makedirs(out_dir, exist_ok=True)
    pages_elem = ET.parse(ann_file).getroot().find("pages")
    if pages_elem is None:
        print(f"pages element not found in {book}")
        return None
    
    page_elements = pages_elem.findall("page")
    all_pages_result = {}
    
    for page_elem in tqdm(page_elements,desc=f"Curating {book}"):
        frame_dict = {}
        page_idx = int(page_elem.attrib["index"])
        img_file = os.path.join(img_dir, f"{page_idx:03d}.jpg")
        
        # まず全てのframeを作成
        frame_elements = page_elem.findall("frame")
        for frame_elem in frame_elements:
            frame_id = frame_elem.attrib["id"]
            xmin = int(frame_elem.attrib["xmin"])
            ymin = int(frame_elem.attrib["ymin"])
            xmax = int(frame_elem.attrib["xmax"])
            ymax = int(frame_elem.attrib["ymax"])
            frame_obj = FrameObject([xmin, ymin, xmax, ymax])
            frame_dict[frame_id] = frame_obj
        
        # 全ての要素を取得
        text_elements = page_elem.findall("text")
        face_elements = page_elem.findall("face")
        body_elements = page_elem.findall("body")
        
        # frameごとにループして、各frameに含まれる要素を処理
        for frame_id, frame_obj in frame_dict.items():
            for text_elem in text_elements:
                text_id = text_elem.attrib["id"]
                xmin = int(text_elem.attrib["xmin"])
                ymin = int(text_elem.attrib["ymin"])
                xmax = int(text_elem.attrib["xmax"])
                ymax = int(text_elem.attrib["ymax"])
                text_bbox = [xmin, ymin, xmax, ymax]
                
                # このtextが現在のframeに属するかチェック
                text_frame = associate_frame(text_bbox, frame_dict)
                if text_frame != frame_id:
                    continue
                
                # text_idに対応するspeaker_idを取得
                speaker_id = find_speaker(book, page_idx, text_id)
                
                # speaker_idに対応するfaceやbodyを特定
                corresponding_face_bbox = None
                corresponding_body_bbox = None
                
                if speaker_id is not None:
                    # speaker_idに対応するfaceを検索
                    for face_elem in face_elements:
                        if face_elem.attrib.get("id") == speaker_id:
                            face_xmin = int(face_elem.attrib["xmin"])
                            face_ymin = int(face_elem.attrib["ymin"])
                            face_xmax = int(face_elem.attrib["xmax"])
                            face_ymax = int(face_elem.attrib["ymax"])
                            face_bbox = [face_xmin, face_ymin, face_xmax, face_ymax]
                            
                            # このfaceが現在のframeに属するかチェック
                            face_frame = associate_frame(face_bbox, frame_dict)
                            if face_frame == frame_id:
                                corresponding_face_bbox = face_bbox
                                break
                    
                    # speaker_idに対応するbodyを検索
                    for body_elem in body_elements:
                        if body_elem.attrib.get("id") == speaker_id:
                            body_xmin = int(body_elem.attrib["xmin"])
                            body_ymin = int(body_elem.attrib["ymin"])
                            body_xmax = int(body_elem.attrib["xmax"])
                            body_ymax = int(body_elem.attrib["ymax"])
                            body_bbox = [body_xmin, body_ymin, body_xmax, body_ymax]
                            
                            # このbodyが現在のframeに属するかチェック
                            body_frame = associate_frame(body_bbox, frame_dict)
                            if body_frame == frame_id:
                                corresponding_body_bbox = body_bbox
                                break
                
                # MangaObjectを作成
                text_content = text_elem.text
                manga_obj = MangaObject(
                    text=text_content,
                    text_bbox=text_bbox,
                    face_bbox=corresponding_face_bbox,
                    body_bbox=corresponding_body_bbox
                )
                
                # 現在のframeにMangaObjectを追加
                frame_obj.objects.append(manga_obj)
            
            # このframeに含まれるがtextと対応付けされていないface要素を処理
            for face_elem in face_elements:
                face_xmin = int(face_elem.attrib["xmin"])
                face_ymin = int(face_elem.attrib["ymin"])
                face_xmax = int(face_elem.attrib["xmax"])
                face_ymax = int(face_elem.attrib["ymax"])
                face_bbox = [face_xmin, face_ymin, face_xmax, face_ymax]
                
                # このfaceが現在のframeに属するかチェック
                face_frame = associate_frame(face_bbox, frame_dict)
                if face_frame != frame_id:
                    continue
                
                # このfaceが既にtextと対応付けされているかチェック
                face_already_used = False
                for obj in frame_obj.objects:
                    if obj.face_bbox == face_bbox:
                        face_already_used = True
                        break
                
                if not face_already_used:
                    # textと対応付けされていないfaceをMangaObjectとして保存
                    manga_obj = MangaObject(
                        text=None,
                        text_bbox=None,
                        face_bbox=face_bbox,
                        body_bbox=None
                    )
                    frame_obj.objects.append(manga_obj)
            
            # このframeに含まれるがtextと対応付けされていないbody要素を処理
            for body_elem in body_elements:
                body_xmin = int(body_elem.attrib["xmin"])
                body_ymin = int(body_elem.attrib["ymin"])
                body_xmax = int(body_elem.attrib["xmax"])
                body_ymax = int(body_elem.attrib["ymax"])
                body_bbox = [body_xmin, body_ymin, body_xmax, body_ymax]
                
                # このbodyが現在のframeに属するかチェック
                body_frame = associate_frame(body_bbox, frame_dict)
                if body_frame != frame_id:
                    continue
                
                # このbodyが既にtextと対応付けされているかチェック
                body_already_used = False
                for obj in frame_obj.objects:
                    if obj.body_bbox == body_bbox:
                        body_already_used = True
                        break
                
                if not body_already_used:
                    # textと対応付けされていないbodyをMangaObjectとして保存
                    manga_obj = MangaObject(
                        text=None,
                        text_bbox=None,
                        face_bbox=None,
                        body_bbox=body_bbox
                    )
                    frame_obj.objects.append(manga_obj)

        
        # このページの結果を保存
        all_pages_result[page_idx] = frame_dict

    if debug:
        visualize_panels(book, all_pages_result)

    generate_curated_dataset(book, all_pages_result, saveroot)
    
    return


def run_curation(saveroot: str) -> None:
    os.makedirs(saveroot,exist_ok=True)
    books = []
    with open(os.path.join(MANGA109_ROOT,"books.txt")) as f:
        books = [line.strip() for line in f]
    for book in tqdm(books):
        curate_book(book, saveroot)
    return

if __name__ == "__main__":
    run_curation("curated_dataset")