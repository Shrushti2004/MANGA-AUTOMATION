import os
import sys
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


MANGA109_ROOT = "./dataset/Manga109"
MANGA109_DIALOG_ROOT = "./dataset/Manga109Dialog"
COLOR_MAP = {
    "face" : "green",
    "frame" : "red",
    "text" : "blue",
    "body" : "purple"
}

def visualize_annotation(saveroot: str, book=None, n=-1) -> None:
    ann_root = os.path.join(MANGA109_ROOT, "annotations")
    img_root = os.path.join(MANGA109_ROOT,"images")
    if not book:
        books = []
        with open(os.path.join(MANGA109_ROOT,"books.txt")) as f:
            books = [line.strip() for line in f]
        book = random.choice(books)
    ann_file = os.path.join(ann_root, f"{book}.xml")
    dialog_ann_file = os.path.join(MANGA109_DIALOG_ROOT,f"{book}.xml")
    img_dir =  os.path.join(img_root, book)
    save_dir = os.path.join(saveroot, book)
    os.makedirs(save_dir, exist_ok=True)
    tree = ET.parse(ann_file)
    root = tree.getroot()
    pages_elem = root.find("pages")
    if pages_elem is None:
        raise RuntimeError(f"pages element is None for {book}")
    page_elements = pages_elem.findall("page")
    if n != -1:
        page_elements = random.sample(page_elements, n)
    dialog_tree = ET.parse(dialog_ann_file)
    dialog_root = dialog_tree.getroot()
    for page_elem in page_elements:
        page_idx = int(page_elem.attrib.get("index", "-1"))
        if page_idx == -1:
            raise RuntimeError(f"page index is None for {book}")
        print(f"Visualize index={page_idx} of {book}")
        img_file = os.path.join(img_dir, f"{page_idx:03d}.jpg")
        out_path = os.path.join(save_dir, f"{page_idx:03d}.jpg")
        img = Image.open(img_file).convert("RGB")
        draw = ImageDraw.Draw(img)
        for child in page_elem:
            if child.tag not in COLOR_MAP:
                raise RuntimeError(f"{child.tag} is not in page element")
            xmin = int(child.attrib['xmin'])
            ymin = int(child.attrib['ymin'])
            xmax = int(child.attrib['xmax'])
            ymax = int(child.attrib['ymax']) 

            draw.rectangle([xmin,ymin,xmax,ymax],outline=COLOR_MAP[child.tag],width=4)
        dialog_page_elem = dialog_root.find(f'.//pages/page[@index="{page_idx}"]')
        if dialog_page_elem is not None:
            for dialog_s2t_elem in dialog_page_elem:
                text_id = dialog_s2t_elem.attrib["text_id"]
                speaker_id = dialog_s2t_elem.attrib["speaker_id"]
                text_elem = page_elem.find(f".//*[@id='{text_id}']")
                speaker_elem = page_elem.find(f".//*[@id='{speaker_id}']")
                if text_elem is None:
                    raise RuntimeError("text_elem is None")
                if speaker_elem is None:
                    raise RuntimeError("speaker_elem is None")
                center_of_text = ((int(text_elem.attrib["xmin"]) + int(text_elem.attrib["xmax"])) // 2, (int(text_elem.attrib["ymin"]) + int(text_elem.attrib["ymax"])) // 2)
                center_of_speaker = ((int(speaker_elem.attrib["xmin"])+ int(speaker_elem.attrib["xmax"])) // 2, (int(speaker_elem.attrib["ymin"]) + int(speaker_elem.attrib["ymax"])) // 2)
                draw.line([center_of_text, center_of_speaker],fill="yellow",width=6)

        img.save(out_path)

def visualize_curated_dataset(datasetdir: str, book=None, n=-1) -> None:
    """キュレーションデータセットをビジュアライズする関数"""
    import json
    
    if not book:
        # キュレーションデータセットから利用可能な本を取得
        if not os.path.exists(datasetdir):
            print(f"キュレーションデータセットが見つかりません: {datasetdir}")
            return
        
        books = [d for d in os.listdir(datasetdir) if os.path.isdir(os.path.join(datasetdir, d))]
        if not books:
            print(f"キュレーションデータセットに本が見つかりません: {datasetdir}")
            return
        book = random.choice(books)
    
    book_dir = os.path.join(datasetdir, book)
    annotation_file = os.path.join(book_dir, "annotation.json")
    
    if not os.path.exists(annotation_file):
        print(f"アノテーションファイルが見つかりません: {annotation_file}")
        return
    
    # アノテーションファイルを読み込み
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 表示するアノテーション数を制限
    if n != -1:
        annotations = random.sample(annotations, min(n, len(annotations)))
    
    # ビジュアライゼーション用のディレクトリを作成
    viz_dir = os.path.join("visualized", "curated", book)
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"キュレーションデータセットをビジュアライズ中: {book}")
    print(f"処理する画像数: {len(annotations)}")
    
    for annotation in annotations:
        frame_id = annotation["id"]
        img_file = os.path.join(book_dir, f"{frame_id}.png")
        
        if not os.path.exists(img_file):
            print(f"画像ファイルが見つかりません: {img_file}")
            continue
        
        # 画像を読み込み
        img = Image.open(img_file).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 各オブジェクトを描画
        for text_obj in annotation["text_objects"]:
            bbox = text_obj["bbox"]
            draw.rectangle(bbox, outline="blue", width=2)
            # テキストIDを表示
            draw.text((bbox[0], bbox[1] - 15), f"T:{text_obj['id'].split('_')[-1]}", fill="blue")
        
        for face_obj in annotation["face_objects"]:
            bbox = face_obj["bbox"]
            draw.rectangle(bbox, outline="green", width=2)
            # 顔IDを表示
            draw.text((bbox[0], bbox[1] - 15), f"F:{face_obj['id'].split('_')[-1]}", fill="green")
        
        for body_obj in annotation["body_objects"]:
            bbox = body_obj["bbox"]
            draw.rectangle(bbox, outline="yellow", width=2)
            # 体IDを表示
            draw.text((bbox[0], bbox[1] - 15), f"B:{body_obj['id'].split('_')[-1]}", fill="yellow")
        
        # 関係性を線で描画
        for relation in annotation["relations"]:
            if relation["type"] == "text_to_face":
                # テキストと顔の中心を結ぶ線
                text_obj = next((t for t in annotation["text_objects"] if t["id"] == relation["text_id"]), None)
                face_obj = next((f for f in annotation["face_objects"] if f["id"] == relation["face_id"]), None)
                
                if text_obj and face_obj:
                    text_bbox = text_obj["bbox"]
                    face_bbox = face_obj["bbox"]
                    text_center = ((text_bbox[0] + text_bbox[2]) // 2, (text_bbox[1] + text_bbox[3]) // 2)
                    face_center = ((face_bbox[0] + face_bbox[2]) // 2, (face_bbox[1] + face_bbox[3]) // 2)
                    draw.line([text_center, face_center], fill="purple", width=2)
            
            elif relation["type"] == "text_to_body":
                # テキストと体の中心を結ぶ線
                text_obj = next((t for t in annotation["text_objects"] if t["id"] == relation["text_id"]), None)
                body_obj = next((b for b in annotation["body_objects"] if b["id"] == relation["body_id"]), None)
                
                if text_obj and body_obj:
                    text_bbox = text_obj["bbox"]
                    body_bbox = body_obj["bbox"]
                    text_center = ((text_bbox[0] + text_bbox[2]) // 2, (text_bbox[1] + text_bbox[3]) // 2)
                    body_center = ((body_bbox[0] + body_bbox[2]) // 2, (body_bbox[1] + body_bbox[3]) // 2)
                    draw.line([text_center, body_center], fill="orange", width=2)
        
        # 画像を保存
        save_path = os.path.join(viz_dir, f"{frame_id}_visualized.png")
        img.save(save_path)
        
        print(f"ビジュアライゼーションを保存: {save_path}")
    
    print(f"キュレーションデータセットのビジュアライゼーションが完了しました: {viz_dir}")

if __name__ == "__main__":
    visualize_curated_dataset("curated_dataset", book="AisazuNihaIrarenai")
