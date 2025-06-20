import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# 日本語フォントの設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP']

def load_annotation_file(annotation_path: str) -> List[Dict]:
    """アノテーションファイルを読み込む"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_book(book_dir: str) -> Tuple[List[Dict], int]:
    """1つの本のアノテーションファイルを分析"""
    annotation_path = os.path.join(book_dir, "annotation.json")
    if not os.path.exists(annotation_path):
        print(f"アノテーションファイルが見つかりません: {annotation_path}")
        return [], 0
    
    annotations = load_annotation_file(annotation_path)
    total_panels = len(annotations)
    
    # 条件を満たすパネルを抽出（textとhuman bounding boxが最低1個ずつ存在）
    valid_panels = []
    
    for annotation in annotations:
        text_count = len(annotation.get("text_objects", []))
        face_count = len(annotation.get("face_objects", []))
        body_count = len(annotation.get("body_objects", []))
        human_count = face_count + body_count  # human bounding boxの総数
        
        # 条件: textとhuman bounding boxがどちらも最低1個ずつ存在
        if text_count >= 1 and human_count >= 1:
            valid_panels.append({
                "id": annotation["id"],
                "text_count": text_count,
                "face_count": face_count,
                "body_count": body_count,
                "human_count": human_count,
                "relation_count": len(annotation.get("relations", []))
            })
    
    return valid_panels, total_panels

def analyze_all_books(curated_dataset_path: str) -> Tuple[List[Dict], int]:
    """全ての本を分析"""
    all_valid_panels = []
    total_all_panels = 0
    
    # 各本のディレクトリを処理
    for book_name in os.listdir(curated_dataset_path):
        book_dir = os.path.join(curated_dataset_path, book_name)
        if os.path.isdir(book_dir):
            valid_panels, total_panels = analyze_book(book_dir)
            all_valid_panels.extend(valid_panels)
            total_all_panels += total_panels
    
    return all_valid_panels, total_all_panels

def categorize_panels(valid_panels: List[Dict]) -> Dict:
    """パネルを各種カテゴリ別に分類"""
    categories = {
        "text_count": Counter(),
        "human_count": Counter(),
        "face_count": Counter(),
        "body_count": Counter(),
        "relation_count": Counter()
    }
    
    for panel in valid_panels:
        categories["text_count"][panel["text_count"]] += 1
        categories["human_count"][panel["human_count"]] += 1
        categories["face_count"][panel["face_count"]] += 1
        categories["body_count"][panel["body_count"]] += 1
        categories["relation_count"][panel["relation_count"]] += 1
    
    return categories

def create_visualization(categories: Dict, total_valid_panels: int, total_all_panels: int, save_path: str = "dataset_analysis.png"):
    """分析結果を可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('キュレーションデータセット分析結果', fontsize=16)
    
    # テキスト数別分布
    ax1 = axes[0, 0]
    text_counts = sorted(categories["text_count"].keys())
    text_values = [categories["text_count"][count] for count in text_counts]
    bars1 = ax1.bar(text_counts, text_values)
    ax1.set_title('テキストボックス数別分布')
    ax1.set_xlabel('テキストボックス数')
    ax1.set_ylabel('パネル数')
    ax1.set_xticks(text_counts)
    
    # ヒューマン数別分布
    ax2 = axes[0, 1]
    human_counts = sorted(categories["human_count"].keys())
    human_values = [categories["human_count"][count] for count in human_counts]
    bars2 = ax2.bar(human_counts, human_values)
    ax2.set_title('ヒューマンバウンディングボックス数別分布')
    ax2.set_xlabel('ヒューマンバウンディングボックス数')
    ax2.set_ylabel('パネル数')
    ax2.set_xticks(human_counts)
    
    # 顔数別分布
    ax3 = axes[0, 2]
    face_counts = sorted(categories["face_count"].keys())
    face_values = [categories["face_count"][count] for count in face_counts]
    bars3 = ax3.bar(face_counts, face_values)
    ax3.set_title('顔バウンディングボックス数別分布')
    ax3.set_xlabel('顔バウンディングボックス数')
    ax3.set_ylabel('パネル数')
    ax3.set_xticks(face_counts)
    
    # 体数別分布
    ax4 = axes[1, 0]
    body_counts = sorted(categories["body_count"].keys())
    body_values = [categories["body_count"][count] for count in body_counts]
    bars4 = ax4.bar(body_counts, body_values)
    ax4.set_title('体バウンディングボックス数別分布')
    ax4.set_xlabel('体バウンディングボックス数')
    ax4.set_ylabel('パネル数')
    ax4.set_xticks(body_counts)
    
    # 関係性数別分布
    ax5 = axes[1, 1]
    relation_counts = sorted(categories["relation_count"].keys())
    relation_values = [categories["relation_count"][count] for count in relation_counts]
    bars5 = ax5.bar(relation_counts, relation_values)
    ax5.set_title('関係性数別分布')
    ax5.set_xlabel('関係性数')
    ax5.set_ylabel('パネル数')
    ax5.set_xticks(relation_counts)
    
    # 全体統計
    ax6 = axes[1, 2]
    valid_ratio = total_valid_panels / total_all_panels * 100
    ax6.text(0.5, 0.6, f'総パネル数\n{total_all_panels}', 
             ha='center', va='center', fontsize=16, transform=ax6.transAxes)
    ax6.text(0.5, 0.4, f'有効パネル数\n{total_valid_panels}', 
             ha='center', va='center', fontsize=16, transform=ax6.transAxes)
    ax6.text(0.5, 0.2, f'有効率\n{valid_ratio:.1f}%', 
             ha='center', va='center', fontsize=16, transform=ax6.transAxes)
    ax6.set_title('全体統計')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可視化結果を保存しました: {save_path}")
    print(f"総パネル数: {total_all_panels}")
    print(f"有効パネル数: {total_valid_panels}")
    print(f"有効率: {valid_ratio:.1f}%")

def analyze_bbox_combinations(valid_panels: List[Dict]) -> Counter:
    """bboxの個数の組み合わせパターンを分析"""
    combinations = Counter()
    
    for panel in valid_panels:
        text_count = panel["text_count"]
        human_count = panel["human_count"]
        relation_count = panel["relation_count"]
        
        # 組み合わせをタプルとしてカウント
        combination = (text_count, human_count, relation_count)
        combinations[combination] += 1
    
    return combinations

def create_combination_visualization(combinations: Counter, save_path: str = "bbox_combinations.png"):
    """bboxの組み合わせパターンを可視化"""
    if not combinations:
        print("組み合わせデータがありません")
        return
    
    # テキスト数と関係性数の組み合わせでカテゴリをまとめる
    aggregated_combinations = {}
    
    for (text, human, relation), count in combinations.items():
        if relation == 0:
            # R0の場合は、すべて「R0」としてまとめる
            key = "R0"
        else:
            # その他の場合は、テキスト数と関係性数の組み合わせでまとめる
            key = f"T{text}-R{relation}"
        
        if key in aggregated_combinations:
            aggregated_combinations[key] += count
        else:
            aggregated_combinations[key] = count
    
    # 組み合わせをソート（R0を最初に、その後はテキスト数、関係性数の順）
    def sort_key(item):
        key = item[0]
        if key == "R0":
            return (-1, 0)  # R0を最初に
        else:
            # T{text}-R{relation}から数値を抽出
            parts = key.split('-')
            text_num = int(parts[0][1:])  # T1 -> 1
            relation_num = int(parts[1][1:])  # R1 -> 1
            return (text_num, relation_num)
    
    sorted_combinations = sorted(aggregated_combinations.items(), key=sort_key)
    
    # データを準備
    labels = []
    values = []
    
    for key, count in sorted_combinations:
        labels.append(key)
        values.append(count)
    
    # グラフを作成
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 8))
    
    bars = ax.bar(range(len(labels)), values)
    ax.set_title('bboxの組み合わせパターン分布', fontsize=16)
    ax.set_xlabel('パターン (T:テキスト数, R:関係性数)', fontsize=12)
    ax.set_ylabel('パネル数', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # バーの上に値を表示
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                f'{value}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"組み合わせ分析結果を保存しました: {save_path}")
    
    # 簡単な統計情報を表示
    total_panels = sum(values)
    print(f"\nパターン数: {len(aggregated_combinations)}")
    print(f"総パネル数: {total_panels}")
    
    # 各パターンの詳細を表示
    print("\n各パターンの詳細:")
    for key, count in sorted_combinations:
        percentage = count / total_panels * 100
        print(f"{key}: {count}パネル ({percentage:.1f}%)")

def get_random_sample_from_category(valid_panels: List[Dict], category: str, curated_dataset_path: str) -> Tuple[str, str, Dict]:
    """指定されたカテゴリからランダムに1つのパネルを選択"""
    category_panels = []
    
    for panel in valid_panels:
        text_count = panel["text_count"]
        human_count = panel["human_count"]
        relation_count = panel["relation_count"]
        
        # カテゴリを判定
        if relation_count == 0:
            panel_category = "R0"
        else:
            panel_category = f"T{text_count}-R{relation_count}"
        
        if panel_category == category:
            category_panels.append(panel)
    
    if not category_panels:
        print(f"カテゴリ {category} のパネルが見つかりません")
        return "", "", {}
    
    # ランダムに1つ選択
    selected_panel = random.choice(category_panels)
    panel_id = selected_panel["id"]
    
    # 画像ファイルのパスを構築
    # panel_idの形式: "book_vol_page_frame_globalid" または "book_page_frame_globalid" または "BOOK_NAME_page_frame_globalid"
    parts = panel_id.split('_')
    
    # ディレクトリ名を決定
    book_name = None
    
    # まず、実際のディレクトリが存在するかチェック
    for i in range(1, min(4, len(parts))):
        candidate_name = '_'.join(parts[:i])
        candidate_path = os.path.join(curated_dataset_path, candidate_name)
        if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
            # annotation.jsonが存在するかもチェック
            annotation_path = os.path.join(candidate_path, "annotation.json")
            if os.path.exists(annotation_path):
                book_name = candidate_name
                break
    
    # 見つからない場合は、従来の方法で推定
    if book_name is None:
        if len(parts) >= 4 and parts[1].startswith('vol'):
            # "book_vol_page_frame_globalid" 形式
            book_name = f"{parts[0]}_{parts[1]}"
        else:
            # "book_page_frame_globalid" 形式
            book_name = parts[0]
    
    image_path = os.path.join(curated_dataset_path, book_name, f"{panel_id}.png")
    
    # デバッグ情報
    print(f"カテゴリ {category}: パネルID={panel_id}, 書籍名={book_name}, 画像パス={image_path}, 存在={os.path.exists(image_path)}")
    
    # アノテーション情報を取得
    annotation_path = os.path.join(curated_dataset_path, book_name, "annotation.json")
    annotation_data = {}
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            for ann in annotations:
                if ann["id"] == panel_id:
                    annotation_data = ann
                    break
    
    return panel_id, image_path, annotation_data

def draw_annotations_on_image(img, annotation_data):
    """画像にアノテーションを描画"""
    if not annotation_data:
        return img
    
    # PIL画像をnumpy配列に変換
    img_array = np.array(img)
    img_with_annotations = img_array.copy()
    
    # 描画用の画像を作成
    from PIL import ImageDraw
    draw_img = Image.fromarray(img_with_annotations)
    draw = ImageDraw.Draw(draw_img)
    
    # テキストボックスを描画（青色）
    for text_obj in annotation_data.get("text_objects", []):
        bbox = text_obj["bbox"]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=2)
    
    # 顔バウンディングボックスを描画（緑色）
    for face_obj in annotation_data.get("face_objects", []):
        bbox = face_obj["bbox"]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="green", width=2)
    
    # 体バウンディングボックスを描画（黄色）
    for body_obj in annotation_data.get("body_objects", []):
        bbox = body_obj["bbox"]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="yellow", width=2)
    
    # 関係性を描画（紫色の線）
    for relation in annotation_data.get("relations", []):
        if relation["type"] == "text_to_face":
            # テキストと顔を結ぶ線
            text_id = relation["text_id"]
            face_id = relation["face_id"]
            
            # テキストの中心を取得
            text_center = None
            for text_obj in annotation_data.get("text_objects", []):
                if text_obj["id"] == text_id:
                    bbox = text_obj["bbox"]
                    text_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    break
            
            # 顔の中心を取得
            face_center = None
            for face_obj in annotation_data.get("face_objects", []):
                if face_obj["id"] == face_id:
                    bbox = face_obj["bbox"]
                    face_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    break
            
            if text_center and face_center:
                draw.line([text_center, face_center], fill="purple", width=2)
        
        elif relation["type"] == "text_to_body":
            # テキストと体を結ぶ線
            text_id = relation["text_id"]
            body_id = relation["body_id"]
            
            # テキストの中心を取得
            text_center = None
            for text_obj in annotation_data.get("text_objects", []):
                if text_obj["id"] == text_id:
                    bbox = text_obj["bbox"]
                    text_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    break
            
            # 体の中心を取得
            body_center = None
            for body_obj in annotation_data.get("body_objects", []):
                if body_obj["id"] == body_id:
                    bbox = body_obj["bbox"]
                    body_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    break
            
            if text_center and body_center:
                draw.line([text_center, body_center], fill="purple", width=2)
    
    return draw_img

def create_sample_visualization(valid_panels: List[Dict], categories: List[str], curated_dataset_path: str, save_path: str = "sample_images.png"):
    """各カテゴリからランダムに画像を選んでグリッド表示"""
    if not categories:
        print("カテゴリがありません")
        return
    
    # グリッドサイズを決定
    num_categories = len(categories)
    cols = min(5, num_categories)  # 最大5列
    rows = (num_categories + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle('各カテゴリのサンプル画像（アノテーション付き）', fontsize=16)
    
    # 1行の場合はaxesを配列として扱う
    if rows == 1:
        axes = [axes]
    
    for i, category in enumerate(categories):
        row = i // cols
        col = i % cols
        ax = axes[row][col]
        
        # ランダムにサンプルを選択
        panel_id, image_path, annotation_data = get_random_sample_from_category(valid_panels, category, curated_dataset_path)
        
        if panel_id and image_path and os.path.exists(image_path):
            # 画像を読み込んでアノテーションを描画
            img = Image.open(image_path)
            img_with_annotations = draw_annotations_on_image(img, annotation_data)
            ax.imshow(img_with_annotations)
            ax.set_title(f'{category}\n{panel_id}', fontsize=10)
        else:
            # 画像が見つからない場合
            ax.text(0.5, 0.5, f'{category}\nNo Image', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{category}', fontsize=10)
        
        ax.axis('off')
    
    # 余分なサブプロットを非表示
    for i in range(num_categories, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"サンプル画像を保存しました: {save_path}")

def main():
    curated_dataset_path = "curated_dataset"
    
    if not os.path.exists(curated_dataset_path):
        print(f"キュレーションデータセットディレクトリが見つかりません: {curated_dataset_path}")
        return
    
    all_valid_panels, total_all_panels = analyze_all_books(curated_dataset_path)
    
    if not all_valid_panels:
        print("分析対象の書籍が見つかりませんでした。")
        return
    
    categories = categorize_panels(all_valid_panels)
    create_visualization(categories, len(all_valid_panels), total_all_panels)
    
    combinations = analyze_bbox_combinations(all_valid_panels)
    create_combination_visualization(combinations)
    
    # 各カテゴリからサンプル画像を表示（T4まで）
    category_list = [
        "R0", "T1-R1", "T2-R1", "T2-R2", "T3-R1", "T3-R2", "T3-R3",
        "T4-R1", "T4-R2", "T4-R3", "T4-R4"
    ]
    create_sample_visualization(all_valid_panels, category_list, curated_dataset_path)

if __name__ == "__main__":
    main()
