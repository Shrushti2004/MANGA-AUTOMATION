#!/usr/bin/env python3
"""
lib/layout/score.pyの動作をテストするスクリプト
"""

import sys
import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.layout.score import Layout, LayoutElement, calc_similarity, calculate_weight_matrix, calculate_iou_matrix, solve_bipartite_matching


def load_database(database_path: str) -> dict:
    """database.jsonを読み込む"""
    with open(database_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def search_by_category_counts(database: dict, related_text: int, nonrelated_text: int, 
                            face_objects: int, body_objects: int) -> list:
    """指定されたカテゴリの個数に対応する画像情報を取得する（新しいデータベース構造対応）"""
    key = f"related_text:{related_text}_nonrelated_text:{nonrelated_text}_face:{face_objects}_body:{body_objects}"
    
    if key in database:
        category_data = database[key]
        
        # 新しい形式の場合は'images'フィールドを使用
        if 'images' in category_data:
            return category_data['images']  # [{"path": "...", "width": ..., "height": ...}, ...]
        
        # 後方互換性：古い形式の場合は'image_paths'を使用
        elif 'image_paths' in category_data:
            return category_data['image_paths']  # ["path1", "path2", ...]
        
        else:
            return []
    else:
        return []


def load_annotation_from_image_path(image_path: str) -> dict:
    """画像パスから対応するアノテーションデータを取得"""
    # 画像パスからディレクトリを取得 (例: "manga1/image1.png" -> "manga1")
    manga_dir = os.path.dirname(image_path)
    
    # annotation.jsonのパスを構築
    annotation_path = os.path.join("curated_dataset", manga_dir, "annotation.json")
    
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 画像IDを取得 (例: "image1.png" -> "image1")
    image_filename = os.path.basename(image_path)
    image_id = os.path.splitext(image_filename)[0]
    
    # 対応する画像のアノテーションデータを検索
    for item in data:
        if item['id'] == image_id:
            return item
    
    raise ValueError(f"No annotation found for image: {image_id}")


def load_annotation_from_image_info(image_info: dict) -> dict:
    """画像情報から対応するアノテーションデータを取得（新しいデータベース構造対応）"""
    if isinstance(image_info, str):
        # 後方互換性：文字列の場合は古い形式として処理
        return load_annotation_from_image_path(image_info)
    
    image_path = image_info['path']
    return load_annotation_from_image_path(image_path)


def create_layout_from_annotation(annotation_data: dict) -> Layout:
    """アノテーションデータからLayoutオブジェクトを作成"""
    elements = []
    
    # face_objectsを追加
    for face_obj in annotation_data.get('face_objects', []):
        elements.append(LayoutElement('face', face_obj['bbox']))
    
    # body_objectsを追加
    for body_obj in annotation_data.get('body_objects', []):
        elements.append(LayoutElement('body', body_obj['bbox']))
    
    # text_objectsを追加（relationsを基にrelated/nonrelatedを判定）
    related_text_ids = set()
    for relation in annotation_data.get('relations', []):
        if 'text_id' in relation:
            related_text_ids.add(relation['text_id'])
    
    for text_obj in annotation_data.get('text_objects', []):
        text_type = 'related_text' if text_obj['id'] in related_text_ids else 'nonrelated_text'
        elements.append(LayoutElement(text_type, text_obj['bbox']))
    
    # 画像サイズを取得
    width = annotation_data.get('frame_width', 1000)
    height = annotation_data.get('frame_height', 1000)
    
    return Layout(elements, width, height)


def draw_layout_visualization(layout: Layout, ax, title: str = "Layout"):
    """レイアウトをrectangleで可視化"""
    # 要素タイプ別の色設定
    type_colors = {
        'face': 'red',
        'body': 'blue',
        'related_text': 'green',
        'nonrelated_text': 'orange'
    }
    
    # 白い背景を設定
    ax.set_xlim(0, layout.width)
    ax.set_ylim(0, layout.height)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    
    # Y軸を反転（画像座標系に合わせる）
    ax.invert_yaxis()
    
    # 各要素をrectangleで描画
    for i, elem in enumerate(layout.elements):
        x1, y1, x2, y2 = elem.bbox
        width = x2 - x1
        height = y2 - y1
        
        # rectangleを作成
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=type_colors.get(elem.element_type, 'black'),
            facecolor=type_colors.get(elem.element_type, 'gray'),
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # 要素番号とタイプを表示
        ax.text(x1 + width/2, y1 + height/2, f"{i}\n{elem.element_type}", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 凡例を作成
    legend_elements = []
    for elem_type, color in type_colors.items():
        if any(elem.element_type == elem_type for elem in layout.elements):
            legend_elements.append(patches.Patch(color=color, alpha=0.3, label=elem_type))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))


def filter_by_aspect_ratio(layout1: Layout, matching_images: list, aspect_ratio_threshold: float = 0.20) -> list:
    """layout1のアスペクト比と画像のアスペクト比の差がthreshold以下のものに限定してフィルタリング"""
    layout1_aspect_ratio = layout1.width / layout1.height
    filtered_images = []
    
    print(f"アスペクト比フィルタリング開始...")
    print(f"layout1のアスペクト比: {layout1_aspect_ratio:.3f}")
    print(f"閾値: {aspect_ratio_threshold}")
    
    for image_item in matching_images:
        try:
            # 新しいデータベース構造への対応
            if isinstance(image_item, dict):
                # 新しい形式：{"path": "...", "width": ..., "height": ...}
                image_path = image_item['path']
                width = image_item.get('width', 1000)
                height = image_item.get('height', 1000)
            else:
                # 古い形式：文字列パス - アノテーションから取得
                image_path = image_item
                annotation_data = load_annotation_from_image_path(image_path)
                width = annotation_data.get('frame_width', 1000)
                height = annotation_data.get('frame_height', 1000)
            
            # アスペクト比計算
            image_aspect_ratio = width / height
            aspect_ratio_diff = abs(layout1_aspect_ratio - image_aspect_ratio)
            
            # 閾値以下の場合のみフィルタリング結果に追加
            if aspect_ratio_diff <= aspect_ratio_threshold:
                filtered_images.append(image_item)
                
        except Exception as e:
            print(f"エラー: {image_path if 'image_path' in locals() else 'unknown'} のアスペクト比計算に失敗: {e}")
            continue
    
    print(f"フィルタリング結果: {len(matching_images)} → {len(filtered_images)} 画像")
    print(f"除外された画像数: {len(matching_images) - len(filtered_images)}")
    
    return filtered_images


def calculate_similarity_for_all_images(layout1: Layout, matching_images: list) -> list:
    """全てのマッチング画像に対して類似度を計算（tqdmで進捗表示、100枚ごとに結果更新）"""
    results = []
    update_interval = 100  # 100枚ごとに結果を更新
    
    print(f"全{len(matching_images)}枚の画像に対して類似度計算を実行中...")
    
    # tqdmで進捗バーを表示
    with tqdm(total=len(matching_images), desc="類似度計算中", unit="画像") as pbar:
        for i, image_item in enumerate(matching_images):
            try:
                # 新しいデータベース構造への対応
                if isinstance(image_item, dict):
                    # 新しい形式：{"path": "...", "width": ..., "height": ...}
                    image_path = image_item['path']
                    original_width = image_item.get('width', 1000)
                    original_height = image_item.get('height', 1000)
                else:
                    # 古い形式：文字列パス
                    image_path = image_item
                    original_width = 1000
                    original_height = 1000
                
                # アノテーションデータを取得
                annotation_data = load_annotation_from_image_path(image_path)
                
                # layout2を作成
                layout2 = create_layout_from_annotation(annotation_data)
                
                # layout2のコピーを作成（スケーリングで元のlayout2が変更されるため）
                layout2_copy = Layout(
                    elements=[LayoutElement(elem.element_type, elem.bbox.copy()) for elem in layout2.elements],
                    width=layout2.width,
                    height=layout2.height
                )
                
                # 類似度計算（デバッグ出力を抑制）
                similarity = calc_similarity(layout1, layout2_copy)
                
                results.append({
                    'image_path': image_path,
                    'image_id': annotation_data['id'],
                    'similarity_score': similarity,
                    'layout2': layout2_copy,
                    'annotation_data': annotation_data,
                    'original_width': original_width,
                    'original_height': original_height
                })
                
                # 進捗バーを更新
                pbar.update(1)
                
                # 100枚ごとに途中結果を表示
                if (i + 1) % update_interval == 0:
                    display_intermediate_results(layout1, results, i + 1, len(matching_images))
                    
            except Exception as e:
                pbar.set_postfix({"エラー": f"{os.path.basename(image_path) if 'image_path' in locals() else 'unknown'}"})
                continue
    
    return results


def display_intermediate_results(layout1: Layout, results: list, processed_count: int, total_count: int):
    """途中結果を表示"""
    if not results:
        return
    
    # 現在の統計を計算
    non_zero_scores = [r for r in results if r['similarity_score'] > 0]
    zero_scores = [r for r in results if r['similarity_score'] == 0]
    
    print(f"\n{'='*60}")
    print(f"途中結果 ({processed_count}/{total_count} 枚処理完了)")
    print(f"{'='*60}")
    print(f"処理完了画像数: {len(results)}")
    print(f"スコア > 0 の画像数: {len(non_zero_scores)}")
    print(f"スコア = 0 の画像数: {len(zero_scores)}")
    
    if len(non_zero_scores) > 0:
        scores = [r['similarity_score'] for r in non_zero_scores]
        print(f"非零スコア統計:")
        print(f"  最大スコア: {max(scores):.4f}")
        print(f"  平均スコア: {np.mean(scores):.4f}")
        print(f"  最小スコア: {min(scores):.4f}")
        print(f"  標準偏差: {np.std(scores):.4f}")
        
        # 途中結果を表示
        display_top_results(layout1, results, top_n=5, is_intermediate=True, processed_count=processed_count)


def display_top_results(layout1: Layout, results: list, top_n: int = 5, is_intermediate: bool = False, processed_count: int = 0):
    """上位N個の結果をmatplotlibで表示（inputレイアウトも含む）"""
    # スコアでソート
    sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    top_results = sorted_results[:top_n]
    
    if len(top_results) == 0:
        print("表示する結果がありません")
        return
    
    # レイアウト：左側にlayout1、右側に上位結果を2行で表示
    n_cols = 3  # layout1用に1列 + 結果用に2列
    n_rows = max(1, (len(top_results) + 1) // 2)  # 上位結果を2列で表示
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    
    # axesを1次元配列に変換
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()
    
    print(f"\n=== 上位{len(top_results)}件の結果 ===")
    
    # 左側にlayout1を表示（インデックス0）
    layout1_ax = axes_flat[0]
    draw_layout_visualization(layout1, layout1_ax, "Query Layout (layout1)")
    
    # 右側に上位結果を表示
    for i, result in enumerate(top_results):
        image_path = result['image_path']
        similarity_score = result['similarity_score']
        image_id = result['image_id']
        
        print(f"{i+1}. {image_path} (スコア: {similarity_score:.4f})")
        
        # 結果画像の位置を計算
        result_row = i // 2
        result_col = (i % 2) + 1  # layout1の右側から開始
        result_index = result_row * n_cols + result_col
        
        if result_index < len(axes_flat):
            ax = axes_flat[result_index]
            
            # 画像を読み込み
            try:
                full_image_path = os.path.join("curated_dataset", image_path)
                if os.path.exists(full_image_path):
                    img = Image.open(full_image_path)
                    img_array = np.array(img)
                    
                    # サブプロットに表示
                    ax.imshow(img_array)
                    ax.set_title(f"#{i+1}: {image_id}\nスコア: {similarity_score:.4f}", fontsize=10)
                    ax.axis('off')
                else:
                    print(f"  警告: 画像が見つかりません: {full_image_path}")
                    ax.text(0.5, 0.5, "画像なし", ha='center', va='center')
                    ax.set_title(f"#{i+1}: {image_id}\nスコア: {similarity_score:.4f}", fontsize=10)
                    ax.axis('off')
                    
            except Exception as e:
                print(f"  エラー: {image_path} の読み込みに失敗: {e}")
                ax.text(0.5, 0.5, "エラー", ha='center', va='center')
                ax.set_title(f"#{i+1}: {image_id}\nスコア: {similarity_score:.4f}", fontsize=10)
                ax.axis('off')
    
    # 残りの空のサブプロットを非表示
    total_plots = n_rows * n_cols
    used_plots = 1 + len(top_results)  # layout1 + 結果画像
    
    for i in range(used_plots, total_plots):
        if i < len(axes_flat):
            axes_flat[i].axis('off')
    
    plt.tight_layout()
    
    # タイトルを途中結果か最終結果かで変更
    if is_intermediate:
        title = f"クエリレイアウト vs 類似度スコア上位{len(top_results)}件 (途中結果: {processed_count}枚処理済み)"
        output_path = f"visualized/intermediate_results_{processed_count}.png"
    else:
        title = f"クエリレイアウト vs 類似度スコア上位{len(top_results)}件 (最終結果)"
        output_path = "visualized/top_similarity_results.png"
    
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # 画像を保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if is_intermediate:
        print(f"\n途中結果画像を {output_path} に保存しました")
    else:
        print(f"\n最終結果画像を {output_path} に保存しました")
    
    # 画像を表示
    plt.show()


def test_scoring():
    """レイアウト類似度計算のテスト"""
    
    # layout1: 固定のテストレイアウト
    layout1 = Layout(
        elements=[
            LayoutElement('face', [600, 300, 900, 600]),
        ],
        width=1000,
        height=1000
    )
    
    # 検索条件を定義
    num_face = 1
    num_body = 1
    num_related_text = 1
    num_nonrelated_text = 0
    
    print("=== レイアウト類似度計算テスト（全画像対象、リアルタイム更新） ===")
    print(f"layout1 要素数: {len(layout1.elements)}")
    print(f"layout1 サイズ: {layout1.width}x{layout1.height}")
    
    # layout1の詳細を表示
    print("\nlayout1 の詳細:")
    for i, elem in enumerate(layout1.elements):
        print(f"  {i}: {elem}")
    
    # 要素タイプ別の統計
    def get_element_stats(layout):
        stats = {}
        for elem in layout.elements:
            stats[elem.element_type] = stats.get(elem.element_type, 0) + 1
        return stats
    
    layout1_stats = get_element_stats(layout1)
    print(f"\nlayout1 タイプ別統計: {layout1_stats}")
    
    try:
        # database.jsonを読み込み
        database_path = "curated_dataset/database.json"
        database = load_database(database_path)
        print(f"\nデータベースを読み込みました: {len(database)}個のカテゴリ")
        
        # 指定された要素数の画像を検索
        matching_images = search_by_category_counts(
            database, num_related_text, num_nonrelated_text, num_face, num_body
        )
        
        print(f"\n検索条件: related_text={num_related_text}, nonrelated_text={num_nonrelated_text}, face={num_face}, body={num_body}")
        print(f"マッチした画像数: {len(matching_images)}")
        
        # 画像サイズの統計を表示（新しいデータベース構造の場合）
        if matching_images and isinstance(matching_images[0], dict):
            widths = [img.get('width', 1000) for img in matching_images]
            heights = [img.get('height', 1000) for img in matching_images]
            
            print(f"\n画像サイズ統計:")
            print(f"  幅: 最小={min(widths)}, 最大={max(widths)}, 平均={sum(widths)/len(widths):.1f}")
            print(f"  高さ: 最小={min(heights)}, 最大={max(heights)}, 平均={sum(heights)/len(heights):.1f}")
            
            # アスペクト比の統計
            aspect_ratios = [w/h for w, h in zip(widths, heights)]
            print(f"  アスペクト比: 最小={min(aspect_ratios):.2f}, 最大={max(aspect_ratios):.2f}, 平均={sum(aspect_ratios)/len(aspect_ratios):.2f}")
        
        if not matching_images:
            print("マッチする画像が見つかりませんでした。")
            return
        
        # アスペクト比フィルタリング
        print(f"\n{'='*60}")
        print("アスペクト比フィルタリング...")
        print(f"{'='*60}")
        
        filtered_images = filter_by_aspect_ratio(layout1, matching_images, aspect_ratio_threshold=0.20)
        
        if not filtered_images:
            print("アスペクト比フィルタリング後にマッチする画像が見つかりませんでした。")
            return
        
        # 全画像に対して類似度計算（tqdmで進捗表示、100枚ごとに結果更新）
        print(f"\n{'='*60}")
        print("全画像に対する類似度計算を開始...")
        print("※100枚ごとに途中結果を表示します")
        print(f"{'='*60}")
        
        results = calculate_similarity_for_all_images(layout1, filtered_images)
        
        # 最終結果の分析
        print(f"\n{'='*60}")
        print("最終結果分析")
        print(f"{'='*60}")
        
        total_images = len(results)
        non_zero_scores = [r for r in results if r['similarity_score'] > 0]
        zero_scores = [r for r in results if r['similarity_score'] == 0]
        
        print(f"処理完了画像数: {total_images}")
        print(f"スコア > 0 の画像数: {len(non_zero_scores)}")
        print(f"スコア = 0 の画像数: {len(zero_scores)}")
        
        if len(non_zero_scores) > 0:
            scores = [r['similarity_score'] for r in non_zero_scores]
            print(f"非零スコア統計:")
            print(f"  最大スコア: {max(scores):.4f}")
            print(f"  平均スコア: {np.mean(scores):.4f}")
            print(f"  最小スコア: {min(scores):.4f}")
            print(f"  標準偏差: {np.std(scores):.4f}")
            
            # 最終結果を表示
            display_top_results(layout1, results, top_n=5, is_intermediate=False)
        else:
            print("非零スコアの画像が見つかりませんでした。")
            # それでもlayout1だけでも表示
            display_top_results(layout1, [], top_n=0, is_intermediate=False)
        
        # テスト結果を要約
        print(f"\n{'='*60}")
        print("テスト結果要約")
        print(f"{'='*60}")
        print(f"✓ データベース検索: 成功 ({len(matching_images)}個の画像)")
        print(f"✓ 類似度計算: 成功 ({total_images}個の画像を処理)")
        print(f"✓ 非零スコア: {len(non_zero_scores)}個 ({len(non_zero_scores)/total_images*100:.1f}%)")
        print(f"✓ リアルタイム更新: 成功 (100枚ごとに途中結果表示)")
        print(f"✓ 最終結果表示: 成功")
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("curated_dataset/database.jsonが見つかりません。")
        print("先に src/dataprepare.py を実行してください。")
    except Exception as e:
        print(f"予期しないエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_scoring()
    