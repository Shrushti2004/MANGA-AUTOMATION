import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def analyze_annotation_file(annotation_path: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    アノテーションファイルを分析して、各画像のカテゴリ数を計算する
    
    Returns:
        Dict[image_id, (related_text_count, nonrelated_text_count, face_count, body_count)]
    """
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_stats = {}
    
    for item in data:
        image_id = item['id']
        
        # text_objectsのIDを取得
        text_ids = set(text_obj['id'] for text_obj in item['text_objects'])
        
        # relationsに含まれるtext_idを取得
        related_text_ids = set()
        for relation in item['relations']:
            if 'text_id' in relation:
                related_text_ids.add(relation['text_id'])
        
        # related_text_objectsとnonrelated_text_objectsの数を計算
        related_text_count = len(related_text_ids)
        nonrelated_text_count = len(text_ids) - related_text_count
        
        # face_objectsとbody_objectsの数を計算
        face_count = len(item['face_objects'])
        body_count = len(item['body_objects'])
        
        image_stats[image_id] = (related_text_count, nonrelated_text_count, face_count, body_count)
    
    return image_stats


def create_category_annotation(dataset_path: str, output_path: str) -> None:
    """
    curated_datasetを分析して、新しいアノテーションファイルを作成する
    """
    # カテゴリ数ベクトルをキーとする辞書
    category_groups: Dict[Tuple[int, int, int, int], List[str]] = defaultdict(list)
    
    # 各マンガディレクトリを処理
    for manga_dir in os.listdir(dataset_path):
        manga_path = os.path.join(dataset_path, manga_dir)
        if not os.path.isdir(manga_path):
            continue
            
        annotation_file = os.path.join(manga_path, 'annotation.json')
        if not os.path.exists(annotation_file):
            print(f"注意: {manga_dir} にはannotation.jsonが見つかりません")
            continue
        
        print(f"処理中: {manga_dir}")
        
        try:
            # アノテーションファイルを分析
            image_stats = analyze_annotation_file(annotation_file)
            
            # 各画像について、カテゴリ数ベクトルを計算してグループ化
            for image_id, (related_text_count, nonrelated_text_count, face_count, body_count) in image_stats.items():
                # 画像ファイルパスを構築
                image_filename = f"{image_id}.png"
                image_path = os.path.join(manga_path, image_filename)
                
                # 画像ファイルが存在するかチェック
                if os.path.exists(image_path):
                    # curated_datasetから見た相対パスを作成
                    relative_image_path = os.path.join(manga_dir, image_filename)
                    category_vector = (related_text_count, nonrelated_text_count, face_count, body_count)
                    category_groups[category_vector].append(relative_image_path)
                else:
                    print(f"警告: 画像ファイルが見つかりません: {image_path}")
                    
        except Exception as e:
            print(f"エラー: {manga_dir} の処理中にエラーが発生しました: {e}")
            continue
    
    # 結果を辞書形式で整理
    result = {}
    for category_vector, image_paths in category_groups.items():
        related_text_count, nonrelated_text_count, face_count, body_count = category_vector
        key = f"related_text:{related_text_count}_nonrelated_text:{nonrelated_text_count}_face:{face_count}_body:{body_count}"
        result[key] = {
            "category_counts": {
                "related_text_objects": related_text_count,
                "nonrelated_text_objects": nonrelated_text_count,
                "face_objects": face_count,
                "body_objects": body_count
            },
            "image_paths": image_paths,
            "count": len(image_paths)
        }
    
    # 結果をJSONファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果を {output_path} に保存しました")
    
    # 統計情報を表示
    print(f"\n統計情報:")
    print(f"ユニークなカテゴリ数ベクトル: {len(result)}")
    total_images = sum(data['count'] for data in result.values())
    print(f"総画像数: {total_images}")
    
    # 上位5つのカテゴリを表示
    sorted_categories = sorted(result.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"\n上位5つのカテゴリ:")
    for i, (key, data) in enumerate(sorted_categories[:5]):
        print(f"{i+1}. {key}: {data['count']}枚")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='curated_datasetディレクトリのパス')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"エラー: 指定されたパス '{args.dataset}' が見つかりません")
        return
    
    # 出力ファイルパスを dataset/database.json に設定
    output_path = os.path.join(args.dataset, 'database.json')
    
    # ファイルが既に存在する場合はスキップ
    if os.path.exists(output_path):
        print(f"database.json が既に存在します: {output_path}")
        print("実行をスキップします")
        return
    
    # カテゴリアノテーションを作成
    create_category_annotation(args.dataset, output_path)


if __name__ == "__main__":
    main()