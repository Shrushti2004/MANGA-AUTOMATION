import json
import os
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple


def load_database(database_path: str) -> Dict:
    """
    database.jsonを読み込む
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file not found: {database_path}")
    
    with open(database_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def search_by_category_counts(database: Dict, related_text: int, nonrelated_text: int, 
                            face_objects: int, body_objects: int) -> List[str]:
    """
    指定されたカテゴリの個数に対応するimage_pathsを取得する
    
    Args:
        database: database.jsonの内容
        related_text: related_text_objectsの個数
        nonrelated_text: nonrelated_text_objectsの個数
        face_objects: face_objectsの個数
        body_objects: body_objectsの個数
    
    Returns:
        List[str]: マッチするimage_pathsのリスト
    """
    # キーを構築
    key = f"related_text:{related_text}_nonrelated_text:{nonrelated_text}_face:{face_objects}_body:{body_objects}"
    
    if key in database:
        return database[key]['image_paths']
    else:
        return []


def search_by_partial_match(database: Dict, **kwargs) -> Dict[str, List[str]]:
    """
    部分的なマッチングで検索する
    
    Args:
        database: database.jsonの内容
        **kwargs: 検索条件（related_text, nonrelated_text, face_objects, body_objects）
    
    Returns:
        Dict[str, List[str]]: マッチするキーとimage_pathsの辞書
    """
    results = {}
    
    for key, data in database.items():
        category_counts = data['category_counts']
        match = True
        
        # 指定された条件をチェック
        for field, value in kwargs.items():
            if field == 'related_text' and category_counts['related_text_objects'] != value:
                match = False
                break
            elif field == 'nonrelated_text' and category_counts['nonrelated_text_objects'] != value:
                match = False
                break
            elif field == 'face_objects' and category_counts['face_objects'] != value:
                match = False
                break
            elif field == 'body_objects' and category_counts['body_objects'] != value:
                match = False
                break
        
        if match:
            results[key] = data['image_paths']
    
    return results





def display_search_results(results: Dict[str, List[str]]) -> None:
    """
    検索結果を表示する
    """
    if not results:
        print("マッチする結果が見つかりませんでした。")
        return
    
    print(f"検索結果: {len(results)}個のカテゴリがマッチしました")
    print("=" * 50)
    
    total_images = 0
    for key, image_paths in results.items():
        print(f"\nカテゴリ: {key}")
        print(f"画像数: {len(image_paths)}")
        total_images += len(image_paths)
        
        # 最初の5つの画像パスを表示
        if len(image_paths) > 5:
            print("画像パス（最初の5つ）:")
            for i, path in enumerate(image_paths[:5]):
                print(f"  {i+1}. {path}")
            print(f"  ... (他{len(image_paths) - 5}個)")
        else:
            print("画像パス:")
            for i, path in enumerate(image_paths):
                print(f"  {i+1}. {path}")
    
    print(f"\n総画像数: {total_images}")


def display_images_grid(image_paths: List[str], output_path: str) -> None:
    """
    画像パスからランダムに16個を選んで4x4のグリッドで表示する
    
    Args:
        image_paths: 画像パスのリスト
        output_path: 出力ファイルのパス
    """
    if not image_paths:
        print("表示する画像がありません。")
        return
    
    # 16個をランダムに選択
    selected_paths = random.sample(image_paths, min(16, len(image_paths)))
    
    # 4x4のグリッドを作成
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    print(f"選択された画像: {len(selected_paths)}個")
    
    for i, image_path in enumerate(selected_paths):
        try:
            # 画像パスを構築（curated_dataset/プレフィックスを追加）
            full_image_path = os.path.join("curated_dataset", image_path)
            
            # 画像を読み込み
            if os.path.exists(full_image_path):
                img = Image.open(full_image_path)
                img_array = np.array(img)
                
                # サブプロットに表示
                ax = axes[i]
                ax.imshow(img_array)
                ax.set_title(os.path.basename(image_path), fontsize=8)
                ax.axis('off')
            else:
                print(f"警告: 画像が見つかりません: {full_image_path}")
                ax = axes[i]
                ax.text(0.5, 0.5, "画像なし", ha='center', va='center')
                ax.set_title(os.path.basename(image_path), fontsize=8)
                ax.axis('off')
                
        except Exception as e:
            print(f"エラー: {image_path} の読み込みに失敗: {e}")
            ax = axes[i]
            ax.text(0.5, 0.5, "エラー", ha='center', va='center')
            ax.set_title(os.path.basename(image_path), fontsize=8)
            ax.axis('off')
    
    # 残りの空のサブプロットを非表示
    for i in range(len(selected_paths), 16):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"検索結果からランダムに選択された画像（{len(selected_paths)}個）", fontsize=16, y=0.98)
    
    # 画像を保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"4x4グリッド画像を {output_path} に保存しました")
    
    # 画像を表示
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='database.jsonから画像を検索')
    parser.add_argument('--database', default='curated_dataset/database.json',
                       help='database.jsonファイルのパス')
    
    # 完全マッチ検索
    parser.add_argument('--related-text', type=int, 
                       help='related_text_objectsの個数')
    parser.add_argument('--nonrelated-text', type=int,
                       help='nonrelated_text_objectsの個数')
    parser.add_argument('--face-objects', type=int,
                       help='face_objectsの個数')
    parser.add_argument('--body-objects', type=int,
                       help='body_objectsの個数')
    

    
    # 出力オプション
    parser.add_argument('--output', help='4x4グリッド画像の出力ファイル名')
    parser.add_argument('--list-only', action='store_true',
                       help='画像パスのみを出力')
    
    args = parser.parse_args()
    
    try:
        # データベースを読み込み
        database = load_database(args.database)
        print(f"データベースを読み込みました: {len(database)}個のカテゴリ")
        
        # 検索実行
        results = {}
        
        # 完全マッチ検索
        if all(x is not None for x in [args.related_text, args.nonrelated_text, 
                                      args.face_objects, args.body_objects]):
            image_paths = search_by_category_counts(
                database, args.related_text, args.nonrelated_text,
                args.face_objects, args.body_objects
            )
            if image_paths:
                key = f"related_text:{args.related_text}_nonrelated_text:{args.nonrelated_text}_face:{args.face_objects}_body:{args.body_objects}"
                results[key] = image_paths
        
        # 部分マッチ検索
        elif any(x is not None for x in [args.related_text, args.nonrelated_text, 
                                        args.face_objects, args.body_objects]):
            search_params = {}
            if args.related_text is not None:
                search_params['related_text'] = args.related_text
            if args.nonrelated_text is not None:
                search_params['nonrelated_text'] = args.nonrelated_text
            if args.face_objects is not None:
                search_params['face_objects'] = args.face_objects
            if args.body_objects is not None:
                search_params['body_objects'] = args.body_objects
            
            results = search_by_partial_match(database, **search_params)
        
        else:
            print("検索条件を指定してください。")
            print("例: python src/search.py --related-text 2 --nonrelated-text 1 --face-objects 1 --body-objects 3")
            return
        
        # 結果を表示
        if args.list_only:
            all_paths = []
            for paths in results.values():
                all_paths.extend(paths)
            for path in all_paths:
                print(path)
        else:
            display_search_results(results)
        
        # 4x4グリッド画像を出力
        if args.output:
            all_paths = []
            for paths in results.values():
                all_paths.extend(paths)
            
            if all_paths:
                display_images_grid(all_paths, args.output)
            else:
                print("画像パスが見つからないため、グリッド画像を作成できません。")
    
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()
