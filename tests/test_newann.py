import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


def test_database_annotation():
    """
    database.jsonファイルを読み込んで、各カテゴリの画像数を列挙し、
    2つの要素の個数を縦軸・横軸にしたヒストグラムを作成してビジュアライズする
    """
    # database.jsonファイルのパス
    database_path = "curated_dataset/database.json"
    
    # ファイルが存在するかチェック
    if not os.path.exists(database_path):
        print(f"エラー: {database_path} が見つかりません")
        return
    
    # database.jsonを読み込み
    with open(database_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== データベースアノテーション分析 ===")
    print(f"総カテゴリ数: {len(data)}")
    print()
    
    # 各要素ごとにimage_pathsの数を列挙
    print("各カテゴリの画像数:")
    category_data_list = []
    
    for category_key, category_data in data.items():
        image_count = len(category_data['image_paths'])
        print(f"{category_key}: {image_count}個")
        
        # カテゴリ情報を保存
        counts = category_data['category_counts']
        stats = category_data.get('statistics', {})
        
        category_data_list.append({
            'key': category_key,
            'image_count': image_count,
            'related_humans': counts['related_humans'],
            'non_related_humans': counts['non_related_humans'],
            'total_humans': counts['related_humans'] + counts['non_related_humans'],
            'total_text_chars': stats.get('total_related_text_chars', 0),
            'avg_text_chars': stats.get('avg_related_text_chars_per_image', 0)
        })
    
    print(f"\n総カテゴリ数: {len(category_data_list)}")
    
    # 2次元ヒストグラムを作成
    create_2d_histograms(category_data_list)


def create_2d_histograms(category_data_list):
    """
    2つの要素の組み合わせごとに2次元ヒストグラムを作成
    """
    # 要素の組み合わせを定義
    combinations = [
        ('related_humans', 'non_related_humans', 'Related Humans', 'Non-related Humans'),
        ('related_humans', 'total_humans', 'Related Humans', 'Total Humans'),
        ('non_related_humans', 'total_humans', 'Non-related Humans', 'Total Humans'),
        ('related_humans', 'avg_text_chars', 'Related Humans', 'Avg Text Chars per Image'),
        ('total_humans', 'avg_text_chars', 'Total Humans', 'Avg Text Chars per Image'),
        ('non_related_humans', 'avg_text_chars', 'Non-related Humans', 'Avg Text Chars per Image')
    ]
    
    # 2x3のサブプロットを作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (x_key, y_key, x_label, y_label) in enumerate(combinations):
        # データを抽出
        x_values = [item[x_key] for item in category_data_list]
        y_values = [item[y_key] for item in category_data_list]
        weights = [item['image_count'] for item in category_data_list]
        
        # 軸の最大値を設定（文字数の場合は動的に設定）
        if 'text_chars' in y_key:
            y_max = min(int(max(y_values)) + 1, 100)  # 最大100文字まで
        else:
            y_max = 20
        x_max = 20
        
        # ヒストグラムのビンを作成
        if 'text_chars' in y_key:
            y_bins = np.linspace(0, y_max, 21)  # 文字数は連続値として扱う
        else:
            y_bins = np.arange(0, y_max + 1)
        x_bins = np.arange(0, x_max + 1)
        
        # 2次元ヒストグラムを作成
        hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins], weights=weights)
        
        # ヒートマップを描画
        ax = axes[i]
        
        if 'text_chars' in y_key:
            # 文字数の場合はimshowを使用
            im = ax.imshow(hist.T, cmap='viridis', origin='lower', aspect='auto',
                          extent=(0, x_max, 0, y_max))
        else:
            # 整数の場合はpcolormeshを使用
            X, Y = np.meshgrid(np.arange(x_max + 1), np.arange(y_max + 1))
            im = ax.pcolormesh(X, Y, hist.T, cmap='viridis', shading='flat')
        
        # カラーバーを追加
        plt.colorbar(im, ax=ax, label='画像数')
        
        # ラベルとタイトルを設定
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{x_label} vs {y_label}')
        
        # 軸の範囲と目盛りを設定
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xticks(np.arange(0, x_max + 1, 2))
        
        if 'text_chars' in y_key:
            ax.set_yticks(np.arange(0, y_max + 1, 10))
        else:
            ax.set_yticks(np.arange(0, y_max + 1, 2))
        
        # アスペクト比を設定
        if 'text_chars' in y_key:
            ax.set_aspect('auto')
        else:
            ax.set_aspect('equal')
        
        # 値を表示（画像数が多い場合は省略）
        if len(category_data_list) < 50 and 'text_chars' not in y_key:
            for j in range(x_max):
                for k in range(y_max):
                    value = hist[j, k]
                    if value > 0:
                        ax.text(j + 0.5, k + 0.5, f'{int(value)}',
                               ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('カテゴリ要素の2次元分布（色の濃さ＝画像数）', fontsize=16, y=1.02)
    plt.show()
    
    # 結果を保存
    output_path = "visualized/category_2d_histograms.png"
    os.makedirs("visualized", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n2次元ヒストグラムを {output_path} に保存しました")


def create_individual_2d_histogram(category_data_list, x_key, y_key, x_label, y_label):
    """
    個別の2次元ヒストグラムを作成
    """
    # データを抽出
    x_values = [item[x_key] for item in category_data_list]
    y_values = [item[y_key] for item in category_data_list]
    weights = [item['image_count'] for item in category_data_list]
    
    # 軸の最大値を設定
    if 'text_chars' in y_key:
        y_max = min(int(max(y_values)) + 1, 100)
    else:
        y_max = 20
    x_max = 20
    
    # ヒストグラムのビンを作成
    if 'text_chars' in y_key:
        y_bins = np.linspace(0, y_max, 21)
    else:
        y_bins = np.arange(0, y_max + 1)
    x_bins = np.arange(0, x_max + 1)
    
    # 2次元ヒストグラムを作成
    hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins], weights=weights)
    
    # 大きなサイズでプロット
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # ヒートマップを描画
    if 'text_chars' in y_key:
        im = ax.imshow(hist.T, cmap='viridis', origin='lower', aspect='auto',
                      extent=(0, x_max, 0, y_max))
    else:
        X, Y = np.meshgrid(np.arange(x_max + 1), np.arange(y_max + 1))
        im = ax.pcolormesh(X, Y, hist.T, cmap='viridis', shading='flat')
    
    # カラーバーを追加
    plt.colorbar(im, ax=ax, label='画像数')
    
    # ラベルとタイトルを設定
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{x_label} vs {y_label}の分布', fontsize=14)
    
    # 軸の範囲と目盛りを設定
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xticks(np.arange(0, x_max + 1, 2))
    
    if 'text_chars' in y_key:
        ax.set_yticks(np.arange(0, y_max + 1, 10))
        ax.set_aspect('auto')
    else:
        ax.set_yticks(np.arange(0, y_max + 1, 2))
        ax.set_aspect('equal')
    
    # 値を表示（文字数以外の場合）
    if 'text_chars' not in y_key:
        for i in range(x_max):
            for j in range(y_max):
                value = hist[i, j]
                if value > 0:
                    ax.text(i + 0.5, j + 0.5, f'{int(value)}',
                           ha='center', va='center', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 結果を保存
    output_path = f"visualized/{x_key}_vs_{y_key}_histogram.png"
    os.makedirs("visualized", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{x_label} vs {y_label} のヒストグラムを {output_path} に保存しました")


def test_category_statistics():
    """
    カテゴリ統計の詳細分析
    """
    database_path = "curated_dataset/database.json"
    
    if not os.path.exists(database_path):
        print(f"エラー: {database_path} が見つかりません")
        return
    
    with open(database_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n=== カテゴリ統計詳細分析 ===")
    
    # 画像数順にソート
    sorted_categories = sorted(data.items(), key=lambda x: len(x[1]['image_paths']), reverse=True)
    
    print("画像数順ランキング（上位10位）:")
    for i, (category_key, category_data) in enumerate(sorted_categories[:10]):
        image_count = len(category_data['image_paths'])
        counts = category_data['category_counts']
        stats = category_data.get('statistics', {})
        print(f"{i+1:2d}. {category_key}: {image_count}個")
        print(f"    related_humans:{counts['related_humans']}, "
              f"non_related_humans:{counts['non_related_humans']}")
        print(f"    平均関連文字数: {stats.get('avg_related_text_chars_per_image', 0):.1f}")
    
    # 統計情報
    total_images = sum(len(category_data['image_paths']) for category_data in data.values())
    total_text_chars = sum(category_data.get('statistics', {}).get('total_related_text_chars', 0) 
                          for category_data in data.values())
    
    print(f"\n総画像数: {total_images}")
    print(f"平均画像数/カテゴリ: {total_images / len(data):.2f}")
    print(f"総関連文字数: {total_text_chars}")
    print(f"平均関連文字数/画像: {total_text_chars / total_images:.2f}")
    
    # 関連人間数別の統計
    related_humans_stats = {}
    for category_data in data.values():
        related_count = category_data['category_counts']['related_humans']
        if related_count not in related_humans_stats:
            related_humans_stats[related_count] = {'images': 0, 'categories': 0, 'text_chars': 0}
        related_humans_stats[related_count]['images'] += len(category_data['image_paths'])
        related_humans_stats[related_count]['categories'] += 1
        related_humans_stats[related_count]['text_chars'] += category_data.get('statistics', {}).get('total_related_text_chars', 0)
    
    print(f"\n関連人間数別統計:")
    for related_count in sorted(related_humans_stats.keys()):
        stats = related_humans_stats[related_count]
        avg_chars = stats['text_chars'] / stats['images'] if stats['images'] > 0 else 0
        print(f"  {related_count}人: {stats['categories']}カテゴリ, {stats['images']}画像, 平均文字数: {avg_chars:.1f}")


if __name__ == "__main__":
    test_database_annotation()
    test_category_statistics()
