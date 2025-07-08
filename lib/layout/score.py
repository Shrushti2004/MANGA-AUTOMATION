from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import json
import os
import math
import random
import matplotlib.pyplot as plt

from lib.layout.layout import MangaLayout, from_condition, Speaker, NonSpeaker


def _calc_weight_matrix(layout1, layout2, iou_weight: float):

    boxes1 = torch.tensor([elem.bbox for elem in layout1.elements], dtype=torch.float32)
    boxes2 = torch.tensor([elem.bbox for elem in layout2.elements], dtype=torch.float32)
    
    n = len(boxes1)
    m = len(boxes2)
    if n == 0 or m == 0:
        return None
    
    weight_matrix = box_iou(boxes1, boxes2).numpy()
    for i in range(n):
        for j in range(m):
            type1 = type(layout1.elements[i])
            type2 = type(layout2.elements[j])
            if type1 != type2:
                weight_matrix[i,j] = 0
            if type1 == Speaker and type2 == Speaker:
                text_len1 = layout1.elements[i].text_length
                text_len2 = layout2.elements[j].text_length
                sigma = 10 ** 2
                text_weight = math.exp(-(text_len1 - text_len2)**2 / (2 * sigma))
                weight_matrix[i,j] = iou_weight * weight_matrix[i,j] + (1 - iou_weight) * text_weight
                print(weight_matrix[i,j])

    return weight_matrix

def calc_similarity(
        layout1,
        layout2,
        iou_weight: float,
    ):

    if layout1.width != layout2.width or layout1.height != layout2.height:
        raise ValueError("Layouts must have the same width and height")
    
    layout_score = 0.0
    text_len1 = layout1.unrelated_text_length
    text_len2 = layout2.unrelated_text_length
    sigma = 10 ** 2
    layout_score += math.exp(-(text_len1 - text_len2)**2 / (2 * sigma))

    weight_matrix = _calc_weight_matrix(layout1, layout2, iou_weight)

    if weight_matrix is not None:

        cost_matrix = -weight_matrix
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_indices, col_indices):
            layout_score += weight_matrix[i, j]

    return layout_score


# test function for calc_similarity
if __name__ == "__main__":
    layout1 = MangaLayout(
        image_path="",
        width=100,
        height=100,
        elements=[],
        unrelated_text_length=100
    )

    layout1.adjust(512, 512)

    annfile = "./curated_dataset/database.json"
    num_speakers = 0
    num_non_speakers = 0
    base_text_length = 0
    text_length_threshold = 70
    base_width = 512
    base_height = 512
    aspect_ratio_threshold = 0.2
    layouts = from_condition(annfile, num_speakers, num_non_speakers, base_text_length, text_length_threshold, base_width, base_height, aspect_ratio_threshold, True)
    
    # 全てのlayoutについてスコアを計算
    layout_scores = []
    for layout in layouts:
        score = calc_similarity(layout1, layout, 0.4)
        layout_scores.append((layout, score))
    
    # スコアでソート（降順）
    layout_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位4つを取得
    top_4_layouts = layout_scores[:4]
    
    # 可視化
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    
    # layout1を表示
    layout1.plot_data(ax[0])
    ax[0].set_title("Query Layout")
    
    # 上位4つのlayoutを表示
    for i, (layout, score) in enumerate(top_4_layouts):
        layout.plot_data(ax[i+1])
        ax[i+1].set_title(f"Rank {i+1}\nScore: {score:.4f}")
    
    plt.tight_layout()
    plt.savefig("top_4_similar_layouts.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 結果を表示
    print(f"Query layout vs Top 4 similar layouts:")
    for i, (layout, score) in enumerate(top_4_layouts):
        print(f"Rank {i+1}: Score = {score:.4f}")