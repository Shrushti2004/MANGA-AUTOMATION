import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import math


class LayoutElement:
    def __init__(self, element_type: str, bbox: List[int]):
        self.element_type = element_type
        self.bbox = bbox
    
    def get_area(self) -> float:
        """要素の面積を計算"""
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))
    
    def __repr__(self):
        return f"LayoutElement(type={self.element_type}, bbox={self.bbox})"

class Layout:
    def __init__(self, elements: List[LayoutElement], width:int, height:int):
        self.width = width
        self.height = height
        self.elements = elements

    def scale(self, scale_factor: float):
        self.width = int(self.width * scale_factor)
        self.height = int(self.height * scale_factor)
        for element in self.elements:
            element.bbox = [int(bbox * scale_factor) for bbox in element.bbox]

    def __repr__(self):
        return f"Layout(elements={self.elements}, width={self.width}, height={self.height})"

def calc_similarity(layout1: Layout, layout2: Layout, aspect_ratio_threshold: float = 0.20, 
                   use_area: bool = True, area_weight: float = 0.1, area_method: str = "ratio") -> float:
    """
    レイアウト類似度を計算する（面積考慮オプション付き）
    
    重み付き二部グラフマッチング問題として定式化:
    max Σ Σ w(ei, ej^) * γij
    s.t. γij ∈ {0,1}, Σγij ≤ 1, Σγij ≤ 1
    
    Args:
        layout1: クエリレイアウト
        layout2: 取得レイアウト
        aspect_ratio_threshold: アスペクト比の閾値
        use_area: 面積を考慮するかどうか
        area_weight: 面積類似度の重み（0.0-1.0）
        area_method: 面積類似度の計算方法（"ratio", "log", "hybrid"）
        
    Returns:
        類似度スコア
    """
    # アスペクト比チェック
    if abs(layout1.width / layout1.height - layout2.width / layout2.height) > aspect_ratio_threshold:
        print(f"aspect ratio mismatch: {layout1.width / layout1.height} != {layout2.width / layout2.height}")
        return 0.0

    # 軸独立スケーリングによりlayout2のサイズをlayout1のサイズに合わせる
    print(f"Scaling layout2 to match layout1")
    print("Before scaling:")
    print(f"layout1: {layout1}")
    print(f"layout2: {layout2}")

    scale_factor = layout1.width / layout2.width
    layout2.scale(scale_factor)

    print("After scaling:")
    print(f"layout1: {layout1}")
    print(f"layout2: {layout2}")

    # 要素が空の場合の処理
    if len(layout1.elements) == 0 or len(layout2.elements) == 0:
        return 0.0
    
    # IoU行列を計算
    iou_matrix = calculate_iou_matrix(layout1.elements, layout2.elements)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    
    # 面積類似度行列を計算（オプション）
    area_matrix = None
    if use_area:
        area_matrix = calculate_area_similarity_matrix(layout1.elements, layout2.elements, method=area_method)
        print(f"Area similarity matrix shape: {area_matrix.shape}")
        print(f"Area similarity matrix:\n{area_matrix}")
    
    # 重み行列を計算（IoUと面積類似度を組み合わせ）
    weight_matrix = calculate_weight_matrix_with_area(
        layout1.elements, layout2.elements, iou_matrix, area_matrix, 
        area_weight=area_weight if use_area else 0.0
    )
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Weight matrix:\n{weight_matrix}")
    
    # ハンガリアンアルゴリズム（Kuhn-Munkres）で最適マッチングを求める
    similarity_score, matching_pairs = solve_bipartite_matching(weight_matrix)
    
    print(f"Similarity score: {similarity_score}")
    print(f"Matching pairs: {matching_pairs}")
    
    return similarity_score


def calculate_area_similarity_matrix(elements1: List[LayoutElement], elements2: List[LayoutElement], 
                                   method: str = "log") -> np.ndarray:
    """
    2つのレイアウト要素リスト間の面積類似度行列を計算
    
    Args:
        elements1: 第1のレイアウト要素リスト
        elements2: 第2のレイアウト要素リスト
        method: 面積類似度の計算方法
                "ratio": min(area1, area2) / max(area1, area2)
                "log": exp(-|log(area1/area2)|)
                "hybrid": ratio * log
        
    Returns:
        面積類似度行列 (m x n)
    """
    if len(elements1) == 0 or len(elements2) == 0:
        return np.zeros((len(elements1), len(elements2)))
    
    m = len(elements1)
    n = len(elements2)
    area_matrix = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            area1 = elements1[i].get_area()
            area2 = elements2[j].get_area()
            
            if area1 <= 0 or area2 <= 0:
                area_matrix[i, j] = 0.0
                continue
            
            if method == "ratio":
                # 面積比による類似度 (0-1の範囲)
                area_matrix[i, j] = min(area1, area2) / max(area1, area2)
                
            elif method == "log":
                # 対数面積比による類似度
                log_ratio = abs(math.log(area1 / area2))
                area_matrix[i, j] = math.exp(-log_ratio)
                
            elif method == "hybrid":
                # ハイブリッド: ratio * log
                ratio_sim = min(area1, area2) / max(area1, area2)
                log_ratio = abs(math.log(area1 / area2))
                log_sim = math.exp(-log_ratio)
                area_matrix[i, j] = ratio_sim * log_sim
                
            else:
                raise ValueError(f"Unknown area similarity method: {method}")
    
    return area_matrix


def calculate_iou_matrix(elements1: List[LayoutElement], elements2: List[LayoutElement]) -> np.ndarray:
    """
    2つのレイアウト要素リスト間のIoU行列を計算
    
    Args:
        elements1: 第1のレイアウト要素リスト
        elements2: 第2のレイアウト要素リスト
        
    Returns:
        IoU行列 (m x n)
    """
    if len(elements1) == 0 or len(elements2) == 0:
        return np.zeros((len(elements1), len(elements2)))
    
    # バウンディングボックスをtensorに変換
    boxes1 = torch.tensor([elem.bbox for elem in elements1], dtype=torch.float32)
    boxes2 = torch.tensor([elem.bbox for elem in elements2], dtype=torch.float32)
    
    # IoU行列を計算
    iou_matrix = box_iou(boxes1, boxes2)
    
    return iou_matrix.numpy()


def calculate_weight_matrix(elements1: List[LayoutElement], elements2: List[LayoutElement], 
                          iou_matrix: np.ndarray) -> np.ndarray:
    """
    論文の式(2)に従って重み行列を計算（後方互換性のため）
    
    w(ei, ej^) = IoU(ei, ej^) if T(ei) = T(ej^), else 0
    
    Args:
        elements1: クエリレイアウト要素リスト
        elements2: 取得レイアウト要素リスト
        iou_matrix: IoU行列
        
    Returns:
        重み行列 (m x n)
    """
    return calculate_weight_matrix_with_area(elements1, elements2, iou_matrix, None, 0.0)


def calculate_weight_matrix_with_area(elements1: List[LayoutElement], elements2: List[LayoutElement], 
                                    iou_matrix: np.ndarray, area_matrix: Optional[np.ndarray] = None,
                                    area_weight: float = 0.0) -> np.ndarray:
    """
    面積を考慮した重み行列を計算
    
    w(ei, ej^) = (1-α)*IoU(ei, ej^) + α*Area_sim(ei, ej^) if T(ei) = T(ej^), else 0
    
    Args:
        elements1: クエリレイアウト要素リスト
        elements2: 取得レイアウト要素リスト
        iou_matrix: IoU行列
        area_matrix: 面積類似度行列（Noneの場合は面積を考慮しない）
        area_weight: 面積類似度の重み（0.0-1.0）
        
    Returns:
        重み行列 (m x n)
    """
    m = len(elements1)
    n = len(elements2)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    # タイプマスクを適用
    weight_matrix = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            # 同じタイプの要素のみに重みを設定
            if elements1[i].element_type == elements2[j].element_type:
                iou_weight = iou_matrix[i, j]
                
                if area_matrix is not None and area_weight > 0.0:
                    # IoUと面積類似度の重み付き線形結合
                    area_sim = area_matrix[i, j]
                    weight_matrix[i, j] = (1.0 - area_weight) * iou_weight + area_weight * area_sim
                else:
                    # 従来通りのIoUのみ
                    weight_matrix[i, j] = iou_weight
            else:
                weight_matrix[i, j] = 0.0
    
    return weight_matrix


def solve_bipartite_matching(weight_matrix: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """
    ハンガリアンアルゴリズム（Kuhn-Munkres）を使用して重み付き二部マッチング問題を解く
    
    論文の式(1)の最適化問題:
    max Σ Σ w(ei, ej^) * γij
    s.t. γij ∈ {0,1}, Σγij ≤ 1, Σγij ≤ 1
    
    Args:
        weight_matrix: 重み行列
        
    Returns:
        Tuple[float, List[Tuple[int, int]]]: (最適スコア, マッチング結果)
    """
    if weight_matrix.size == 0:
        return 0.0, []
    
    # scipyのlinear_sum_assignmentは最小化問題を解くため、重みを負にする
    cost_matrix = -weight_matrix
    
    # ハンガリアンアルゴリズムを実行
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 最適マッチングのスコアを計算
    total_score = 0.0
    matching_pairs: List[Tuple[int, int]] = []
    
    for i, j in zip(row_indices, col_indices):
        if weight_matrix[i, j] > 0:  # 有効なマッチングのみ
            total_score += weight_matrix[i, j]
            matching_pairs.append((int(i), int(j)))
    
    return float(total_score), matching_pairs

