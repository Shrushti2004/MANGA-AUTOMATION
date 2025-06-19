from util.curate_dataset import curate_book, visualize_panels
from PIL import Image, ImageDraw, ImageFont
import os


def test_curate_book_01():
    print("=== AisazuNihaIrarenai のキュレーション結果 ===")
    curate_book("AisazuNihaIrarenai", "curated_dataset")

if __name__ == "__main__":
    test_curate_book_01()