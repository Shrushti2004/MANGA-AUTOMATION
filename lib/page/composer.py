#!/usr/bin/env python3
import os
import json
from typing import List
from PIL import Image

from reportlab.platypus import (
    SimpleDocTemplate,
    Image as RLImage,
    Spacer,
    PageBreak,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


# --------------------------------------------------
# LOAD WINNER IMAGES (robust version)
# --------------------------------------------------
def load_winner_images(scoring_dir: str) -> List[str]:
    winners = []

    if not os.path.exists(scoring_dir):
        print("[ERROR] scoring_dir does not exist:", scoring_dir)
        return winners

    panel_dirs = sorted(
        d for d in os.listdir(scoring_dir)
        if d.startswith("panel") and os.path.isdir(os.path.join(scoring_dir, d))
    )

    for pdir in panel_dirs:
        panel_path = os.path.join(scoring_dir, pdir)
        scores_path = os.path.join(panel_path, "scores.json")
        final_img = None

        # Read winner file
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            winner = data.get("winner", {})
            gen_path = winner.get("generated_image_path", "")

            if gen_path:
                # Convert relative paths to actual paths
                full_gen_path = (
                    gen_path if os.path.isabs(gen_path)
                    else os.path.join(panel_path, os.path.basename(gen_path))
                )

                # Try onlyname
                onlyname = full_gen_path[:-4] + "_onlyname.png"
                if os.path.exists(onlyname):
                    final_img = onlyname
                elif os.path.exists(full_gen_path):
                    final_img = full_gen_path

        # Fallback → 00_anime.png
        if final_img is None:
            fallback = os.path.join(panel_path, "00_anime.png")
            if os.path.exists(fallback):
                final_img = fallback

        if final_img:
            winners.append(final_img)
        else:
            print(f"[WARN] No image found for {pdir}")

    return winners


# --------------------------------------------------
# COMPOSE PDF – 2 panels per page
# --------------------------------------------------
def compose_manga_pdf(scoring_dir: str, output_pdf: str):

    winner_images = load_winner_images(scoring_dir)

    if not winner_images:
        raise RuntimeError("No winner images found.")

    print(f"[COMPOSER] Found {len(winner_images)} panels")

    # PDF setup
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    story = []
    PANEL_W = 120 * mm
    PANEL_H = 120 * mm

    for idx, img_path in enumerate(winner_images):
        try:
            # load once (no overwriting)
            img = Image.open(img_path)
            img = img.convert("RGB")
            resized_path = img_path + "_resized_tmp.jpg"
            img.resize((1024, 1024), Image.BICUBIC).save(resized_path)
        except Exception as e:
            print("[ERROR] Failed image:", img_path, e)
            continue

        story.append(RLImage(resized_path, width=PANEL_W, height=PANEL_H))

        # After two images → new page
        if idx % 2 == 0:
            story.append(Spacer(1, 8 * mm))
        else:
            story.append(PageBreak())

    # If odd → last pagebreak
    if len(winner_images) % 2 == 1:
        story.append(PageBreak())

    doc.build(story)
    print(f"[COMPOSER] PDF created successfully → {output_pdf}")


# --------------------------------------------------
# MAIN (for testing)
# --------------------------------------------------
if __name__ == "__main__":
    compose_manga_pdf(
        scoring_dir="output/scoring",
        output_pdf="output/manga.pdf",
    )
