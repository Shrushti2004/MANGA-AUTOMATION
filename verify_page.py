#!/usr/bin/env python3
import os
import json
import time
from tqdm import tqdm

# local modules (your project)
from lib.image.image import generate_image_with_sd
from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes, check_open
from lib.layout.layout import generate_layout, similar_layouts
from lib.name.name import generate_name, generate_animepose_image
from lib.scoring.scorer import (
    load_clip_model,
    get_verification_prompt,
    calculate_clip_score,
    calculate_geometric_penalty,
)
from lib.page.composer import compose_manga_pdf

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, default=str)


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------
def run_offline_pipeline(
    refined_elements_path,
    enhanced_prompts_path,
    panels_path,
    output_dir,
    num_images=3,
    num_names=5,
    controlnet_check=True,
):
    # ---------------------------------------------------------
    # NEW FOLDER WITH DATE + TIME
    # ---------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_out_dir = os.path.join(output_dir, timestamp)
    os.makedirs(final_out_dir, exist_ok=True)

    print(f"\n[INFO] Output directory created: {final_out_dir}")

    # ---------------------------------------------------------
    # 1. Load input JSON files
    # ---------------------------------------------------------
    print("Loading local files...")
    with open(refined_elements_path, "r", encoding="utf-8") as f:
        refined = json.load(f)
    with open(enhanced_prompts_path, "r", encoding="utf-8") as f:
        enhanced_prompts = json.load(f)
    with open(panels_path, "r", encoding="utf-8") as f:
        panels = json.load(f)

    if len(enhanced_prompts) != len(panels):
        print("[WARN] prompt count and panel count differ. Using minimum length.")
    panel_count = min(len(enhanced_prompts), len(panels))

    # ---------------------------------------------------------
    # 2. Check ControlNet (Optional)
    # ---------------------------------------------------------
    if controlnet_check:
        ok = check_open()
        if not ok:
            print("[WARN] ControlNet/SD endpoint not reachable!")

    # ---------------------------------------------------------
    # 3. Load CLIP once
    # ---------------------------------------------------------
    print("Loading CLIP model...")
    clip_model, clip_processor, device = load_clip_model()

    final_summary = {
        "panels": [],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": final_out_dir
    }

    # ---------------------------------------------------------
    # 4. Process each panel
    # ---------------------------------------------------------
    for i in range(panel_count):
        prompt = enhanced_prompts[i]
        panel = panels[i]

        panel_dir = os.path.join(final_out_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)

        panel_entry = {
            "panel_index": i,
            "prompt": prompt,
            "variations": []
        }

        print(f"\n[Panel {i}] prompt_length={len(prompt)} elements={len(panel)}")

        for j in range(num_images):
            print(f"  - Generating variation {j} ...")

            # Base image
            image_path = os.path.join(panel_dir, f"{j:02d}.png")
            try:
                generate_image_with_sd(prompt, image_path)
            except Exception as e:
                print(f"    [ERROR] SD generation failed: {e}")
                continue

            # Anime pose
            anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
            try:
                generate_animepose_image(image_path, prompt, anime_image_path)
            except Exception as e:
                print(f"    [WARN] generate_animepose_image failed: {e}")
                anime_image_path = None

            # ControlNet OpenPose
            try:
                openpose_result = run_controlnet_openpose(image_path, anime_image_path)
            except Exception as e:
                print(f"    [ERROR] OpenPose failed: {e}")
                continue

            # Bounding boxes
            try:
                bboxes = controlnet2bboxes(openpose_result)
            except Exception as e:
                print(f"    [WARN] controlnet2bboxes failed: {e}")
                cw, ch = getattr(openpose_result, "canvas_width", 512), getattr(openpose_result, "canvas_height", 512)
                bboxes = [[cw//4, ch//4, 3*cw//4, 3*ch//4]]

            layout = generate_layout(bboxes, panel, openpose_result.canvas_width, openpose_result.canvas_height)
            if layout is None:
                print("    [WARN] invalid layout found.")
                continue

            scored_layouts = similar_layouts(layout)
            layout_options = []

            for idxL, item in enumerate(scored_layouts[:num_names]):
                try:
                    ref_layout = item[0]
                    sim_score = safe_float(item[1], 0.0)

                    geom_penalty = calculate_geometric_penalty(ref_layout, panel, openpose_result)

                    save_name_path = os.path.join(panel_dir, f"{j:02d}_name_{idxL}.png")

                    try:
                        generate_name(openpose_result, layout, item, panel, save_name_path)
                    except Exception as e:
                        print(f"      [WARN] generate_name failed: {e}")

                    # Ensure saved path is absolute when storing
                    abs_save_name_path = os.path.abspath(save_name_path)

                    layout_options.append({
                        "rank": idxL,
                        "template_path": getattr(ref_layout, "image_path", None),
                        "generated_image_path": abs_save_name_path,
                        "sim_score": sim_score,
                        "geom_penalty": safe_float(geom_penalty, 0.0),
                    })
                except Exception as e:
                    print(f"      [WARN] layout scoring failed: {e}")
                    continue

            panel_entry["variations"].append({
                "variation_id": j,
                "image_path": os.path.abspath(image_path),
                "anime_image_path": os.path.abspath(anime_image_path) if anime_image_path else None,
                "layout_options": layout_options
            })

            write_json(os.path.join(panel_dir, "scores.json"), panel_entry)

        # ----------------------------
        # CLIP Scoring
        # ----------------------------
        try:
            ver_prompt = get_verification_prompt(panel)
        except Exception:
            descs = [e["content"] for e in panel if e.get("type") == "description"]
            ver_prompt = " ".join(descs) if descs else ""

        best_score = -1e9
        winner = {"variation": None, "layout_idx": None, "score": None}

        for var in panel_entry["variations"]:
            img_path = var["image_path"]
            if not os.path.exists(img_path):
                continue

            try:
                c_score = calculate_clip_score(img_path, ver_prompt, clip_model, clip_processor, device)
            except Exception:
                c_score = 0.0

            var["clip_score"] = safe_float(c_score, 0.0)

            for lo in var["layout_options"]:
                sim = safe_float(lo["sim_score"])
                geom = safe_float(lo["geom_penalty"])

                final_score = (sim * 100.0) + (var["clip_score"] * 50.0) - geom
                lo["final_score"] = final_score

                if final_score > best_score:
                    best_score = final_score
                    # ensure generated_image_path stored is absolute
                    gen_path = lo.get("generated_image_path", "")
                    gen_path_abs = os.path.abspath(gen_path) if gen_path else gen_path
                    winner = {
                        "variation": var["variation_id"],
                        "layout_idx": lo["rank"],
                        "score": final_score,
                        "generated_image_path": gen_path_abs,
                        "template_path": lo.get("template_path")
                    }

        panel_entry["winner"] = winner
        write_json(os.path.join(panel_dir, "scores.json"), panel_entry)

        final_summary["panels"].append({
            "panel_index": i,
            "panel_dir": panel_dir,
            "winner": winner,
            "variations_count": len(panel_entry["variations"])
        })

    # ---------------------------------------------------------
    # FINAL SUMMARY + PDF CREATION
    # ---------------------------------------------------------
    final_json_path = os.path.join(final_out_dir, "final_output.json")
    manga_pdf_path = os.path.join(final_out_dir, "manga_output.pdf")

    compose_manga_pdf(
        scoring_dir=final_out_dir,
        output_pdf=manga_pdf_path
    )

    write_json(final_json_path, final_summary)

    print("\nPipeline finished.")
    print("Saved results to:", final_out_dir)

    return final_summary


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    refined_elements_path = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251203_113127/20251203_1131/elements_refined.json"
    enhanced_prompts_path = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251203_113127/20251203_1131/enhanced_image_prompts.json"
    panels_path = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251203_113127/20251203_1131/panel.json"
    output_dir = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/scoring2"

    run_offline_pipeline(
        refined_elements_path=refined_elements_path,
        enhanced_prompts_path=enhanced_prompts_path,
        panels_path=panels_path,
        output_dir=output_dir,
        num_images=3,
        num_names=5
    )
