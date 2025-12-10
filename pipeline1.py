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

# -----------------------
# Helpers
# -----------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, default=str)

# -----------------------
# Pipeline
# -----------------------
def run_offline_pipeline(
    refined_elements_path,
    enhanced_prompts_path,
    panels_path,
    output_dir,
    num_images=3,
    num_names=5,
    controlnet_check=True,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load files
    print("Loading local files...")
    with open(refined_elements_path, "r", encoding="utf-8") as f:
        refined = json.load(f)
    with open(enhanced_prompts_path, "r", encoding="utf-8") as f:
        enhanced_prompts = json.load(f)
    with open(panels_path, "r", encoding="utf-8") as f:
        panels = json.load(f)

    # basic sanity checks
    if len(enhanced_prompts) != len(panels):
        print("[WARN] number of prompts does not match number of panels. Using min length.")
    panel_count = min(len(enhanced_prompts), len(panels))

    # 2. Check ControlNet endpoint (optional)
    if controlnet_check:
        ok = check_open()
        if not ok:
            print("[WARN] ControlNet/SD server not reachable. Pipeline may fail.")

    # 3. Load CLIP once for scoring
    print("Loading CLIP model for scoring...")
    clip_model, clip_processor, device = load_clip_model()

    final_summary = {
        "panels": [],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 4. Iterate panels
    for i in range(panel_count):
        prompt = enhanced_prompts[i]
        panel = panels[i]
        panel_dir = os.path.join(output_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)

        panel_entry = {
            "panel_index": i,
            "prompt": prompt,
            "variations": []
        }

        print(f"\n[Panel {i}] prompt length={len(prompt)}; elements={len(panel)}")
        for j in range(num_images):
            print(f"  - Generating image var {j} ...")
            image_path = os.path.join(panel_dir, f"{j:02d}.png")

            # Generate image using your local SD function
            try:
                generate_image_with_sd(prompt, image_path)
            except Exception as e:
                print(f"    [ERROR] SD generation failed for panel {i} var {j}: {e}")
                continue

            anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
            try:
                generate_animepose_image(image_path, prompt, anime_image_path)
            except Exception as e:
                print(f"    [WARN] generate_animepose_image failed: {e}; continuing with original image")
                anime_image_path = None

            try:
                openpose_result = run_controlnet_openpose(image_path, anime_image_path)
            except Exception as e:
                print(f"    [ERROR] ControlNet openpose failed: {e}; skipping this variation")
                continue

            try:
                bboxes = controlnet2bboxes(openpose_result)
            except Exception as e:
                print(f"    [WARN] controlnet2bboxes failed: {e}; using fallback bbox")
                cw, ch = getattr(openpose_result, "canvas_width", 512), getattr(openpose_result, "canvas_height", 512)
                bboxes = [[cw//4, ch//4, 3*cw//4, 3*ch//4]]

            layout = generate_layout(bboxes, panel, openpose_result.canvas_width, openpose_result.canvas_height)
            if layout is None:
                print("    [WARN] invalid layout; skipping variation")
                continue

            scored_layouts = similar_layouts(layout)
            layout_options = []
            for idx_ref_layout, scored_layout in enumerate(scored_layouts[:num_names]):
                try:
                    ref_layout = scored_layout[0]
                    sim_score = safe_float(scored_layout[1], 0.0)
                    geom_penalty = calculate_geometric_penalty(ref_layout, panel, openpose_result)

                    save_name_path = os.path.join(panel_dir, f"{j:02d}_name_{idx_ref_layout}.png")
                    try:
                        generate_name(openpose_result, layout, scored_layout, panel, save_name_path)
                    except Exception as e:
                        print(f"      [WARN] generate_name failed: {e}")

                    layout_options.append({
                        "rank": idx_ref_layout,
                        "template_path": getattr(ref_layout, "image_path", None),
                        "generated_image_path": save_name_path,
                        "sim_score": sim_score,
                        "geom_penalty": safe_float(geom_penalty, 0.0),
                    })
                except Exception as e:
                    print(f"      [WARN] error in scored_layout idx {idx_ref_layout}: {e}")
                    continue

            panel_entry["variations"].append({
                "variation_id": j,
                "image_path": image_path,
                "anime_image_path": anime_image_path,
                "layout_options": layout_options
            })

            write_json(os.path.join(panel_dir, "scores.json"), panel_entry)

        # Scoring with CLIP
        try:
            ver_prompt = get_verification_prompt(panel)
        except Exception:
            descs = [e["content"] for e in panel if e.get("type") == "description"]
            ver_prompt = " ".join(descs) if descs else ""

        best_score = -1e9
        winner = {"variation": None, "layout_idx": None, "score": None}

        for var in panel_entry.get("variations", []):
            img_path = var.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                c_score = calculate_clip_score(img_path, ver_prompt, clip_model, clip_processor, device)
            except Exception:
                c_score = 0.0
            var["clip_score"] = safe_float(c_score, 0.0)

            for layout_opt in var.get("layout_options", []):
                sim = safe_float(layout_opt.get("sim_score", 0.0))
                geom = safe_float(layout_opt.get("geom_penalty", 0.0))
                final_score = (sim * 100.0) + (var["clip_score"] * 50.0) - geom
                layout_opt["final_score"] = final_score

                if final_score > best_score:
                    best_score = final_score
                    winner = {
                        "variation": var.get("variation_id"),
                        "layout_idx": layout_opt.get("rank"),
                        "score": final_score,
                        "generated_image_path": layout_opt.get("generated_image_path"),
                        "template_path": layout_opt.get("template_path"),
                    }

        panel_entry["winner"] = winner
        write_json(os.path.join(panel_dir, "scores.json"), panel_entry)
        final_summary["panels"].append({
            "panel_index": i,
            "panel_dir": panel_dir,
            "winner": winner,
            "variations_count": len(panel_entry.get("variations", []))
        })

    final_out_path = os.path.join(output_dir, "final_output.json")
    write_json(final_out_path, final_summary)
    print("\nPipeline finished. Results saved to:", output_dir)
    return final_summary

# -----------------------
# RUN DIRECTLY
# -----------------------
if __name__ == "__main__":
    # <-- MANUALLY SET PATHS HERE -->
    refined_elements_path = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251126_151109/20251126_1511/elements_refined.json"
    enhanced_prompts_path = "//home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251126_151109/20251126_1511/enhanced_image_prompts.json"
    panels_path = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/ui_run_20251126_151109/20251126_1511/panel.json"
    output_dir = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/output/scoring"

    run_offline_pipeline(
        refined_elements_path=refined_elements_path,
        enhanced_prompts_path=enhanced_prompts_path,
        panels_path=panels_path,
        output_dir=output_dir,
        num_images=3,
        num_names=5
    )
