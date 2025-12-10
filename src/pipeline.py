import argparse
from tqdm import tqdm
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from lib.layout.layout import generate_layout, similar_layouts
from lib.script.divide import divide_script, ele2panels, refine_elements
from lib.image.image import generate_image, generate_image_prompts, enhance_prompts, generate_image_with_sd
from lib.image.controlnet import detect_human, check_open, controlnet2bboxes, run_controlnet_openpose
from lib.name.name import generate_name, generate_animepose_image
from lib.scoring.scorer import load_clip_model, get_verification_prompt, calculate_clip_score, calculate_geometric_penalty

from openai import OpenAI

# ✅ Import composer for PDF generation
from lib.page.composer import compose_manga_pdf

# -------------------------------
# ARG PARSER
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline script for processing")
    parser.add_argument("--script_path", help="Path to the script file")
    parser.add_argument("--output_path", help="Path for output")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_names", type=int, default=5)
    return parser.parse_args()

# -------------------------------
# MAIN
# -------------------------------
def main():
    args = parse_args()
    NUM_REFERENCES = args.num_names
    resume_latest = args.resume_latest

    load_dotenv()
    api_key = os.getenv("api_key")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    if not check_open():
        raise ValueError("ControlNet is not running")

    # Set directories
    if resume_latest:
        date_dirs = [d for d in os.listdir(args.output_path) if os.path.isdir(os.path.join(args.output_path, d))]
        date_dirs.sort(key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
        date = date_dirs[-1]
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = os.path.join(args.output_path, date)
    image_base_dir = os.path.join(base_dir, "images")
    os.makedirs(image_base_dir, exist_ok=True)

    # OpenAI client
    client = OpenAI(api_key=api_key)

    # ---------------------------
    # SCRIPT PROCESSING
    # ---------------------------
    print("Dividing script...")
    elements = divide_script(client, args.script_path, base_dir)
    print("Refining elements...")
    elements = refine_elements(elements, base_dir)

    speakers = list(set([e["speaker"] for e in elements if e["speaker"] != ""]))
    print("Detected Speakers:", speakers)

    print("Converting elements to panels...")
    panels = ele2panels(client, elements, base_dir)

    print("Generating image prompts...")
    prompts = generate_image_prompts(client, panels, speakers, base_dir)

    print("Enhancing prompts...")
    prompts = enhance_prompts(client, prompts, base_dir)

    # ---------------------------
    # IMAGE GENERATION
    # ---------------------------
    for i, prompt in tqdm(enumerate(prompts), desc="Generating panel images"):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)

        panel_entry = {"panel_index": i, "panel_dir": panel_dir, "prompt": prompt, "variations": []}

        for j in range(args.num_images):
            image_path = os.path.join(panel_dir, f"{j:02d}.png")

            # ✔ FIXED: Correct SD call
            generate_image_with_sd(prompt, image_path)

            anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
            generate_animepose_image(image_path, prompt, anime_image_path)

            openpose_result = run_controlnet_openpose(image_path, anime_image_path)
            bboxes = controlnet2bboxes(openpose_result)

            layout = generate_layout(
                bboxes,
                panels[i],
                openpose_result.canvas_width,
                openpose_result.canvas_height
            )

            if layout is None:
                print(f"Invalid layout for panel {i} image {j}, skipping...")
                continue

            # Layout scoring
            scored_layouts = similar_layouts(layout)
            layout_options = []
            for idx_ref_layout, scored_layout in enumerate(scored_layouts[:NUM_REFERENCES]):
                ref_layout = scored_layout[0]
                sim_score = scored_layout[1]
                geom_penalty = calculate_geometric_penalty(ref_layout, panels[i], openpose_result)

                save_path = os.path.join(panel_dir, f"{j:02d}_name_{idx_ref_layout:1d}.png")
                generate_name(openpose_result, layout, scored_layout, panels[i], save_path)

                layout_options.append({
                    "rank": idx_ref_layout,
                    "template_path": ref_layout.image_path,
                    "generated_image_path": save_path,
                    "sim_score": sim_score,
                    "geom_penalty": geom_penalty
                })

            panel_entry["variations"].append({
                "variation_id": j,
                "image_path": image_path,
                "anime_image_path": anime_image_path,
                "layout_options": layout_options
            })

        # Save JSON per panel
        with open(os.path.join(panel_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(panel_entry, f, indent=4, default=str)

    # ---------------------------
    # SCORING & WINNER SELECTION
    # ---------------------------
    clip_model, clip_processor, device = load_clip_model()
    for i, prompt in enumerate(prompts):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        score_file = os.path.join(panel_dir, "scores.json")
        if not os.path.exists(score_file):
            continue

        with open(score_file, "r", encoding="utf-8") as f:
            panel_entry = json.load(f)

        ver_prompt = get_verification_prompt(panels[i])
        best_score = -9999
        winner_info = None

        print(f"Scoring Panel {i}...")
        for var in panel_entry.get("variations", []):
            c_score = calculate_clip_score(
                var.get("image_path"),
                ver_prompt,
                clip_model,
                clip_processor,
                device
            )
            var["clip_score"] = c_score

            for layout_opt in var.get("layout_options", []):
                try:
                    sim = float(layout_opt.get("sim_score", 0))
                except ValueError:
                    sim = 0.0
                try:
                    geom = float(layout_opt.get("geom_penalty", 0))
                except ValueError:
                    geom = 0.0

                final_score = (sim * 100) + (c_score * 50) - geom
                layout_opt["final_score"] = final_score

                if final_score > best_score:
                    best_score = final_score
                    winner_info = {
                        "variation_id": var.get("variation_id"),
                        "generated_image_path": layout_opt.get("generated_image_path")
                    }

        # Save winner info in scores.json
        panel_entry["winner"] = winner_info or {}

        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(panel_entry, f, indent=4, default=str)

        if winner_info:
            fname = os.path.basename(winner_info.get("generated_image_path", "??"))
            print(f"  🏆 Winner: Var {winner_info['variation_id']} / {fname} (Score: {best_score:.2f})")
        else:
            print(f"  ⚠️ No winner for panel {i}")

    # ---------------------------
    # PDF GENERATION
    # ---------------------------
    print("[PIPELINE] Generating final manga PDF…")
    try:
        scoring_dir = image_base_dir  # all panels are inside image_base_dir
        pdf_path = os.path.join(base_dir, "manga.pdf")
        compose_manga_pdf(scoring_dir, pdf_path)
        print(f"[PIPELINE] PDF saved to: {pdf_path}")
    except Exception as e:
        print("[ERROR] PDF generation failed:", e)


if __name__ == "__main__":
    main()
