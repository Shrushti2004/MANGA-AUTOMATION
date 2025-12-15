import argparse
from tqdm import tqdm
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from PIL import Image


from lib.layout.layout import generate_layout, similar_layouts
from lib.script.divide import divide_script, ele2panels, refine_elements
from lib.image.image import generate_image, generate_image_prompts, enhance_prompts, generate_image_with_sd
from lib.image.controlnet import detect_human, check_open, controlnet2bboxes, run_controlnet_openpose
from lib.name.name import generate_name, generate_animepose_image
from lib.scoring.scorer import load_clip_model, get_verification_prompt, calculate_clip_score, calculate_geometric_penalty

from openai import OpenAI

# PDF composer
from lib.page.composer import compose_manga_pdf

# Storyboard analyzer (now uses GPT)
from lib.script.analyze import analyze_storyboard

from lib.page.layout_generator import CaoInitialLayout
from lib.page.layout_optimizer import LayoutOptimizer
from lib.page.page_compositor import PageCompositor
from lib.image.resolution import get_optimal_resolution

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
        raise ValueError("api_key is not set")

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

    # Page Config
    PAGE_WIDTH, PAGE_HEIGHT = 1039, 1476
    MARGIN, GUTTER = 80, 15
    LIVE_WIDTH = PAGE_WIDTH - (MARGIN * 2)
    LIVE_HEIGHT = PAGE_HEIGHT - (MARGIN * 2)

    style_model_path = "style_model/style_models_manga109.json"
    if not os.path.exists(style_model_path):
        print(f"[WARNING] Style model not found at {style_model_path}. Layout generation may fail.")

    # Initialize layout tools
    layout_gen = CaoInitialLayout(style_model_path, page_width=LIVE_WIDTH, page_height=LIVE_HEIGHT)
    layout_optimizer = LayoutOptimizer(style_model_path, page_width=LIVE_WIDTH, page_height=LIVE_HEIGHT, gutter=GUTTER)
    compositor = PageCompositor(PAGE_WIDTH, PAGE_HEIGHT, MARGIN, MARGIN)

    # USE GPT CLIENT
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

    print("Analyzing Storyboard...")
    storyboard_metadata = analyze_storyboard(client, panels, base_dir)

    # Group panels by page
    pages = {}
    for p in storyboard_metadata:
        pg_idx = p.get("page_index", 1)
        if pg_idx not in pages: pages[pg_idx] = []
        pages[pg_idx].append(p)

    print(f"Calculated {len(pages)} pages. Generating layout geometry...")

    page_layouts_map = {}
    panel_resolutions = {}

    for pg_idx, pg_panels in pages.items():
        flat_layout = layout_gen.generate_layout(pg_panels, return_tree=False)
    
        final_layout = []
        for p in flat_layout:
            x, y, w, h = p['bbox']
            p['polygon'] = [
                [x, y], 
                [x + w, y], 
                [x + w, y + h], 
                [x, y + h]
            ]
            final_layout.append(p)
    
        page_layouts_map[pg_idx] = final_layout

        for panel_geom in final_layout:
            xs = [pt[0] for pt in panel_geom['polygon']]
            ys = [pt[1] for pt in panel_geom['polygon']]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            res = get_optimal_resolution(int(w), int(h))
            panel_resolutions[panel_geom['panel_index']] = res

    # ---------------------------
    # IMAGE PROMPTS
    # ---------------------------
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
        target_w, target_h = panel_resolutions.get(i, (512, 512))

        for j in range(args.num_images):
            image_path = os.path.join(panel_dir, f"{j:02d}.png")
            generate_image_with_sd(prompt, image_path, width=target_w, height=target_h)

            anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
            generate_animepose_image(image_path, prompt, anime_image_path, width=target_w, height=target_h)

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

        with open(os.path.join(panel_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(panel_entry, f, indent=4, default=str)

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

    print("\n=== COMPOSITING FINAL MANGA PAGES ===")
    final_chapter_dir = os.path.join(base_dir, "final_page")
    os.makedirs(final_chapter_dir, exist_ok=True)

    for pg_idx, layout in page_layouts_map.items():
        # 1. Collect the "Winner" images for this page
        page_images = {}
        for p in layout:
            p_idx = p['panel_index']
            # Logic to find the best image for this panel
            # (You can read the scores.json you just saved to find the winner)
            panel_dir = os.path.join(image_base_dir, f"panel{p_idx:03d}")
            scores_path = os.path.join(panel_dir, "scores.json")
            
            if os.path.exists(scores_path):
                with open(scores_path, 'r') as f:
                    data = json.load(f)
                    best_img_path = data.get("winner", {}).get("generated_image_path")
                    if best_img_path:
                        page_images[p_idx] = best_img_path[:-4] + "_onlyname.png"
                    else:
                        best_img_path = os.path.join(panel_dir, "00_anime.png")
                        page_images[p_idx] = best_img_path
            else:
                best_img_path = os.path.join(panel_dir, "00_anime.png")
                page_images[p_idx] = best_img_path

        # 2. Create the Page
        output_file = os.path.join(final_chapter_dir, f"page_{pg_idx:02d}.png")
        compositor.create_page(layout, page_images, output_file)

    page_files = [
        f for f in os.listdir(final_chapter_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith("page_")
    ]
    
    # Sort them by number (page_01, page_02, etc.)
    # We use a lambda to extract the number from the filename to sort correctly
    try:
        page_files.sort(key=lambda x: int(x.replace("page_", "").split(".")[0]))
    except ValueError:
        page_files.sort() # Fallback to alphabetical if naming is weird

    if not page_files:
        print("⚠️ No pages found to combine.")
    else:
        pdf_images = []
        print(f"  > Found {len(page_files)} pages. converting...")
        
        for filename in page_files:
            file_path = os.path.join(final_chapter_dir, filename)
            try:
                # Open image and convert to RGB (PDF doesn't like Alpha channels/Transparency)
                img = Image.open(file_path).convert("RGB")
                pdf_images.append(img)
            except Exception as e:
                print(f"    ! Error loading {filename}: {e}")

        # 3. Save as PDF
        if pdf_images:
            pdf_filename = f"Manga_Chapter_{date}.pdf"
            pdf_path = os.path.join(base_dir, pdf_filename)
            
            pdf_images[0].save(
                pdf_path, 
                save_all=True, 
                append_images=pdf_images[1:]
            )
            print(f"✅ PDF Generated Successfully: {pdf_path}")
        else:
            print("⚠️ Failed to generate PDF image list.")

    print("\nPipeline Finished Successfully!")


if __name__ == "__main__":
    main()
