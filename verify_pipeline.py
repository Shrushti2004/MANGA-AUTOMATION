import streamlit as st
import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

# --- SETUP: Ensure we can import from lib ---
sys.path.append(os.getcwd())

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def serialize_runtime_object(obj):
    """Helper to convert complex pipeline objects into standard dicts/lists for JSON display."""
    if "ControlNetResult" in str(type(obj)):
        return {
            "canvas_width": obj.canvas_width,
            "canvas_height": obj.canvas_height,
            "people_count": len(obj.people),
            "people": [serialize_runtime_object(p) for p in obj.people]
        }
    if "People" in str(type(obj)):
        return {
            "pose_keypoints_2d_count": len(obj.pose_keypoints_2d) if obj.pose_keypoints_2d else 0,
            "pose_keypoints_sample": obj.pose_keypoints_2d[:5] if obj.pose_keypoints_2d else [],
            "face_keypoints_count": len(obj.face_keypoints_2d) if obj.face_keypoints_2d else 0
        }
    if isinstance(obj, MangaLayout):
        return {
            "image_path": obj.image_path,
            "width": obj.width,
            "height": obj.height,
            "total_elements": len(obj.elements),
            "elements": [serialize_runtime_object(ele) for ele in obj.elements],
            "unrelated_text_length": obj.unrelated_text_length
        }
    if isinstance(obj, Speaker):
        return {
            "type": "Speaker",
            "bbox": obj.bbox,
            "text_length": obj.text_length,
            "text_info (Bubble Candidates)": obj.text_info
        }
    if isinstance(obj, NonSpeaker):
        return {
            "type": "NonSpeaker",
            "bbox": obj.bbox
        }
    return str(obj)

@st.cache_resource
def load_clip_model_cached():
    """
    Loads the CLIP model and processor.
    Cached by Streamlit so it only runs once per session.
    """
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model on {device}...")
    try:
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor, device
    except Exception as e:
        st.error(f"Failed to load CLIP: {e}")
        return None, None, None

def get_verification_prompt(panel_data):
    """
    Constructs the text prompt for CLIP.
    Combines scene descriptions with 'Rough Sketch' style tags.
    """
    # 1. Extract descriptions from the panel data
    descriptions = [ele['content'] for ele in panel_data if ele['type'] == 'description']
    scene_text = " ".join(descriptions)
    
    # Fallback if no description found (use dialogue snippet)
    if not scene_text:
        dialogues = [ele['content'] for ele in panel_data if ele['type'] == 'dialogue']
        scene_text = " ".join(dialogues)[:50] # First 50 chars

    if len(scene_text) > 200:
        scene_text = scene_text[:200] + "..."
    
    # 2. Append the "Manga Name/Sketch" style definition
    # This is crucial: We tell CLIP to EXPECT a messy sketch.
    style_suffix = ", (messy:1.5), (scribble:1.4), (bad art:1.3), lineart, clean lines, sketches, simple,  (monochrome:2), (white background:1.5)"
    
    return scene_text + style_suffix

def calculate_clip_score(image_path, text_prompt, model, processor, device):
    """
    Calculates the Cosine Similarity (0.0 - 1.0) between an image and text.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True, truncation=True,    # <--- FIX: Cut off excess text
            max_length=77).to(device)
        
        # Run Model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get Embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize Embeddings (Important for Cosine Similarity)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate Score (Dot Product)
        score = torch.matmul(text_embeds, image_embeds.t()).item()
        return score
        
    except Exception as e:
        st.error(f"Error calculating score: {e}")
        return 0.0

def get_face_bbox(person, width, height):
    """Calculates the bounding box of the face from normalized keypoints."""
    if not person.face_keypoints_2d:
        return None
    
    # Keypoints are normalized (0.0-1.0), so multiply by canvas size
    xs = [kp[0] for kp in person.face_keypoints_2d]
    ys = [kp[1] for kp in person.face_keypoints_2d]
    
    x1 = min(xs) * width
    y1 = min(ys) * height
    x2 = max(xs) * width
    y2 = max(ys) * height
    
    return [x1, y1, x2, y2]

def calculate_intersection_area(box1, box2):
    """Calculates the area of intersection between two [x1, y1, x2, y2] boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def calculate_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


try:
    from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes
    from lib.layout.layout import generate_layout, similar_layouts, MangaLayout, Speaker, NonSpeaker
    FONTPATH = "fonts/NotoSansCJK-Regular.ttc"
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure this script is in the project root.")
    st.stop()

st.set_page_config(page_title="Manga Pipeline Debugger", layout="wide")
st.title("🔧 Manga Pipeline Debugger")

# ==========================================
# 0. CONFIGURATION & DATA LOADING
# ==========================================
st.sidebar.header("Configuration")
default_output = "output" 
output_root = st.sidebar.text_input("Output Root Folder", value=default_output)

if os.path.exists(output_root):
    timestamps = sorted(os.listdir(output_root), reverse=True)
    selected_timestamp = st.sidebar.selectbox("Select Run (Timestamp)", timestamps)
else:
    st.sidebar.error("Output folder not found.")
    st.stop()

run_dir = os.path.join(output_root, selected_timestamp)
images_dir = os.path.join(run_dir, "images")

if os.path.exists(images_dir):
    panels = sorted(os.listdir(images_dir))
    selected_panel = st.sidebar.selectbox("Select Panel", panels)
    panel_dir = os.path.join(images_dir, selected_panel)
else:
    st.error("No 'images' folder found in the selected run.")
    st.stop()

panel_json_path = os.path.join(run_dir, "panel.json")
if os.path.exists(panel_json_path):
    with open(panel_json_path, "r", encoding="utf-8") as f:
        all_panels_data = json.load(f)
    try:
        panel_index = int(selected_panel.replace("panel", ""))
        current_panel_data = all_panels_data[panel_index]
    except:
        st.warning("Could not infer panel index from folder name. Using first panel data.")
        current_panel_data = all_panels_data[0]
else:
    st.error("panel.json not found.")
    st.stop()

st.write(f"**Debugging:** `{selected_panel}` in `{selected_timestamp}`")

# ==========================================
# SECTION 1: OPENPOSE CHECK
# ==========================================
st.header("1. OpenPose Skeleton Check")
st.info("Comparing detection results between the raw generation and the anime/lineart version.")

col1, col2 = st.columns(2)
raw_img_path = os.path.join(panel_dir, "00.png")
anime_img_path = os.path.join(panel_dir, "00_anime.png")

if not os.path.exists(raw_img_path) or not os.path.exists(anime_img_path):
    st.error("Images (00.png or 00_anime.png) not found in panel folder.")
    st.stop()

def get_pose_text(controlnet_result):
    text_data = []
    for i, person in enumerate(controlnet_result.people):
        text_data.append(f"Person {i+1}:")
        if person.pose_keypoints_2d:
            text_data.append(f"  Keypoints: {len(person.pose_keypoints_2d)} detected")
            text_data.append(f"  Sample (Nose): {person.pose_keypoints_2d[0] if len(person.pose_keypoints_2d) > 0 else 'N/A'}")
    return "\n".join(text_data)

with col1:
    st.subheader("Raw Image (00.png)")
    st.image(raw_img_path, width='stretch')
    if st.button("Run OpenPose on Raw"):
        with st.spinner("Running ControlNet..."):
            res_raw = run_controlnet_openpose(raw_img_path)
            st.image(res_raw.image, caption="OpenPose Result (Raw)", width='stretch')
            st.text_area("Raw Pose Data", get_pose_text(res_raw), height=150)
            st.session_state['res_raw'] = res_raw

with col2:
    st.subheader("Lineart Image (00_anime.png)")
    st.image(anime_img_path, width='stretch')
    if st.button("Run OpenPose on Lineart"):
        with st.spinner("Running ControlNet..."):
            res_anime = run_controlnet_openpose(anime_img_path)
            st.image(res_anime.image, caption="OpenPose Result (Lineart)", width='stretch')
            st.text_area("Lineart Pose Data", get_pose_text(res_anime), height=150)
            st.session_state['res_anime'] = res_anime

# ==========================================
# SECTION 2: LAYOUT & BBOX CALCULATION
# ==========================================
st.header("2. Layout & Bubble Logic")
speaker_count = sum(1 for ele in current_panel_data if ele["type"] == "dialogue")
st.caption(f"**Requirement:** Script contains **{speaker_count}** speaking turns (dialogues).")

final_res = None
used_source = "None"

res_anime = st.session_state.get('res_anime')
if res_anime is None and os.path.exists(anime_img_path):
    with st.spinner("Automating Step 1: Running OpenPose on Lineart..."):
        res_anime = run_controlnet_openpose(anime_img_path)
        st.session_state['res_anime'] = res_anime

lineart_valid = False
if res_anime:
    bboxes_anime = controlnet2bboxes(res_anime)
    if len(bboxes_anime) >= speaker_count:
        final_res = res_anime
        used_source = "Lineart (00_anime.png)"
        lineart_valid = True
    else:
        st.warning(f"⚠️ Lineart detection found **{len(bboxes_anime)}** people, but script needs **{speaker_count}**. Attempting fallback...")

if not lineart_valid:
    res_raw = st.session_state.get('res_raw')
    if res_raw is None and os.path.exists(raw_img_path):
        with st.spinner("Fallback: Running OpenPose on Raw Image..."):
            res_raw = run_controlnet_openpose(raw_img_path)
            st.session_state['res_raw'] = res_raw
    
    if res_raw:
        bboxes_raw = controlnet2bboxes(res_raw)
        if len(bboxes_raw) >= speaker_count:
            final_res = res_raw
            used_source = "Raw Image (00.png)"
            st.success(f"✅ Fallback Successful! Using Raw Image (Found {len(bboxes_raw)} people).")
        elif res_anime and len(bboxes_raw) > len(bboxes_anime):
            final_res = res_raw
            used_source = "Raw Image (00.png)"
            st.info(f"Fallback used Raw Image (Found {len(bboxes_raw)}) - better than Lineart, but still short of {speaker_count}.")
        elif res_anime:
            final_res = res_anime
            used_source = "Lineart (00_anime.png)"
            st.error(f"Fallback Failed. Raw image ({len(bboxes_raw)}) didn't improve on Lineart ({len(bboxes_anime)}). Reverting to Lineart.")
        else:
             final_res = res_raw
             used_source = "Raw Image (00.png)"

if final_res is None:
    st.error("❌ Could not generate OpenPose data from either image.")
else:
    st.write(f"**Active Source:** `{used_source}`")
    width, height = final_res.canvas_width, final_res.canvas_height
    
    st.subheader("2a. Generated Layout Object")
    bboxes = controlnet2bboxes(final_res)
    layout = generate_layout(bboxes, current_panel_data, width, height)

    if layout is None:
        st.error(f"Layout generation failed. Detected {len(bboxes)} people, but needed {speaker_count} speakers.")
    else:
        layout_viz = Image.new("RGB", (width, height), "white")
        draw_layout = ImageDraw.Draw(layout_viz)
        for i, element in enumerate(layout.elements):
            box = element.bbox
            if isinstance(element, Speaker):
                draw_layout.rectangle(box, outline="red", width=5)
                draw_layout.text((box[0]+5, box[1]+5), f"Speaker {i}", fill="red")
            elif isinstance(element, NonSpeaker):
                draw_layout.rectangle(box, outline="blue", width=5)
                draw_layout.text((box[0]+5, box[1]+5), f"NonSpeaker {i}", fill="blue")
        st.image(layout_viz, caption="Red = Speaker, Blue = NonSpeaker", width='stretch')

        st.subheader("2b. Template Matching & Bubble Placement")
        db_path = "./curated_dataset/database.json"
        if not os.path.exists(db_path):
            st.error(f"Database not found at {db_path}")
        else:
            scored_layouts = similar_layouts(layout, annfile=db_path)
            dialogues = [ele["content"] for ele in current_panel_data if ele["type"] == "dialogue"]
            monologues = [ele["content"] for ele in current_panel_data if ele["type"] == "monologue"]
            total_monologue = "".join(monologues)
            
            num_to_show = st.slider("Number of layouts to inspect", 1, 5, 1)
            tabs = st.tabs([f"Rank {i+1}" for i in range(num_to_show)])
            base_viz_path = raw_img_path if "Raw" in used_source else anime_img_path
            
            for idx, tab in enumerate(tabs):
                if idx >= len(scored_layouts): break
                ref_layout = scored_layouts[idx][0]
                score = scored_layouts[idx][1]
                
                with tab:
                    st.info(f"Matched with Template: `{os.path.basename(ref_layout.image_path)}` (Score: {score:.3f})")
                    if os.path.exists(base_viz_path):
                        viz_image = Image.open(base_viz_path).convert("RGB")
                    else:
                        viz_image = Image.new("RGB", (width, height), "gray")
                    draw = ImageDraw.Draw(viz_image)
                    
                    try:
                        font = ImageFont.truetype(FONTPATH, 20)
                        char_bbox = font.getbbox("あ")
                        char_width = char_bbox[2] - char_bbox[0]
                        char_height = char_bbox[3] - char_bbox[1]
                        vertical_margin = 2
                    except Exception as e:
                        st.error(f"Font load error: {e}")
                        font = None

                    bubble_log = []
                    ref_speakers = [ele for ele in ref_layout.elements if isinstance(ele, Speaker)]
                    def custom_sort(element):
                         if not isinstance(element, Speaker): return -1
                         if element.text_info is None: return 0
                         center_pos = []
                         for text_obj in element.text_info:
                             center_pos.append((text_obj["bbox"][0] + text_obj["bbox"][2]) / 2)
                         return max(center_pos) if center_pos else 0
                    ref_speakers = sorted(ref_speakers, key=custom_sort, reverse=True)
                    
                    panel_pointer = 0
                    for ref_element in ref_speakers:
                        if panel_pointer >= len(dialogues): break
                        ref_bbox = [-1, -1, -1, -1]
                        if ref_element.text_info:
                            for text_obj in ref_element.text_info:
                                if text_obj["bbox"][2] > ref_bbox[2]:
                                    ref_bbox = text_obj["bbox"]
                        draw.rectangle(ref_bbox, outline="blue", width=2)
                        final_bbox = None
                        if font:
                            text_content = dialogues[panel_pointer]
                            box_height = ref_bbox[3] - ref_bbox[1]
                            chars_per_col = int(box_height / (char_height + vertical_margin))
                            if chars_per_col < 1: chars_per_col = 1
                            num_cols = int(len(text_content) / chars_per_col) + 1
                            estimated_width = char_width * num_cols
                            final_x2 = ref_bbox[2]
                            final_x1 = ref_bbox[2] - estimated_width
                            final_y1 = ref_bbox[1]
                            final_y2 = ref_bbox[3]
                            final_bbox = [final_x1, final_y1, final_x2, final_y2]
                            draw.rectangle(final_bbox, outline="red", width=3)
                            bubble_log.append({"Type": "Dialogue", "Content": text_content, "Template BBox": ref_bbox, "Final BBox": final_bbox})
                        panel_pointer += 1

                    if total_monologue and hasattr(ref_layout, 'unrelated_text_bbox') and ref_layout.unrelated_text_bbox:
                        unrelated_bboxes = sorted([item["bbox"] for item in ref_layout.unrelated_text_bbox], key=lambda x: x[2], reverse=True)
                        if len(unrelated_bboxes) > 0:
                            target_bbox = unrelated_bboxes[0]
                            draw.rectangle(target_bbox, outline="cyan", width=2)
                            if font:
                                box_height = target_bbox[3] - target_bbox[1]
                                chars_per_col = int(box_height / (char_height + vertical_margin))
                                if chars_per_col < 1: chars_per_col = 1
                                num_cols = int(len(total_monologue) / chars_per_col) + 1
                                estimated_width = char_width * num_cols
                                final_x2 = target_bbox[2]
                                final_x1 = target_bbox[2] - estimated_width
                                final_y1 = target_bbox[1]
                                final_y2 = target_bbox[3]
                                final_bbox_mono = [final_x1, final_y1, final_x2, final_y2]
                                draw.rectangle(final_bbox_mono, outline="magenta", width=3)
                                bubble_log.append({"Type": "Monologue", "Content": total_monologue, "Template BBox": target_bbox, "Final BBox": final_bbox_mono})
                    
                    col_img, col_data = st.columns([2, 1])
                    with col_img:
                        st.image(viz_image, caption="Blue/Cyan: Template | Red/Magenta: Final", width='stretch')
                    with col_data:
                        st.markdown("### Bubble Data")
                        if bubble_log:
                            st.dataframe(pd.DataFrame(bubble_log))
                        else:
                            st.warning("No text placed")

                    # --- DATA CAPTURE FOR SECTION 3 ---
                    # We automatically capture the Rank 1 (idx 0) data to use in the Scoring section
                    if idx == 0:
                        st.session_state['rank1_bubbles'] = bubble_log
                        st.session_state['rank1_source_path'] = base_viz_path
                        st.session_state['rank1_people'] = final_res.people

        st.markdown("---")
        
    st.subheader("🔍 Deep Variable Inspection")
    var_tab1, var_tab2, var_tab3, var_tab4 = st.tabs(["1. OpenPose Data", "2. Detected BBoxes", "3. Matched Layout", "4. Speech Bubbles"])
    with var_tab1:
        if final_res: st.json(serialize_runtime_object(final_res))
    with var_tab2:
        if 'bboxes' in locals(): st.json(bboxes)
    with var_tab3:
        if 'scored_layouts' in locals() and len(scored_layouts) > 0: st.json(serialize_runtime_object(scored_layouts[0][0]))
    with var_tab4:
        if 'bubble_log' in locals() and len(bubble_log) > 0: st.json(bubble_log)

# ==========================================
# SECTION 3: OVERLAP SCORING SYSTEM
# ==========================================
st.header("3. Overlap Scoring System")
st.info("Evaluating the quality of Rank 1 layout based on overlap with characters and faces.")

if 'rank1_bubbles' not in st.session_state:
    st.warning("⚠️ Please run Section 2b first to generate bubble data for the best layout.")
else:
    rank1_bubbles = st.session_state['rank1_bubbles']
    rank1_people = st.session_state['rank1_people']
    source_path = st.session_state.get('rank1_source_path', "")

    if not rank1_bubbles:
        st.warning("No bubbles found in Rank 1 layout.")
    else:
        score_log = []
        
        # Prepare Scoring Visual
        if os.path.exists(source_path):
            score_img = Image.open(source_path).convert("RGBA")
        else:
            score_img = Image.new("RGBA", (width, height), (200, 200, 200, 255))
        
        # Create a transparent layer for drawing overlaps
        overlay = Image.new("RGBA", score_img.size, (255, 255, 255, 0))
        draw_score = ImageDraw.Draw(overlay)
        
        total_penalty = 0

        for b_idx, bubble in enumerate(rank1_bubbles):
            b_bbox = bubble["Final BBox"]
            b_area = calculate_box_area(b_bbox)
            
            # Draw Bubble (White outline)
            draw_score.rectangle(b_bbox, outline=(255, 255, 255, 255), width=2)
            
            bubble_penalty = 0
            overlap_details = []

            for p_idx, person in enumerate(rank1_people):
                # 1. Body Overlap
                # Body is derived from all keypoints, but we can use the already calculated bboxes from Section 2
                # However, 'rank1_people' is the raw object. Let's re-calculate body bbox roughly or use the one passed?
                # To be precise, let's use the standard 'controlnet2bboxes' logic just for this person
                # Note: Section 2 calculated all bboxes. Let's just recalculate simply here for the specific person object
                # OR better: assume bboxes match people index 1-to-1 (They usually do in the list order)
                
                # Let's do a quick body bbox calculation from pose keypoints
                if person.pose_keypoints_2d:
                    pxs = [kp[0] for kp in person.pose_keypoints_2d if kp[0] > 0]
                    pys = [kp[1] for kp in person.pose_keypoints_2d if kp[1] > 0]
                    p_bbox = [min(pxs)*width, min(pys)*height, max(pxs)*width, max(pys)*height]
                    
                    # Calc Intersection
                    intersect_body = calculate_intersection_area(b_bbox, p_bbox)
                    
                    if intersect_body > 0:
                        # Penalty: 1 point per 1% overlap (relative to bubble size)
                        pct_body = (intersect_body / b_area) * 100
                        points_body = pct_body * 1.0 # Weight 1
                        bubble_penalty += points_body
                        overlap_details.append(f"P{p_idx} Body: {pct_body:.1f}%")
                        
                        # Draw Body Box (Green)
                        draw_score.rectangle(p_bbox, outline=(0, 255, 0, 128), width=2)

                # 2. Face Overlap (Extra Penalty)
                f_bbox = get_face_bbox(person, width, height)
                if f_bbox:
                    intersect_face = calculate_intersection_area(b_bbox, f_bbox)
                    if intersect_face > 0:
                        # Penalty: 5 points per 1% overlap (High penalty!)
                        pct_face = (intersect_face / b_area) * 100
                        points_face = pct_face * 5.0 # Weight 5
                        bubble_penalty += points_face
                        overlap_details.append(f"P{p_idx} **FACE**: {pct_face:.1f}%")
                        
                        # Draw Face Box (Red Filled)
                        draw_score.rectangle(f_bbox, fill=(255, 0, 0, 100), outline="red")

            score_log.append({
                "Bubble Content": bubble["Content"][:20] + "...",
                "Penalty Score": round(bubble_penalty, 1),
                "Details": ", ".join(overlap_details) if overlap_details else "Clean"
            })
            total_penalty += bubble_penalty

        # Composite Image
        score_img = Image.alpha_composite(score_img, overlay)

        col_score_vis, col_score_data = st.columns([2, 1])
        with col_score_vis:
            st.image(score_img, caption="Green: Body | Red Fill: Face Intersection", width='stretch')
        with col_score_data:
            st.metric("Total Penalty Score", f"{total_penalty:.1f}", delta=None, help="Lower is better. Points added for overlaps.")
            st.dataframe(pd.DataFrame(score_log))


# ==========================================
# SECTION 4: PIPELINE VARIABLE INSPECTOR
# ==========================================
st.header("4. Pipeline Variable Inspector")
st.info("Inspect the intermediate JSON data generated at each step of the pipeline.")

available_files = {
    "Step 1: Raw Split (elements)": [f for f in os.listdir(run_dir) if f.endswith("_divided.json")],
    "Step 2: Refined Text (elements)": ["elements_refined.json"],
    "Step 3: Panels Grouping (panels)": ["panel.json"],
    "Step 4: Initial Prompts (prompts)": ["image_prompts.json"],
    "Step 5: Enhanced Tags (prompts)": ["enhanced_image_prompts.json"],
    "Step 6: Richfy Panel (optional)": ["panel_richfy.json"]
}

tabs = st.tabs(list(available_files.keys()))

for i, (step_name, file_candidates) in enumerate(available_files.items()):
    with tabs[i]:
        found_file = None
        for candidate in file_candidates:
            full_path = os.path.join(run_dir, candidate)
            if os.path.exists(full_path):
                found_file = full_path
                break
        
        if found_file:
            st.write(f"📂 **Source File:** `{os.path.basename(found_file)}`")
            try:
                with open(found_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                    df = pd.DataFrame(data, columns=["Content"])
                    df.index.name = "Panel ID"
                    st.dataframe(df, width='stretch')
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    st.dataframe(data, width='stretch')
                    with st.expander("View Raw JSON"):
                        st.json(data)
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    panel_id = st.number_input(f"Select Panel Index (0 to {len(data)-1})", 
                                             min_value=0, max_value=len(data)-1, 
                                             key=f"p_select_{i}")
                    st.subheader(f"Content of Panel {panel_id}")
                    st.dataframe(data[panel_id],width='stretch')
                    with st.expander("View Raw JSON for this Panel"):
                        st.json(data[panel_id])
                else:
                    st.json(data)
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
        else:
            st.warning(f"File not found for this step. (Expected: {file_candidates[0]})")

st.header("5. Semantic Similarity (CLIP)")
st.info("Evaluate if the generated sketch matches the script description.")

# State management for loading the heavy model
if 'clip_loaded' not in st.session_state:
    st.session_state['clip_loaded'] = False

if not st.session_state['clip_loaded']:
    if st.button("Load CLIP Model (Heavy Operation)"):
        with st.spinner("Loading OpenAI CLIP Model..."):
            model, processor, device = load_clip_model_cached()
            if model:
                st.session_state['clip_loaded'] = True
                st.success(f"✅ Model loaded successfully on `{device}`!")
                st.rerun()
else:
    # Model is loaded, show the scoring interface
    if st.button("Unload CLIP Model"):
        st.session_state['clip_loaded'] = False
        st.rerun()
        
    model, processor, device = load_clip_model_cached()
    
    st.divider()
    
    # 1. Construct the Verification Prompt
    ver_prompt = get_verification_prompt(current_panel_data)
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.text_area("Verification Prompt (Auto-Generated)", value=ver_prompt, height=80,
                     help="This text is what CLIP compares the image against.")
    with col_p2:
        st.info("💡 We append 'rough sketch' tags so CLIP doesn't penalize the messy style.")

    # 2. Score the Images
    # We check for 00.png, 01.png, 02.png (variations)
    st.subheader("Image Scores")
    
    cols = st.columns(3)
    
    # Loop through potential variations (assuming standard naming 00.png, 01.png, etc.)
    # If you only have 00.png, it will just show that.
    for i in range(3):
        img_filename = f"{i:02d}_anime.png"
        img_path = os.path.join(panel_dir, img_filename)
        
        with cols[i]:
            if os.path.exists(img_path):
                st.image(img_path, width='stretch')
                
                # Calculate Score
                score = calculate_clip_score(img_path, ver_prompt, model, processor, device)
                
                # Display Metric
                # Color coding: Green > 0.25, Yellow > 0.20, Red < 0.20
                if score > 0.25:
                    st.success(f"**Score: {score:.4f}** (Good)")
                elif score > 0.20:
                    st.warning(f"**Score: {score:.4f}** (Okay)")
                else:
                    st.error(f"**Score: {score:.4f}** (Bad/Hallucination)")
            else:
                st.caption(f"Variation {i} not found")