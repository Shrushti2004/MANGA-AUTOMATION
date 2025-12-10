import streamlit as st
import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. Setup & Imports ---
try:
    # Import your Adapter Client
    from lib.llm.geminiadapter import GeminiClient
    
    # Import the original library functions
    from lib.script.divide import divide_script, ele2panels, refine_elements
    from lib.image.image import generate_image_prompts, enhance_prompts, generate_image_with_sd
    from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes, ControlNetResult
    from lib.name.name import generate_animepose_image, generate_name
    from lib.layout.layout import generate_layout, similar_layouts, Speaker, NonSpeaker
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.error("Make sure you run this script from the project root and have run 'pip install -e .'")
    st.stop()

# Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set in .env file")
    st.stop()

try:
    client = GeminiClient(api_key=api_key, model="gemini-2.0-flash")
except Exception as e:
    st.error(f"Failed to initialize GeminiClient: {e}")
    st.stop()

# Config
OUTPUT_DIR = "output_debug_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_dir = OUTPUT_DIR

st.set_page_config(page_title="Full Pipeline Debugger", layout="wide")
st.title("🚀 Full Pipeline Debugger (Gemini Adapter)")

# --- 1. Script Input ---
st.sidebar.header("Input Script")
default_script = """無関心な人々を見て、諦めたようにうなだれる蝶子。
蝶子「この世にあたしの味方なんていないんだ…。」

すれ違いざまに蝶子の言葉を聞いた椿。
気になって立ち止まる。


椿「あんた、味方が欲しいの？」

急に声をかけられ、驚いて振り返る蝶子。


（椿の容姿）
細身の黒いスリーピーススーツ、金髪の美青年。二十歳。色白。前髪長め。蝶子との身長差は頭一つ分くらい。

凛とした出で立ちでこちらを見つめる椿。


蝶子M「カッコイイ人…」
蝶子、椿の容姿に目を奪われる。


慎二「バカ！こっち来い！」

腕を思い切り引っ張って、慎二が蝶子を自分の方へ引き寄せる。


ホテル前で慎二に腕を掴まれ、痛そうに顔をしかめる蝶子。
椿、前髪をかき上げる。

椿「俺が味方になってあげよっか。」

蝶子M「（例え気紛れだったとしても、その言葉に、あの時のあたしがどれだけ救われたかなんて、王子様にはわからないだろうな。）」


蝶子焦ったように。
蝶子「お願いッ、たすけ…」

慎二、舌打ちしながら蝶子の言葉を遮る。
慎二「俺以外の男と話すな！無視しろよ。」
慎二に怒鳴られ、身をすくめる蝶子。


椿「ねぇ、そのDV男って、あんたの彼氏？」
パンツのポケットに両手を突っ込み、冷めた目で慎二を見ている椿。

蝶子、慎二に引っ張られながらも椿に向かって必死に手を伸ばす。
蝶子「ちがう！こいつストーカーなの。助けて！」


慎二「他の男と話すなって言ってんだろ！」

椿が止める間もなく、慎二が思い切り蝶子を平手打ち。


蝶子、路上に倒れ込み、頬を押さえる。

ボロボロ涙をこぼしながらも、慎二を睨みつける蝶子。

蝶子「最ッ低。」


慎二、自分がしたことにハッとして戸惑う。

慎二「た、叩いてごめんね。でも、蝶子ちゃんがいけないんだよ。俺の言うこと聞いてくれないから。」

蝶子、座り込んだ状態で震えたまま、引きつった笑顔で歩み寄る慎二を見上げる。

慎二「ホテルで続きしよう、ね？」

腰をかがめ、蝶子の顔を覗き込む慎二。怯えて青ざめる蝶子。


"""
script_text = st.sidebar.text_area("Enter Script", default_script, height=300)

if st.button("Run Full Pipeline", type="primary"):
    
    script_path = os.path.join(OUTPUT_DIR, "temp_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    st.header("1. Script Processing (LLM)")
    
    # --- Step 1: Script Parsing ---
    with st.status("Parsing Script...", expanded=True) as status:
        st.write("running `divide_script`...")
        elements = divide_script(client, script_path, base_dir)
        st.write("running `refine_elements`...")
        elements = refine_elements(elements, base_dir)
        
        speakers = list(set([e["speaker"] for e in elements if "speaker" in e]) - {""})
        st.success(f"Detected Speakers: {speakers}")

        st.write("running `ele2panels`...")
        panels = ele2panels(client, elements, base_dir)
        
        st.write("running `generate_image_prompts`...")
        prompts = generate_image_prompts(client, panels, speakers, base_dir)
        
        st.write("running `enhance_prompts`...")
        enhanced_prompts = enhance_prompts(client, prompts, base_dir)
        
        status.update(label="Script Processing Complete!", state="complete", expanded=False)

    st.header("2. Image Generation & Layout (All Panels)")
    
    progress_bar = st.progress(0)
    
    for panel_idx, panel_data in enumerate(panels):
        progress_bar.progress((panel_idx + 1) / len(panels))
        
        with st.expander(f"🖼️ Panel {panel_idx + 1} Processing", expanded=(panel_idx==0)):
            
            # 1. Get Prompt
            try:
                prompt_data = enhanced_prompts[panel_idx]
                final_prompt = prompt_data["prompt"] if isinstance(prompt_data, dict) else prompt_data
                st.info(f"**Prompt:** {final_prompt}")
            except IndexError:
                st.error(f"Error: No prompt found for Panel {panel_idx}")
                continue

            col1, col2 = st.columns(2)
            
            # --- Image Gen ---
            with col1:
                st.subheader("A. Generation")
                img_name = f"panel{panel_idx}_00.png"
                img_path = os.path.join(OUTPUT_DIR, img_name)
                anime_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_00_anime.png")
                
                generate_image_with_sd(final_prompt, img_path)
                st.image(img_path, caption="Main Image")
                
                generate_animepose_image(img_path, final_prompt, anime_path)
                st.image(anime_path, caption="Sketch (for OpenPose)")

            # --- ControlNet ---
            with col2:
                st.subheader("B. Analysis")
                openpose_result = run_controlnet_openpose(img_path, anime_path)
                
                if openpose_result.image:
                    st.image(openpose_result.image, caption="OpenPose Skeleton")
                
                # --- Detailed OpenPose Data ---
                with st.expander("💀 OpenPose Keypoint Data (Coordinates)"):
                    for i, person in enumerate(openpose_result.people):
                        st.markdown(f"**Person {i}**")
                        
                        # Handle typos in lib/image/controlnet.py gracefully
                        left_hand = getattr(person, 'hand_left_keypoints_2d', None) or getattr(person, 'hand_left_keyopints_2d', None)
                        right_hand = getattr(person, 'hand_right_keypoints_2d', None) or getattr(person, 'hand_right_keyoints_2d', None)
                        
                        keypoint_data = {
                            "Body (Pose)": str(person.pose_keypoints_2d[:3]) + "..." if person.pose_keypoints_2d else "None",
                            "Face": str(person.face_keypoints_2d[:3]) + "..." if person.face_keypoints_2d else "None",
                            "Left Hand": str(left_hand[:3]) + "..." if left_hand else "None",
                            "Right Hand": str(right_hand[:3]) + "..." if right_hand else "None"
                        }
                        st.json(keypoint_data)
                # -------------------------------

                bboxes = controlnet2bboxes(openpose_result)
                
                base_img = Image.open(img_path)
                draw = ImageDraw.Draw(base_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 40) 
                except:
                    font = None 

                for i, box in enumerate(bboxes):
                    draw.rectangle(box, outline="red", width=5)
                    draw.text((box[0], box[1]), f"Char {i}", fill="red", font=font)
                st.image(base_img, caption="Detected Bounding Boxes")

            # --- Layout ---
            st.subheader("C. Layout & Bubbles")
            width, height = openpose_result.canvas_width, openpose_result.canvas_height
            layout = generate_layout(bboxes, panel_data, width, height)
            
            if layout is None:
                st.error("Failed to generate layout. (Character/Dialogue mismatch)")
            else:
                scored_layouts = similar_layouts(layout)
                
                if not scored_layouts:
                    st.error("No matching layouts found in database.")
                else:
                    best_match = scored_layouts[0]
                    ref_layout = best_match[0]
                    score = best_match[1]
                    pairs_iterator = best_match[2]
                    pairs_list = list(pairs_iterator)
                    
                    st.success(f"Best Match Found! Score: {score:.4f}")
                    
                    reconstructed_scored_layout = (ref_layout, score, pairs_list)
                    save_name_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_final.png")
                    
                    generate_name(
                        openpose_result, 
                        layout, 
                        reconstructed_scored_layout, 
                        panel_data, 
                        save_name_path
                    )
                    
                    st.image(save_name_path, caption="✨ Final Result", use_column_width=True)

                    # --- Detailed Bubble & Mapping Data ---
                    st.divider()
                    with st.expander("🔍 Deep Dive: Coordinate Mapping & Bubble Locations"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("### 1. Generated Character BBoxes")
                            st.write("Format: `[min_x, min_y, max_x, max_y]`")
                            for idx, bbox in enumerate(bboxes):
                                st.code(f"Char {idx}: {bbox}")

                        with c2:
                            st.write("### 2. Template Bubble BBoxes")
                            st.write("Format: `[min_x, min_y, max_x, max_y]`")
                            
                            # Extract bubble coordinates from the template layout
                            for idx, element in enumerate(ref_layout.elements):
                                if isinstance(element, Speaker):
                                    speaker_label = f"Template Speaker {idx}"
                                    if element.text_info:
                                        # Show all bubbles for this speaker
                                        bubbles = [t["bbox"] for t in element.text_info]
                                        st.code(f"{speaker_label}: {bubbles}")
                                    else:
                                        st.code(f"{speaker_label}: No text info found")
                                elif isinstance(element, NonSpeaker):
                                    st.code(f"Template Element {idx}: Non-Speaker")

                        st.write("### 3. Final Assignment Map (The Intelligent Fix)")
                        st.caption("This shows exactly which of YOUR characters maps to which TEMPLATE bubble.")
                        
                        mapping_debug = []
                        for base_idx, ref_idx in pairs_list:
                            base_bbox = bboxes[base_idx]
                            ref_speaker = ref_layout.elements[ref_idx]
                            
                            # Re-simulate the logic to find WHICH bubble is used
                            target_bubble_bbox = "None"
                            if isinstance(ref_speaker, Speaker) and ref_speaker.text_info:
                                 bbox = [-1, -1, -1, -1]
                                 for text_obj in ref_speaker.text_info:
                                     if text_obj["bbox"][2] > bbox[2]:
                                         bbox = text_obj["bbox"]
                                 target_bubble_bbox = str(bbox)
                            
                            mapping_debug.append({
                                "Your Character (Index)": base_idx,
                                "Mapped to Template (Index)": ref_idx,
                                "Target Bubble BBox": target_bubble_bbox
                            })
                        st.table(mapping_debug)