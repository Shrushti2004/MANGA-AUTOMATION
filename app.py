import streamlit as st
import subprocess
import os
import json
import tempfile
import glob
from datetime import datetime
import re
from collections import defaultdict
import sys
import base64
import streamlit.components.v1 as components

# --- 1. SETUP & IMPORTS ---
try:
    from lib.layout.layout import MangaLayout
except ModuleNotFoundError:
    st.error("Error: 'lib' module not found. Please run 'pip install -e .' in your terminal.")
    st.stop()
except ImportError as e:
    st.error(f"Error importing: {e}. Did you run 'pip install -r requirements.txt'?")
    st.stop()

st.set_page_config(page_title="MANGAGEN PARSOLA", layout="wide")
st.title("Manga Generation")

# --- 2. SESSION STATE MANAGEMENT ---
if 'current_view_path' not in st.session_state:
    st.session_state['current_view_path'] = None
if 'generation_log' not in st.session_state:
    st.session_state['generation_log'] = ""

# Ensure output folder exists
OUTPUT_ROOT = "output"
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

# --- 3. GENERATION INPUT ---
st.subheader("1. GENERATION")
script_text = st.text_area("Script:", value="", height=150)

if st.button("🚀 Generate Panels", type="primary"):
    if not script_text.strip():
        st.warning("Please enter a script first.")
    else:
        # A. Prepare File
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(script_text)
            temp_script_path = f.name
        
        # B. Prepare Output Path
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_ROOT, f"ui_run_{date_str}")

        st.info("Running pipeline... (Logs will appear below)")
        
        # C. Run Pipeline
        command = [
            sys.executable, 
            "src/pipeline.py",
            "--script_path", temp_script_path,
            "--output_path", output_dir
        ]
        
        try:
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            
            live_log_placeholder = st.empty()
            full_log_buffer = ""

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=my_env,
                bufsize=1
            )

            # Capture logs in real-time
            for line in process.stdout:
                full_log_buffer += line
                live_log_placeholder.text_area("Executing...", value=full_log_buffer, height=300)

            process.wait()

            st.session_state['generation_log'] = full_log_buffer

            if process.returncode != 0:
                st.error("Pipeline failed. Check the log below.")
            else:
                st.success("Generation Complete!")
                st.session_state['current_view_path'] = output_dir
                st.rerun()  # Refresh to update Results View

        except Exception as e:
            st.error(f"Error running pipeline: {e}")
        
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# --- 4. PERSISTENT LOG DISPLAY ---
if st.session_state['generation_log']:
    with st.expander("📜 Last Generation Log", expanded=False):
        st.text_area("Log Output:", value=st.session_state['generation_log'], height=300)

st.markdown("---")

# --- 5. RESULT VIEWER ---
st.subheader("3. Results Viewer")

def get_run_folders():
    if not os.path.exists(OUTPUT_ROOT): return []
    dirs = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
    return sorted(dirs, reverse=True)

run_folders = get_run_folders()
run_options = ["Select a run"] + run_folders

current_index = 0
if st.session_state['current_view_path']:
    current_folder_name = os.path.basename(st.session_state['current_view_path'])
    if current_folder_name in run_options:
        current_index = run_options.index(current_folder_name)

col_sel, _ = st.columns([1, 2])
with col_sel:
    selected_run_name = st.selectbox(
        "Choose a run to inspect:",
        options=run_options,
        index=current_index
    )

view_path = os.path.join(OUTPUT_ROOT, selected_run_name) if selected_run_name != "Select a run" else None

if view_path and os.path.exists(view_path):
    with st.expander("⚙️ View Settings", expanded=True):
        col_sort, col_filter = st.columns(2)
        with col_sort:
            sort_mode = st.radio(
                "Sort Images By:",
                ["Variation ID (Default)", "Highest Total Score", "Lowest Penalty"],
                horizontal=True
            )
        with col_filter:
            show_best_only = st.checkbox(" Show Only Best Candidate per Panel")

    # --- Collect images with fallback ---
    images_root = os.path.join(view_path, "*", "images")
    panel_groups = defaultdict(list)
    panel_pattern = re.compile(r"(panel\d+)")

    # Scan all panel directories
    panel_dirs = glob.glob(os.path.join(images_root, "panel*"))
    for panel_path in panel_dirs:
        # Try _onlyname.png first
        images_onlyname = glob.glob(os.path.join(panel_path, "*_onlyname.png"))
        if images_onlyname:
            for img_path in images_onlyname:
                match = panel_pattern.search(img_path)
                if match:
                    panel_name = match.group(1)
                    panel_groups[panel_name].append(img_path)
        else:
            # Fallback to *_anime.png
            images_anime = glob.glob(os.path.join(panel_path, "*_anime.png"))
            for img_path in images_anime:
                match = panel_pattern.search(img_path)
                if match:
                    panel_name = match.group(1)
                    panel_groups[panel_name].append(img_path)

    if not panel_groups:
        st.warning(f"No images found in `{selected_run_name}`.")
    else:
        panel_names = sorted(panel_groups.keys())
        tabs = st.tabs(panel_names)
        
        for i, panel_name in enumerate(panel_names):
            with tabs[i]:
                image_list = panel_groups[panel_name]
                
                scores_map = {} 
                if image_list:
                    panel_img_dir = os.path.dirname(image_list[0])
                    score_file = os.path.join(panel_img_dir, "scores.json")
                    
                    if os.path.exists(score_file):
                        try:
                            with open(score_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                for var in data.get("variations", []):
                                    v_id = var.get("variation_id")
                                    clip = var.get("clip_score", 0.0)
                                    for layout in var.get("layout_options", []):
                                        rank = layout.get("rank")
                                        scores_map[(v_id, rank)] = {
                                            "final": layout.get("final_score", -999),
                                            "clip": clip,
                                            "sim": layout.get("sim_score", 0),
                                            "geom": layout.get("geom_penalty", 999)
                                        }
                        except Exception: pass

                display_items = []
                for img_path in image_list:
                    try:
                        fname = os.path.basename(img_path)
                        parts = fname.split("_")
                        var_id = int(parts[0])
                        rank_id = int(parts[2])
                        
                        s = scores_map.get((var_id, rank_id), {
                            "final": -999, "clip": 0, "sim": 0, "geom": 0
                        })
                        
                        display_items.append({
                            "path": img_path,
                            "name": fname,
                            "score": s
                        })
                    except:
                        continue

                if sort_mode == "Highest Total Score":
                    display_items.sort(key=lambda x: x["score"]["final"], reverse=True)
                elif sort_mode == "Lowest Penalty":
                    display_items.sort(key=lambda x: x["score"]["geom"])
                else:
                    display_items.sort(key=lambda x: x["name"])

                if show_best_only and display_items:
                    if sort_mode == "Lowest Penalty":
                         best = min(display_items, key=lambda x: x["score"]["geom"])
                         display_items = [best]
                    else:
                         best = max(display_items, key=lambda x: x["score"]["final"])
                         display_items = [best]

                if not display_items:
                    st.info("No images found.")
                else:
                    cols = st.columns(3)
                    for j, item in enumerate(display_items):
                        col = cols[j % 3]
                        s = item["score"]
                        
                        try:
                            col.image(item["path"], caption=item["name"], use_container_width=True)
                            
                            if s["final"] != -999:
                                score_color = "green" if float(s['final']) > 80 else "orange" if float(s['final']) > 50 else "red"
                                border_color = "#d4edda" if float(s['final']) > 80 else "#fff3cd" if float(s['final']) > 50 else "#f8d7da"
                                
                                col.markdown(f"""
                                <div style="text-align:center; background-color:{border_color}; padding:8px; border-radius:8px; margin-bottom:15px; border:1px solid {score_color};">
                                    <div style="font-size:20px; font-weight:bold; color:{score_color};">
                                        {float(s['final']):.1f}
                                    </div>
                                    <div style="font-size:11px; color:#444; margin-top:4px;">
                                        CLIP: <b>{float(s['clip']):.2f}</b> • Sim: <b>{float(s['sim']):.2f}</b>
                                    </div>
                                    <div style="font-size:11px; color:#d9534f; font-weight:bold;">
                                        Penalty: -{float(s['geom']):.1f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            col.error(f"Error: {e}")

# --- PDF PREVIEW & DOWNLOAD (Auto-detect in selected run folder and subfolders) ---
st.subheader("📄 Manga PDF Preview / Download")

if view_path and os.path.exists(view_path):
    pdf_candidates = glob.glob(os.path.join(view_path, "**", "manga.pdf"), recursive=True)
    
    if pdf_candidates:
        pdf_path = pdf_candidates[0]  # take first found
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Manga PDF",
                data=pdf_bytes,
                file_name="manga.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error displaying PDF: {e}")
    else:
        st.info("No manga.pdf found in the selected run folder or its subfolders.")
else:
    st.info("No run folder selected.")
