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
                st.rerun()

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
run_options = ["Select a run..."] + run_folders

current_index = 0
if st.session_state['current_view_path']:
    current_folder_name = os.path.basename(st.session_state['current_view_path'])
    if current_folder_name in run_options:
        current_index = run_options.index(current_folder_name)

col_sel, col_dummy = st.columns([1, 2])
with col_sel:
    selected_run_name = st.selectbox(
        "Choose a run to inspect:",
        options=run_options,
        index=current_index
    )

if selected_run_name != "Select a run...":
    view_path = os.path.join(OUTPUT_ROOT, selected_run_name)
else:
    view_path = None

# CSS fixes
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; gap: 10px; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab"] { height: 30px; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-width: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
#     IMAGE LOADING + PER-PANEL FALLBACK
# -------------------------------
if view_path and os.path.exists(view_path):

    # --- VIEW SETTINGS ---
    with st.expander("⚙️ View Settings", expanded=True):
        col_sort, col_filter = st.columns(2)
        with col_sort:
            sort_mode = st.radio(
                "Sort Images By:",
                ["Variation ID (Default)", "Highest Total Score", "Lowest Penalty"],
                horizontal=True
            )
        with col_filter:
            show_best_only = st.checkbox("🏆 Show Only Best Candidate per Panel")

    # ---------- Build panel list from folder names ----------
    panel_dirs = glob.glob(os.path.join(view_path, "*", "images", "panel*"))
    panel_names = []
    for p in panel_dirs:
        folder = os.path.basename(p)
        if folder.startswith("panel"):
            panel_names.append(folder)
    panel_names = sorted(set(panel_names))

    if not panel_names:
        st.warning(f"No panel folders found in `{selected_run_name}`.")
    else:
        # For each panel, try per-panel fallback
        panel_groups = {}             # panel_name -> list of image paths
        panel_fallbacks = {}          # panel_name -> "onlyname" | "anime" | "none"

        for panel_name in panel_names:
            # Determine the folder for this panel
            panel_dir = os.path.join(view_path, "*", "images", panel_name)
            # glob to match possible nested structure (some runs may have different nesting)
            matched_dirs = glob.glob(panel_dir)
            images = []
            fallback = "none"
            if matched_dirs:
                # use first matched dir (expected to be one)
                pdir = matched_dirs[0]
                onlyname_pattern = os.path.join(pdir, "*_onlyname.png")
                anime_pattern = os.path.join(pdir, "*_anime.png")

                images = sorted(glob.glob(onlyname_pattern))
                if images:
                    fallback = "onlyname"
                else:
                    images = sorted(glob.glob(anime_pattern))
                    if images:
                        fallback = "anime"
                    else:
                        images = []
                        fallback = "none"
            else:
                # If there is no images/ panel folder found due to unexpected structure,
                # attempt looser search across any subfolders of the run
                loose_only = glob.glob(os.path.join(view_path, "**", panel_name, "*_onlyname.png"), recursive=True)
                if loose_only:
                    images = sorted(loose_only)
                    fallback = "onlyname"
                else:
                    loose_anime = glob.glob(os.path.join(view_path, "**", panel_name, "*_anime.png"), recursive=True)
                    if loose_anime:
                        images = sorted(loose_anime)
                        fallback = "anime"
                    else:
                        images = []
                        fallback = "none"

            panel_groups[panel_name] = images
            panel_fallbacks[panel_name] = fallback

        # Create tabs for every panel (even empty ones)
        tabs = st.tabs(panel_names)

        # Panel name regex for extracting ids from filenames
        panel_pattern = re.compile(r"(panel\d+)")

        for i, panel_name in enumerate(panel_names):
            with tabs[i]:
                image_list = panel_groups.get(panel_name, [])
                fallback_used = panel_fallbacks.get(panel_name, "none")

                # Determine the panel image directory for scores.json lookup
                # find a candidate directory that contains this panel
                candidate_dirs = glob.glob(os.path.join(view_path, "*", "images", panel_name))
                panel_img_dir = candidate_dirs[0] if candidate_dirs else None
                # If not found, attempt recursive search
                if not panel_img_dir:
                    rec = glob.glob(os.path.join(view_path, "**", panel_name), recursive=True)
                    panel_img_dir = rec[0] if rec else None

                # Load scores.json if present in panel image dir
                scores_map = {}
                if panel_img_dir:
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
                        except Exception:
                            # ignore score parsing errors
                            pass

                # If no images found for this panel, show notice but keep tab visible
                if not image_list:
                    if fallback_used == "none":
                        st.warning(f"No images found for **{panel_name}**.")
                    else:
                        # This branch should not occur since fallback_used would be 'anime' or 'onlyname' when images exist
                        st.warning(f"No images found for **{panel_name}** (fallback attempted: {fallback_used}).")
                    # continue to next tab (keeps the tab visible)
                    continue

                # Build display items with scores
                display_items = []
                for img_path in image_list:
                    try:
                        fname = os.path.basename(img_path)
                        parts = fname.split("_")
                        # attempt parsing var and rank; fallback to name-based sorting if parse fails
                        var_id = int(parts[0]) if parts and parts[0].isdigit() else None
                        rank_id = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                        
                        # if parse succeeded and scores available, pick them; else default
                        s = {"final": -999, "clip": 0, "sim": 0, "geom": 0}
                        if var_id is not None and rank_id is not None:
                            s = scores_map.get((var_id, rank_id), s)
                        
                        display_items.append({
                            "path": img_path,
                            "name": fname,
                            "score": s
                        })
                    except Exception:
                        # if anything unexpected in filename, still include image with default score
                        display_items.append({
                            "path": img_path,
                            "name": os.path.basename(img_path),
                            "score": {"final": -999, "clip": 0, "sim": 0, "geom": 0}
                        })

                # Sorting
                if sort_mode == "Highest Total Score":
                    display_items.sort(key=lambda x: x["score"]["final"], reverse=True)
                elif sort_mode == "Lowest Penalty":
                    display_items.sort(key=lambda x: x["score"]["geom"])
                else:
                    display_items.sort(key=lambda x: x["name"])

                # Best-only filter
                if show_best_only and display_items:
                    if sort_mode == "Lowest Penalty":
                        best = min(display_items, key=lambda x: x["score"]["geom"])
                    else:
                        best = max(display_items, key=lambda x: x["score"]["final"])
                    display_items = [best]

                # If we used anime fallback for this panel, show an inline warning
                if fallback_used == "anime":
                    st.warning(f"No `_onlyname.png` images for **{panel_name}** — showing `_anime.png` fallback for this panel.")

                # Display images in 3 columns
                if not display_items:
                    st.info("No images to show for this panel after filtering.")
                else:
                    cols = st.columns(3)
                    for j, item in enumerate(display_items):
                        col = cols[j % 3]
                        s = item["score"]
                        
                        try:
                            col.image(item["path"], caption=item["name"], use_container_width=True)
                            
                            if s["final"] != -999:
                                # choose color and border depending on score
                                try:
                                    final_val = float(s['final'])
                                except Exception:
                                    final_val = -999.0
                                score_color = "green" if final_val > 80 else "orange" if final_val > 50 else "red"
                                border_color = "#d4edda" if final_val > 80 else "#fff3cd" if final_val > 50 else "#f8d7da"
                                
                                col.markdown(f"""
                                <div style="text-align:center; background-color:{border_color}; padding:8px; border-radius:8px; margin-bottom:15px; border:1px solid {score_color};">
                                    <div style="font-size:20px; font-weight:bold; color:{score_color};">
                                        {final_val:.1f}
                                    </div>
                                    <div style="font-size:11px; color:#444; margin-top:4px;">
                                        CLIP: <b>{float(s.get('clip', 0)):.2f}</b> • Sim: <b>{float(s.get('sim', 0)):.2f}</b>
                                    </div>
                                    <div style="font-size:11px; color:#d9534f; font-weight:bold;">
                                        Penalty: -{float(s.get('geom', 0)):.1f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            col.error(f"Error: {e}")

# --- PDF PREVIEW & DOWNLOAD ---
st.subheader("📄 Manga PDF Preview / Download")

if view_path and os.path.exists(view_path):
    pdf_candidates = glob.glob(os.path.join(view_path, "**", "manga.pdf"), recursive=True)
    
    if pdf_candidates:
        pdf_path = pdf_candidates[0]
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
