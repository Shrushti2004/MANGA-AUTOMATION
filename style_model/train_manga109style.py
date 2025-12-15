import json
import os
import glob
import sys
import numpy as np
import cv2  # pip install opencv-python
from pycocotools import mask as maskUtils # pip install pycocotools
from tqdm import tqdm

class Manga109Trainer:
    def __init__(self):
        # 1. Structure Model (H vs V splits)
        self.structure_counts = {}
        # 2. Importance Model (Area %)
        self.importance_data = {} 
        # 3. Shape Model (Vertex Deltas)
        self.vertex_deltas = [] 

    def train_from_folder(self, folder_path):
        """
        Scans a folder for JSON files (e.g. 'annotations/*.json')
        """
        search_path = os.path.join(folder_path, "*.json")
        json_files = glob.glob(search_path)

        if not json_files:
            print(f"❌ No JSON files found in {folder_path}")
            return

        print(f"📚 Found {len(json_files)} books. Starting training...")

        for fpath in tqdm(json_files, desc="Processing Books"):
            try:
                self._process_book(fpath)
            except Exception as e:
                print(f"⚠️ Skipping {os.path.basename(fpath)}: {e}")

        # Save Final Models
        output_path = os.path.join(os.path.dirname(__file__), "style_models_manga109.json")
        self._save_models(output_path)

    def _process_book(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Find 'frame' category ID
        cat_map = {c['id']: c['name'] for c in data['categories']}
        frame_id = next((k for k, v in cat_map.items() if v == 'frame'), None)
        if frame_id is None: return

        # 2. Group annotations by Image
        img_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_anns: img_anns[img_id] = []
            img_anns[img_id].append(ann)

        # 3. Process each Page
        for img_info in data['images']:
            img_id = img_info['id']
            w = img_info['width']
            h = img_info['height']
            
            # Get Panels
            anns = img_anns.get(img_id, [])
            panels = [a for a in anns if a['category_id'] == frame_id]
            if not panels: continue

            # --- DOUBLE PAGE LOGIC ---
            # If Width > Height, it's likely a double spread. Split it.
            if w > h:
                left_panels, right_panels = self._split_double_page(panels, w, h)
                # Treat as two separate training pages
                self._learn_from_page(left_panels)
                self._learn_from_page(right_panels)
            else:
                # Standard Single Page
                processed = self._extract_polygons(panels, w, h)
                self._learn_from_page(processed)

    def _split_double_page(self, raw_panels, page_w, page_h):
        """
        Splits panels into Left Page and Right Page based on the spine (center).
        Normalizes coordinates to 0..1 for each virtual page.
        """
        spine_x = page_w / 2
        left_page = []
        right_page = []

        for p in raw_panels:
            # Get BBox center
            bx, by, bw, bh = p['bbox']
            cx = bx + bw / 2
            
            # Check for spine crossing (Skip if it spans across the middle significantly)
            if bx < spine_x and (bx + bw) > spine_x:
                # Heuristic: If > 80% is on one side, keep it. Else discard.
                overlap_left = max(0, spine_x - bx)
                overlap_right = max(0, (bx + bw) - spine_x)
                if min(overlap_left, overlap_right) > bw * 0.2:
                    continue # Discard spine-crossers (bad training data)

            # Extract Polygon (0..1 normalized to FULL PAGE first)
            poly_norm = self._rle_to_norm_poly(p, page_w, page_h)
            if poly_norm is None: continue

            # Assign to Side
            if cx > spine_x:
                # RIGHT PAGE (Page A)
                # Remap X: (x - 0.5) * 2
                new_poly = np.copy(poly_norm)
                new_poly[:, 0] = (new_poly[:, 0] - 0.5) * 2
                # Validate bounds (0..1)
                if np.all((new_poly[:,0] >= 0) & (new_poly[:,0] <= 1)):
                    right_page.append({'poly': new_poly})
            else:
                # LEFT PAGE (Page B)
                # Remap X: x * 2
                new_poly = np.copy(poly_norm)
                new_poly[:, 0] = new_poly[:, 0] * 2
                if np.all((new_poly[:,0] >= 0) & (new_poly[:,0] <= 1)):
                    left_page.append({'poly': new_poly})

        return left_page, right_page

    def _extract_polygons(self, raw_panels, w, h):
        """Standard processing for single pages."""
        processed = []
        for p in raw_panels:
            poly = self._rle_to_norm_poly(p, w, h)
            if poly is not None:
                processed.append({'poly': poly})
        return processed

    def _rle_to_norm_poly(self, ann, w, h):
        """Decodes RLE -> Binary Mask -> Contours -> 4-Point Polygon."""
        if 'segmentation' not in ann: return None
        try:
            # 1. Decode RLE
            rle = ann['segmentation']
            mask = maskUtils.decode(rle) # Returns HxW binary array
            
            # 2. Find Contours (OpenCV)
            # Need uint8 type
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            
            # Take largest contour
            cnt = max(contours, key=cv2.contourArea)
            
            # 3. Approximate to 4 corners
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # If not 4 corners, force bounding box or skip?
            # We want diagonals, so we try to use the approx if it's 4 points.
            # If not, we fallback to rotated rectangle or bbox.
            final_pts = []
            
            if len(approx) == 4:
                final_pts = approx.reshape(4, 2)
            else:
                # Fallback: Minimum Area Rectangle (Rotated)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                final_pts = np.int32(box)

            # 4. Normalize
            final_pts = final_pts.astype(float)
            final_pts[:, 0] /= w
            final_pts[:, 1] /= h
            
            return final_pts

        except Exception:
            return None

    def _learn_from_page(self, panels):
        """Same logic as previous trainer, but takes polygon list."""
        if not panels: return
        
        # 1. Collect Shape & Area
        normalized_data = []
        for p in panels:
            poly = p['poly']
            
            # AABB Area
            min_x, min_y = np.min(poly, axis=0)
            max_x, max_y = np.max(poly, axis=0)
            pw, ph = max_x - min_x, max_y - min_y
            area = pw * ph
            
            normalized_data.append({
                "bbox": {"x": min_x, "y": min_y, "w": pw, "h": ph},
                "area": area,
                "poly": poly
            })

            # Learn Angles (Shape Model)
            self._record_vertex_deltas(poly, min_x, min_y, pw, ph)

        # 2. Learn Importance
        sorted_panels = sorted(normalized_data, key=lambda x: x['area'], reverse=True)
        num_p = len(sorted_panels)
        if num_p not in self.importance_data: self.importance_data[num_p] = {}
        
        for r, p in enumerate(sorted_panels):
            if r not in self.importance_data[num_p]: self.importance_data[num_p][r] = []
            self.importance_data[num_p][r].append(p['area'])

        # 3. Learn Structure (Tree)
        bboxes = [p['bbox'] for p in normalized_data]
        self._recover_tree(bboxes, depth=0)

    def _record_vertex_deltas(self, poly, min_x, min_y, w, h):
        """Calculates deviation from perfect rectangle corners."""
        max_x = min_x + w
        max_y = min_y + h

        # Sort corners TL, TR, BR, BL
        center = np.mean(poly, axis=0)
        angles = np.arctan2(poly[:,1] - center[1], poly[:,0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_poly = poly[sorted_indices]
        
        # Ideally: TL(-135), TR(-45), BR(45), BL(135)
        # Map to closest ideal corner
        ideals = np.array([
            [min_x, min_y], [max_x, min_y], # TL, TR
            [max_x, max_y], [min_x, max_y]  # BR, BL
        ])
        # Note: Need correct max_x/y from caller scope or re-calc
        max_x, max_y = min_x + w, min_y + h
        ideals = np.array([
            [min_x, min_y], [max_x, min_y], 
            [max_x, max_y], [min_x, max_y]
        ])

        deltas = []
        for i in range(4):
            # Find closest actual point
            dists = np.linalg.norm(sorted_poly - ideals[i], axis=1)
            closest = sorted_poly[np.argmin(dists)]
            
            dx = (closest[0] - ideals[i][0]) / (w if w>0 else 1)
            dy = (closest[1] - ideals[i][1]) / (h if h>0 else 1)
            deltas.extend([dx, dy])
            
        self.vertex_deltas.append(deltas)

    # ... (Reuse _recover_tree and _save_models from previous script exactly) ...
    # I will paste them here for completeness
    
    def _recover_tree(self, panels, depth):
        if len(panels) <= 1: return
        # Sort
        by_y = sorted(panels, key=lambda p: p['y'])
        by_x = sorted(panels, key=lambda p: p['x'])
        
        # Gap Detection
        split_h = -1
        for i in range(len(by_y)-1):
            if by_y[i+1]['y'] >= (by_y[i]['y'] + by_y[i]['h'] - 0.01):
                split_h = i; break
        
        split_v = -1
        for i in range(len(by_x)-1):
            if by_x[i+1]['x'] >= (by_x[i]['x'] + by_x[i]['w'] - 0.01):
                split_v = i; break
        
        split_type = "H" if split_h != -1 else ("V" if split_v != -1 else None)
        
        if split_type:
            key = f"depth_{depth}"
            if key not in self.structure_counts: self.structure_counts[key] = {"H":0, "V":0}
            self.structure_counts[key][split_type] += 1
            
            if split_type == "H":
                self._recover_tree(by_y[:split_h+1], depth+1)
                self._recover_tree(by_y[split_h+1:], depth+1)
            else:
                self._recover_tree(by_x[:split_v+1], depth+1)
                self._recover_tree(by_x[split_v+1:], depth+1)

    def _save_models(self, filename):
        # 1. Structure
        final_struct = {}
        for d, c in self.structure_counts.items():
            tot = c["H"] + c["V"]
            final_struct[d] = {"H": c["H"]/tot, "V": c["V"]/tot} if tot>0 else {"H":0.5, "V":0.5}

        # 2. Importance
        final_imp = {}
        for n, ranks in self.importance_data.items():
            final_imp[n] = {r: sum(a)/len(a) for r, a in ranks.items()}

        # 3. Shape
        deltas_arr = np.array(self.vertex_deltas)
        if len(deltas_arr) > 0:
            final_shape = {
                "mean": np.mean(deltas_arr, axis=0).tolist(),
                "std": np.std(deltas_arr, axis=0).tolist()
            }
        else:
            final_shape = {"mean": [0]*8, "std": [0.05]*8}

        with open(filename, "w") as f:
            json.dump({
                "structure": final_struct,
                "importance": final_imp,
                "shape": final_shape
            }, f, indent=4)
        print(f"✅ Success! Models saved to: {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        Manga109Trainer().train_from_folder(sys.argv[1])
    else:
        print("Usage: python train_manga109.py <path_to_json_folder>")