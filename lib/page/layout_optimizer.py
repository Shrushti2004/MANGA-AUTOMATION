import numpy as np
from scipy.optimize import minimize
import json
import math

class LayoutOptimizer:
    """
    [Geometry Optimizer]
    This class is responsible for turning the abstract binary tree from layout generator into 
    polygon coordinates.
    
    It uses numerical optimization (Scipy L-BFGS-B) to minimize an 'Energy Function' 
    that balances:
    1. Panel Area (matching the Importance Score)
    2. Panel Shape (keeping panels reasonably rectangular, avoiding slivers or weird shapes)
    """
    def __init__(self, style_model_path, page_width=1000, page_height=1414, gutter=20):
        self.w = page_width
        self.h = page_height
        self.gutter = gutter # Gap between panel
        

        # Load style stats (e.g. average angles) - currently used to initialize or bound optimization
        with open(style_model_path, 'r') as f:
            data = json.load(f)
            self.shape_model = data["shape"]
            self.mean_deltas = np.array(self.shape_model["mean"])
            self.std_deltas = np.array(self.shape_model["std"]) + 1e-5

    def shrink_panel(polygon, gutter_px):
        """
        Shrinks a polygon by 'gutter_px' pixels from all sides (approximated via scaling).
        """
        # 1. Calculate Centroid
        poly_arr = np.array(polygon)
        xs = poly_arr[:, 0]
        ys = poly_arr[:, 1]
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        
        # 2. Calculate Width/Height to determine Scale Factor
        w = np.max(xs) - np.min(xs)
        h = np.max(ys) - np.min(ys)
        
        if w <= gutter_px * 2 or h <= gutter_px * 2:
            return polygon # Too small to shrink safely
        
        # Scale factors (approximate uniform shrink)
        scale_x = (w - gutter_px * 2) / w
        scale_y = (h - gutter_px * 2) / h
        
        # Use the smaller scale to be safe (or uniform scale)
        scale = min(scale_x, scale_y)
        
        new_poly = []
        for x, y in polygon:
            # Move point towards centroid
            nx = centroid_x + (x - centroid_x) * scale
            ny = centroid_y + (y - centroid_y) * scale
            new_poly.append([nx, ny])
            
        return new_poly

    def optimize(self, layout_tree, panels_metadata):
        """
        Input: The Binary Tree from Stage 1.

        Process:
        1. Identify every "split" node in the tree.
        2. Define parameters to optimize for each split: [Split Ratio, Split Angle].
        3. Run scipy.minimize to find the best Ratio/Angle combination that fits the 
           panels' importance scores.

        Output: List of panels with 'polygon' (Diagonal shapes).
        """
        # 1. Identify all "Cut Nodes" in the tree
        # We need to optimize 2 variables for every cut: [Ratio, Angle]
        nodes = self._collect_nodes(layout_tree)
        num_cuts = len(nodes)
        
        if num_cuts == 0:
            return self._tree_to_panels(layout_tree, np.array([]), panels_metadata)

        # 2. Setup Initial State (x0)
        # [ratio_0, angle_0, ratio_1, angle_1, ...]
        x0 = []
        bounds = []
        
        for node in nodes:
            # Initial Ratio is start at 0.5 (center split) or derive from rough rect
            # We assume the tree stores the 'split' type ("H" or "V")
            # We initialize Angle to 0.0 (Perfectly straight)
            
            current_ratio = 0.5 # Default (Refining this requires parsing the rects)
            
            # Bounds: 
            # - Ratio: Keep split between 20% and 80% of the parent rectangle area.
            # - Angle: Allow slight diagonal tilts (-0.15 to 0.15 radians, approx +/- 8 degrees).
            #          (Manga usually has straight or slightly slanted cuts, not wild 45 deg cuts).
            x0.extend([current_ratio, 0.0])
            bounds.extend([(0.2, 0.8), (-0.15, 0.15)]) 

        x0 = np.array(x0)

        # 3. Optimize the energy function using Scipy
        res = minimize(
            fun=self._energy_function,
            x0=x0,
            args=(layout_tree, nodes, panels_metadata),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}
        )
        
        # 4. Generate Final Shapes
        raw_panels = self._tree_to_panels(layout_tree, res.x, panels_metadata)

        # Apply the gutters
        return self._apply_gutters(raw_panels)
    
    def _apply_gutters(self, panels):
        """Internal helper to shrink all panels by the gutter amount."""
        if self.gutter <= 0:
            return panels
            
        final_panels = []
        for p in panels:
            new_p = p.copy()
            new_p['polygon'] = self._shrink_polygon(p['polygon'], self.gutter)
            final_panels.append(new_p)
        return final_panels

    def _shrink_polygon(self, polygon, px):
        """Shrinks a polygon towards its centroid to create gutters"""
        poly_arr = np.array(polygon)
        xs = poly_arr[:, 0]
        ys = poly_arr[:, 1]
        
        # Centroid
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # BBox size
        w = np.max(xs) - np.min(xs)
        h = np.max(ys) - np.min(ys)
        
        if w <= px * 2 or h <= px * 2:
            return polygon # Too small to shrink
            
        # Calculate Scale Factor
        # We want to reduce width by (2 * px)
        scale_x = (w - px * 2) / w
        scale_y = (h - px * 2) / h
        scale = min(scale_x, scale_y)
        
        new_poly = []
        for x, y in polygon:
            nx = cx + (x - cx) * scale
            ny = cy + (y - cy) * scale
            new_poly.append([nx, ny])
            
        return new_poly

    def _energy_function(self, x, tree, nodes, meta):
        """
        The Cost Function ("Badness"). The optimizer tries to make this 0.
        
        It reconstructs the page with the current trial angles/ratios and checks:
        1. Does the Area match the Importance Score? (Primary Factor)
        2. Is the Shape weird? (Secondary Factor)
        """
        # 1. Reconstruct the page geometry with these angles
        final_panels = self._tree_to_panels(tree, x, meta)
        
        total_cost = 0
        
        # 2. Calculate Costs
        for p in final_panels:
            # --- Check Shape Validity ---
            poly = np.array(p['polygon'])
            if len(poly) < 3: 
                total_cost += 10000; continue
            
            # --- Area Cost (Importance) ---
            # Calculate Polygon Area using Shoelace Area
            xs, ys = poly[:, 0], poly[:, 1]
            area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
            
            # Get Target Area based on Importance Score
            imp_score = next((m['importance_score'] for m in meta if m['panel_index'] == p['panel_index']), 5)
            # Heuristic target: Importance * constant (Rough estimate of ideal pixel area)
            target_area = imp_score * (self.w * self.h / 25) 
            
            # Penalize deviation from target area
            total_cost += ((area - target_area) / target_area)**2 * 10

        return total_cost

    def _collect_nodes(self, node):
        """Recursively collects all 'node' type objects (splits) from the tree."""
        if node["type"] == "leaf": return []
        return [node] + self._collect_nodes(node["left"]) + self._collect_nodes(node["right"])

    def _tree_to_panels(self, tree, params, meta):
        """
        Reconstructs the polygons by slicing the page recursively.
        """
        # Start with full page polygon
        page_poly = [[0,0], [self.w,0], [self.w,self.h], [0,self.h]]
        
        # We need to map the flat list of optimization parameters back to specific nodes
        nodes_flat = self._collect_nodes(tree)
        
        # Map node object to (ratio, angle)
        param_map = {}
        for i, node in enumerate(nodes_flat):
            if i*2 < len(params):
                param_map[id(node)] = (params[i*2], params[i*2+1])
            else:
                param_map[id(node)] = (0.5, 0.0)

        results = []
        self._slice_recursive(tree, page_poly, param_map, results)
        return results

    def _slice_recursive(self, node, poly, param_map, results):
        if node["type"] == "leaf":
            results.append({
                "panel_index": node["p_idx"],
                "polygon": poly
            })
            return

        # Get cut parameters
        ratio, angle = param_map.get(id(node), (0.5, 0.0))
        
        # Calculate Cut Line
        # 1. Find Bounding Box of current poly to determine split point
        pts = np.array(poly)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        w, h = max_x - min_x, max_y - min_y
        
        center_x = min_x + w/2
        center_y = min_y + h/2
        
        # 2. Determine Pivot Point and Normal Vector for the cut line
        if node["split"] == "H": # Horizontal Cut (y = constant)
            base_y = min_y + h * ratio
            # Normal vector rotated by 'angle'
            # Default normal is (0, 1). Rotated: (sin a, cos a)
            nx = math.sin(angle)
            ny = math.cos(angle)
            # Pivot around (center_x, base_y)
            # Point on line P0 = (center_x, base_y)
            # Normal N = (sin a, cos a)
            # Eq: Nx * (x - P0x) + Ny * (y - P0y) = 0
            P0 = (center_x, base_y)
            
        else: # Vertical Cut (x = constant)
            base_x = min_x + w * ratio
            # Vertical normal is (1, 0). Rotated: (cos a, sin a)
            nx = math.cos(angle)
            ny = math.sin(angle)
            P0 = (base_x, center_y)

        # 3. Clip Polygon against Line
        # Returns two new polygons: Left/Top (A) and Right/Bottom (B)
        poly_a, poly_b = self._clip_polygon(poly, (nx, ny), P0)
        
        self._slice_recursive(node["left"], poly_a, param_map, results)
        self._slice_recursive(node["right"], poly_b, param_map, results)

    def _clip_polygon(self, poly, normal, origin):
        """
        Clips a polygon by an arbitrary line.
        Line is defined by Normal(nx, ny) and Point(ox, oy).
        
        Logic:
        Iterates through all edges of the polygon. If an edge crosses the cut line,
        it calculates the intersection point and creates two new vertices.
        """
        nx, ny = normal
        ox, oy = origin
        
        subject = poly
        new_poly_positive = []
        new_poly_negative = []
        
        for i in range(len(subject)):
            curr_p = subject[i]
            prev_p = subject[i-1]
            
            # Check which side of the line the points are on (using Dot Product)
            curr_dist = nx*(curr_p[0]-ox) + ny*(curr_p[1]-oy)
            prev_dist = nx*(prev_p[0]-ox) + ny*(prev_p[1]-oy)
            
            if curr_dist >= 0:
                if prev_dist < 0:
                    # Line crossed from negative to positive -> Add Intersection
                    t = prev_dist / (prev_dist - curr_dist)
                    ix = prev_p[0] + t * (curr_p[0] - prev_p[0])
                    iy = prev_p[1] + t * (curr_p[1] - prev_p[1])
                    new_poly_positive.append([ix, iy])
                    new_poly_negative.append([ix, iy])
                new_poly_positive.append(curr_p)
            else:
                if prev_dist >= 0:
                    # Line crossed from positive to negative -> Add Intersection
                    t = prev_dist / (prev_dist - curr_dist)
                    ix = prev_p[0] + t * (curr_p[0] - prev_p[0])
                    iy = prev_p[1] + t * (curr_p[1] - prev_p[1])
                    new_poly_positive.append([ix, iy])
                    new_poly_negative.append([ix, iy])
                new_poly_negative.append(curr_p)
                
        return new_poly_negative, new_poly_positive # A and B