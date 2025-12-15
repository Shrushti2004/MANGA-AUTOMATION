import json
import random
import os

class CaoInitialLayout:
    """
    [Layout Generator]
    This class is responsible for generating the abstract structure (arrangement of panel) of the manga page.
    
    It uses a probabilistic model (learned from the Manga109 dataset) to decide how to split the page 
    recursively (Binary Space Partitioning). It decides "Panel A is above Panel B" or "Panel C is to the left of Panel D".
    """
    
    def __init__(self, style_model_path, page_width=1000, page_height=1414, direction='rtl'):
        self.w = page_width
        self.h = page_height
        self.direction = direction.lower() # Page reading order, 'rtl' is right to left
        
        # Load the learned probabilities from the style model file
        if not os.path.exists(style_model_path):
            raise FileNotFoundError(f"Style model not found at: {style_model_path}")
            
        with open(style_model_path, 'r') as f:
            self.models = json.load(f)

    def generate_layout(self, panels, return_tree=False):
        """
        Main Entry Point. Performs a Monte Carlo Search to find the best layout arrangement (Random sampling)
        How it works:
        - Generate 1000 different graph trees (Layouts)
        - Score each tree based on how well each panel correlate to its importance score
        - Return the best tree with lowest cost / penalty

        Args:
            panels (list) : List of panel containing importance score
            return_tree (bool) : If True, return the raw binary tree instead coordinates 
        
        Returns: List of panels with 'bbox' [x, y, w, h] or dict containing winning binary tree
        """
        if not panels: return []
        
        num_panels = len(panels)
        best_tree = None
        best_cost = float('inf')
        
        # Monte Carlo Search, Try to generate 1000 random layout, and keep the best one
        iterations = 1000
        
        for _ in range(iterations):
            # 1. Generate a random tree based on Structure Probability
            root_rect = {'x': 0, 'y': 0, 'w': self.w, 'h': self.h}
            tree = self._random_tree(panels, root_rect, depth=0)
            
            # 2. Score it based on Importance & Shape models
            cost = self._score_tree(tree, num_panels)
            
            # 3. Keep the winner
            if cost < best_cost:
                best_cost = cost
                best_tree = tree
        if return_tree:
            return best_tree
        
        # 4. Convert the winning tree into flat coordinates
        return self._flatten_tree(best_tree, panels)
    
    def generate_top_k(self, panels, k=3, return_trees=False): 
        """
        Generates 1000 layouts and returns the top K *unique* best unique layout.
        Returns: List of tuples [(score, result), ...]
        """
        if not panels: return []
        
        num_panels = len(panels)
        candidates = []
        
        # 1. Run Monte Carlo
        iterations = 1000
        for _ in range(iterations):
            root_rect = {'x': 0, 'y': 0, 'w': self.w, 'h': self.h}
            tree = self._random_tree(panels, root_rect, depth=0)
            cost = self._score_tree(tree, num_panels)
            
            tree_hash = self._hash_tree(tree)
            candidates.append((cost, tree, tree_hash))
            
        # 2. Sort by Cost (Lowest is best)
        candidates.sort(key=lambda x: x[0])
        
        # 3. Collect top K unique layouts
        results = []
        seen_hashes = set()
        
        for cost, tree, t_hash in candidates:
            if len(results) >= k: break
            
            if t_hash in seen_hashes:
                continue 
            
            seen_hashes.add(t_hash)
            
            # CHECK: Return Tree or Flat Panel?
            if return_trees:
                results.append((cost, tree))
            else:
                layout_flat = self._flatten_tree(tree, panels)
                results.append((cost, layout_flat))
            
        return results

    def _hash_tree(self, node):
        """Helper to create a unique signature for a tree structure."""
        if node["type"] == "leaf":
            return f"L({node['p_idx']})"
        else:
            # Hash format: "SplitType[LeftHash|RightHash]"
            # This captures the structure regardless of precise coordinates
            return f"{node['split']}[{self._hash_tree(node['left'])}|{self._hash_tree(node['right'])}]"

    def _random_tree(self, panels, rect, depth):
        """
        Recursively splits the canvas (Binary Space Partitioning).
        
        Decisions made at each step:
        1. Split Direction (Horizontal vs Vertical) based on learned style probabilities.
        2. Split Ratio (Where to cut) based on the 'importance_score' of panels in each group.
        """
        if len(panels) == 1:
            return {
                "type": "leaf", 
                "p_idx": panels[0]['panel_index'], 
                "rect": rect
            }

        # --- DECISION 1: Split Direction ---
        # Reference the style model file where to split at each boxes
        key = f"depth_{depth}"
        if key not in self.models["structure"]: key = "depth_0"
        prob_h = self.models["structure"][key]["H"]
        split_type = "H" if random.random() < prob_h else "V"
        
        # --- DECISION 2: Split Ratio ---
        # Randomly divide the list of panel into two groups to be put into each boxes
        split_idx = random.randint(1, len(panels)-1)
        grp_a = panels[:split_idx] # Earlier panels (e.g. Panel 0)
        grp_b = panels[split_idx:] # Later panels (e.g. Panel 1)
        
        # Calculate importance score weight of Group A vs Total
        w_a = sum(p.get('importance_score', 5) for p in grp_a)
        w_tot = sum(p.get('importance_score', 5) for p in panels)
        target_ratio = w_a / w_tot if w_tot > 0 else 0.5
        
        # Organic Wiggle / Adding a bit randomness
        ratio = max(0.2, min(0.8, target_ratio + random.uniform(-0.05, 0.05)))
        
        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
        
        if split_type == "H": # If the cut is horizontal
            # Horizontal: Top (A) -> Bottom (B). Unchanged for RTL.
            rect_a = {'x': x, 'y': y, 'w': w, 'h': h * ratio}
            rect_b = {'x': x, 'y': y + h * ratio, 'w': w, 'h': h * (1 - ratio)}
            
            return {
                "type": "node", "split": "H",
                "left": self._random_tree(grp_a, rect_a, depth + 1),  # Top
                "right": self._random_tree(grp_b, rect_b, depth + 1)  # Bottom
            }
        else:
            # Vertical: Left vs Right.
            w_a = w * ratio
            w_b = w * (1 - ratio)
            
            if self.direction == 'rtl':
                # In Manga, the earlier panel (Group A) is on the RIGHT.
                # The later panel (Group B) is on the LEFT.
                # The Tree structure usually treats "Left Child" as geometric left,
                # so we map Group B to the "Left Child".
                
                rect_right = {'x': x + w_b, 'y': y, 'w': w_a, 'h': h} # A goes here
                rect_left  = {'x': x,       'y': y, 'w': w_b, 'h': h} # B goes here
                
                return {
                    "type": "node", "split": "V",
                    "left": self._random_tree(grp_b, rect_left, depth + 1),   # Left Branch = Later Panels
                    "right": self._random_tree(grp_a, rect_right, depth + 1)  # Right Branch = Earlier Panels
                }
            else:
                # LTR: Group A goes Left.
                rect_left  = {'x': x,       'y': y, 'w': w_a, 'h': h}
                rect_right = {'x': x + w_a, 'y': y, 'w': w_b, 'h': h}
                
                return {
                    "type": "node", "split": "V",
                    "left": self._random_tree(grp_a, rect_left, depth + 1),   # Left Branch = Earlier Panels
                    "right": self._random_tree(grp_b, rect_right, depth + 1)  # Right Branch = Later Panels
                }
            

    def _score_tree(self, tree, num_panels):
        """
        Calculates the 'Badness' of a layout. Lower is better. 
        "Badness" is calculated by comparing generated area percentages against its importance score
        """
        leaves = self._get_leaves(tree)
        leaves.sort(key=lambda x: x['rect']['w'] * x['rect']['h'], reverse=True)
        
        cost = 0
        total_area = self.w * self.h
        
        # Fetch importance statistics for this number of panels
        imp_key = str(num_panels)
        if imp_key not in self.models["importance"]:
            available_keys = [int(k) for k in self.models["importance"].keys()]
            if available_keys:
                closest = min(available_keys, key=lambda k: abs(k - num_panels))
                imp_key = str(closest)
            else:
                return 0
        
        targets = self.models["importance"][imp_key]

        # 1. Importance Cost: Do bigger rank panels get bigger areas?
        for rank, leaf in enumerate(leaves):
            rank_key = str(rank)
            if rank_key in targets:
                target_pct = targets[rank_key]
                actual_pct = (leaf['rect']['w'] * leaf['rect']['h']) / total_area
                cost += (actual_pct - target_pct) ** 2 * 100
        
        # 2. Shape Scoring: Penalize extremely thin/tall panels
        for leaf in leaves:
            w, h = leaf['rect']['w'], leaf['rect']['h']
            ratio = w / h if h > 0 else 1.0
            
            if ratio < 0.5 or ratio > 2.0:
                cost += 50
            if ratio < 0.15 or ratio > 6.0:
                cost += 1000

        return cost

    def _get_leaves(self, node):
        if node["type"] == "leaf":
            return [node]
        return self._get_leaves(node["left"]) + self._get_leaves(node["right"])

    def _flatten_tree(self, node, original_panels):
        """
        Converts the tree structure back into the flat panel list.
        """
        leaves = self._get_leaves(node)
        result = []
        
        for leaf in leaves:
            # Find the original panel metadata
            orig = next(p for p in original_panels if p['panel_index'] == leaf['p_idx'])
            new_p = orig.copy()
            r = leaf['rect']
            
            # Direct copy of rectangle coordinates
            new_p['bbox'] = [int(r['x']), int(r['y']), int(r['w']), int(r['h'])]
            result.append(new_p)
            
        return result