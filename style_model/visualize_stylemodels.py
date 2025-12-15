import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


# CONFIG
MODEL_PATH = "/home/kazukilablinux/Documents/MangaProject/bubbleAlloc/style_model/style_models_manga109.json"

def plot_style_models():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found. Run training first!")
        return

    with open(MODEL_PATH, "r") as f:
        data = json.load(f)

    # Setup Figures
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Learned Manga Style Models (Ying Cao Method)', fontsize=16)

    # --- PLOT 1: IMPORTANCE MODEL (Area vs Rank) ---
    ax1 = fig.add_subplot(131)
    ax1.set_title("Importance Model: Panel Size by Rank")
    ax1.set_xlabel("Importance Rank (0 = Hero Panel)")
    ax1.set_ylabel("Average Page Area (%)")
    
    # Plot lines for different page densities (4, 5, 6 panels)
    # Filter to show only common page counts
    common_counts = ['4', '5', '6', '7']
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, count in enumerate(common_counts):
        if count in data["importance"]:
            ranks = data["importance"][count]
            # Convert str keys "0", "1" to ints and sort
            x = sorted([int(k) for k in ranks.keys()])
            y = [ranks[str(k)] for k in x]
            ax1.plot(x, y, marker='o', label=f"{count} Panels", color=colors[i % len(colors)])

    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 2: STRUCTURE MODEL (Split Probabilities) ---
    ax2 = fig.add_subplot(132)
    ax2.set_title("Structure Model: Split Direction")
    ax2.set_ylabel("Probability")
    
    depths = sorted(data["structure"].keys())
    h_probs = [data["structure"][d]["H"] for d in depths]
    v_probs = [data["structure"][d]["V"] for d in depths]
    
    x_pos = np.arange(len(depths))
    ax2.bar(x_pos, h_probs, width=0.6, label='Horizontal (Rows)', color='#4c72b0')
    ax2.bar(x_pos, v_probs, width=0.6, bottom=h_probs, label='Vertical (Cols)', color='#55a868')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(depths)
    ax2.legend()

    # --- PLOT 3: SHAPE MODEL (The "Manga Skew") ---
    ax3 = fig.add_subplot(133)
    ax3.set_title("Shape Model: Learned Vertex Deviation")
    ax3.set_aspect('equal')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(1.1, -0.1) # Flip Y
    
    # 1. Draw Perfect Rectangle (Reference)
    rect_x = [0, 1, 1, 0, 0]
    rect_y = [0, 0, 1, 1, 0]
    ax3.plot(rect_x, rect_y, 'k--', label="Perfect Rect", alpha=0.5)
    
    # 2. Draw "Average Manga Panel"
    # Mean deltas are [dx1, dy1, dx2, dy2, ...]
    means = data["shape"]["mean"]
    stds = data["shape"]["std"]
    
    # Ideals: TL(0,0), TR(1,0), BR(1,1), BL(0,1)
    ideals = [[0,0], [1,0], [1,1], [0,1]]
    
    learned_poly = []
    
    for i in range(4):
        dx = means[i*2]
        dy = means[i*2+1]
        
        # Apply delta to ideal corner
        lx = ideals[i][0] + dx
        ly = ideals[i][1] + dy
        learned_poly.append([lx, ly])
        
        # Visualize Standard Deviation (The "Wiggle Room")
        # Draw an ellipse or error bars
        sx = stds[i*2] * 2 # 2 sigma
        sy = stds[i*2+1] * 2
        ellipse = patches.Ellipse((lx, ly), width=sx, height=sy, color='red', alpha=0.2)
        ax3.add_patch(ellipse)

    # Close the loop
    learned_poly.append(learned_poly[0]) 
    lx, ly = zip(*learned_poly)
    
    ax3.plot(lx, ly, 'r-', linewidth=2, label="Learned Mean")
    ax3.fill(lx, ly, 'red', alpha=0.05)
    ax3.legend()

    plt.tight_layout()
    plt.show()
    print("✅ Plot generated!")

if __name__ == "__main__":
    plot_style_models()