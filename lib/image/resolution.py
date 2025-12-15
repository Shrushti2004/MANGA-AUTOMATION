def get_optimal_resolution(target_w, target_h):
    """
    Finds the closest Safe SD 1.5 resolution for a given target width/height.
    
    Args:
        target_w (int): The width from the Layout Engine (e.g. 340).
        target_h (int): The height from the Layout Engine (e.g. 800).
        
    Returns:
        tuple: (width, height) - e.g. (512, 912)
    """
    if target_h == 0: return (512, 512)

    target_ratio = target_w / target_h

    # List of Safe SD 1.5 Resolutions
    # Format: (Width, Height, AspectRatio)
    SAFE_RESOLUTIONS = [
        (512, 512),  # 1:1   Square
        (768, 512),  # 3:2   Landscape
        (512, 768),  # 2:3   Portrait
        (896, 512),  # 16:9  Wide 
        (512, 896),  # 9:16  Tall 
        (1024, 512),  # 2:1   Ultra wide
        (384, 768),  # 1:2   Tall Strip 
        
    ]

    best_res = (512, 512)
    min_diff = float('inf')

    for w, h in SAFE_RESOLUTIONS:
        res_ratio = w / h
        diff = abs(target_ratio - res_ratio)
        
        # Tie-breaker: If ratios are very similar, prefer the one closer in pixel count
        # (Optional, but helps avoid upscaling tiny thumbnails to 912px)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
        elif diff == min_diff:
            # If tie, pick the one with closer area
            target_area = target_w * target_h
            current_area = best_res[0] * best_res[1]
            new_area = w * h
            if abs(new_area - target_area) < abs(current_area - target_area):
                best_res = (w, h)

    return best_res