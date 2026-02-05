"""
FreeFuse Mask Generation Utilities

Ported from diffusers FreeFuse pipeline implementation.
"""

import torch
import torch.nn.functional as F


def linear_normalize(x, dim=1):
    """
    Normalize tensor to [0, 1] range along specified dimension.
    
    This is crucial for balanced_argmax to work properly - softmax normalized
    values have very small variance, while linear normalization preserves
    the relative differences.
    """
    x_min = x.min(dim=dim, keepdim=True)[0]
    x_max = x.max(dim=dim, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-8)


def stabilized_balanced_argmax(
    logits, 
    h, 
    w, 
    target_count=None, 
    max_iter=15,
    lr=0.01,
    gravity_weight=0.0003,   # Match diffusers (was 0.00004)
    spatial_weight=0.0003,   # Match diffusers (was 0.00004)
    momentum=0.2,
    centroid_margin=0.0,
    border_penalty=0.0,
    anisotropy=1.3,          # Match diffusers (was 1.1)
    debug=False
):
    """
    Stabilized balanced argmax algorithm from diffusers FreeFuse implementation.
    
    Uses iterative bias adjustment with spatial constraints to produce
    balanced, spatially coherent mask assignments.
    
    Args:
        logits: (B, C, N) tensor where N = h * w
        h, w: Spatial dimensions
        target_count: Target pixel count per class (default: N/C)
        max_iter: Number of iterations
        lr: Learning rate for bias updates
        gravity_weight: Weight for centroid attraction
        spatial_weight: Weight for neighbor voting
        momentum: Momentum for probability smoothing
        
    Returns:
        (B, N) tensor of class assignments
    """
    B, C, N = logits.shape
    device = logits.device
    
    # Physical space coordinates
    scale_h = 1
    scale_w = 1
    
    y_range = torch.linspace(-scale_h, scale_h, steps=h, device=device)
    x_range = torch.linspace(-scale_w, scale_w, steps=w, device=device)
    x_range = x_range * anisotropy
    
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    flat_y = grid_y.reshape(1, 1, N)
    flat_x = grid_x.reshape(1, 1, N)
    
    # Border mask
    max_dim = max(h, w)
    pixel_size = 2.0 / max_dim
    is_border = (flat_y.abs() > (scale_h - pixel_size * 1.5)) | \
                (flat_x.abs() > (scale_w - pixel_size * 1.5))
    border_mask = is_border.float()
    
    if target_count is None:
        target_count = N / C
    
    bias = torch.zeros(B, C, 1, device=device, dtype=logits.dtype)
    
    # Initialize running_probs using linear normalization
    # CRITICAL: dim=-1 (N) normalizes each concept's spatial distribution independently
    # Using dim=1 (C) would distort relative strength between concepts!
    running_probs = linear_normalize(logits, dim=-1)
    
    # Compute logit scale for adaptive lr
    logit_range = (logits.max() - logits.min()).item()
    logit_scale = max(logit_range, 1e-4)
    effective_lr = lr * logit_scale
    max_bias = logit_scale * 10.0
    
    if debug:
        print(f"\n[BalancedArgmax] Start. B={B}, C={C}, N={N}, Target={target_count:.1f}")
        print(f"Logits range: min={logits.min().item():.5f}, max={logits.max().item():.5f}")
        print(f"Logit scale: {logit_scale:.6f}, Effective LR: {effective_lr:.6f}")
    
    # Spatial convolution kernel for neighbor voting
    neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
    neighbor_kernel[:, :, 1, 1] = 0
    
    current_logits = logits.clone()
    
    for i in range(max_iter):
        # A. Linear normalization (instead of softmax)
        probs = linear_normalize(current_logits - bias, dim=1)
        
        # B. Momentum smoothing
        running_probs = (1 - momentum) * probs + momentum * running_probs
        
        # C. Compute soft centroids
        mass = running_probs.sum(dim=2, keepdim=True) + 1e-6
        center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
        center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass
        
        # Centroid clamping
        if centroid_margin > 0:
            limit_y = scale_h * (1.0 - centroid_margin)
            limit_x = scale_w * (1.0 - centroid_margin)
            center_y = torch.clamp(center_y, -limit_y, limit_y)
            center_x = torch.clamp(center_x, -limit_x, limit_x)
        
        # D. Distance field
        dist_sq = (flat_y - center_y)**2 + (flat_x - center_x)**2
        
        # E. Bias update based on hard assignment counts
        hard_indices = torch.argmax(current_logits - bias, dim=1)
        hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)  # [B, C]
        
        diff = hard_counts - target_count
        cur_lr = effective_lr * (0.95 ** i)
        bias = bias + torch.sign(diff).unsqueeze(2) * cur_lr
        bias = torch.clamp(bias, -max_bias, max_bias)
        
        # F. Spatial voting
        if spatial_weight > 0:
            probs_img = running_probs.view(B, C, h, w)
            probs_img_f32 = probs_img.float()
            kernel_f32 = neighbor_kernel.float()
            neighbor_votes = F.conv2d(probs_img_f32, kernel_f32, padding=1, groups=C)
            neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
        else:
            neighbor_votes = torch.zeros_like(logits)
        
        gravity_term = dist_sq * gravity_weight
        border_term = border_mask * border_penalty
        
        current_logits = logits - bias + \
                        (neighbor_votes * spatial_weight) - \
                        gravity_term - \
                        border_term
        
        if debug and (i == 0 or i == max_iter - 1 or i % 5 == 0):
            hard_counts_list = hard_counts[0].int().tolist()
            print(f"Iter {i:02d}: Counts={hard_counts_list}")
    
    if debug:
        final_assignment = torch.argmax(current_logits, dim=1)[0]
        final_counts = final_assignment.flatten().bincount(minlength=C).tolist()
        print(f"[BalancedArgmax] Final Counts: {final_counts}\n")
    
    return torch.argmax(current_logits, dim=1)


def morphological_clean_mask(mask, h, w, opening_kernel_size=3, closing_kernel_size=3):
    """
    Clean a binary mask using morphological operations.
    
    Applies:
    1. Opening (erosion + dilation): removes small foreground noise
    2. Closing (dilation + erosion): fills small holes in foreground
    
    Args:
        mask: (B, N) binary mask, 1 for foreground, 0 for background
        h, w: spatial dimensions (N = h * w)
        opening_kernel_size: kernel size for opening operation
        closing_kernel_size: kernel size for closing operation
        
    Returns:
        cleaned_mask: (B, N) cleaned binary mask
    """
    B = mask.shape[0]
    device = mask.device
    dtype = mask.dtype
    
    # Reshape to 2D: (B, 1, H, W)
    mask_2d = mask.view(B, 1, h, w).float()
    
    def dilate(x, kernel_size):
        padding = kernel_size // 2
        out = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out
    
    def erode(x, kernel_size):
        padding = kernel_size // 2
        out = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=padding)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out
    
    # Step 1: Opening = Erosion + Dilation
    if opening_kernel_size > 1:
        opened = erode(mask_2d, opening_kernel_size)
        opened = dilate(opened, opening_kernel_size)
    else:
        opened = mask_2d
    
    # Step 2: Closing = Dilation + Erosion
    if closing_kernel_size > 1:
        closed = dilate(opened, closing_kernel_size)
        closed = erode(closed, closing_kernel_size)
    else:
        closed = opened
    
    # Reshape back to (B, N)
    cleaned_mask = closed.view(B, -1)
    
    return cleaned_mask.to(dtype)


def balanced_argmax(stacked, iterations=50, balance_weight=0.1):
    """
    Simple balanced argmax algorithm (legacy).
    
    For better results, use stabilized_balanced_argmax instead.
    """
    n_concepts, h, w = stacked.shape
    total_pixels = h * w
    target_count = total_pixels // n_concepts
    
    biases = torch.zeros(n_concepts, device=stacked.device, dtype=stacked.dtype)
    
    for _ in range(iterations):
        adjusted = stacked + biases.view(-1, 1, 1)
        assignment = adjusted.argmax(dim=0)
        
        counts = torch.bincount(assignment.flatten(), minlength=n_concepts)
        counts = counts[:n_concepts].float()
        
        diff = counts - target_count
        biases = biases - balance_weight * diff / total_pixels
    
    # Final assignment
    adjusted = stacked + biases.view(-1, 1, 1)
    assignment = adjusted.argmax(dim=0)
    
    # Convert to per-concept masks
    masks = torch.zeros_like(stacked)
    for i in range(n_concepts):
        masks[i] = (assignment == i).float()
    
    return masks


def generate_masks(
    similarity_maps, 
    include_background=True, 
    method="stabilized",
    bg_scale=0.95,
    use_morphological_cleaning=True,
    debug=False,
    max_iter=15,
):
    """
    Generate masks from similarity maps using the FreeFuse algorithm.
    
    This implementation follows the diffusers FreeFuse pipeline logic:
    1. Stack all similarity maps including background
    2. Use stabilized_balanced_argmax for concept assignment
    3. Apply background exclusion
    4. Optionally apply morphological cleaning
    
    Args:
        similarity_maps: Dict[name -> (H, W) or (B, N, 1) tensor]
        include_background: Whether background channel participates in argmax
        method: "stabilized" (recommended), "balanced", or "argmax"
        bg_scale: Scaling factor for background channel (higher = more background)
        use_morphological_cleaning: Whether to apply morphological operations
        debug: Whether to print debug info
        
    Returns:
        Dict[name -> (H, W) mask tensor]
    """
    if not similarity_maps:
        return {}
    
    names = list(similarity_maps.keys())
    maps = [similarity_maps[n] for n in names]
    
    # Detect format and get spatial dimensions
    first_map = maps[0]
    if first_map.dim() == 3:
        # (B, N, 1) format from attention - convert to spatial
        B = first_map.shape[0]
        N = first_map.shape[1]
        h = w = int(N ** 0.5)
        if h * w != N:
            # Try to find factors
            for i in range(int(N ** 0.5), 0, -1):
                if N % i == 0:
                    h = i
                    w = N // i
                    break
        
        maps_spatial = []
        for m in maps:
            if m.dim() == 3:
                m = m.squeeze(-1)  # (B, N)
            spatial = m[0].view(h, w)  # Take first batch, reshape to spatial
            maps_spatial.append(spatial)
        maps = maps_spatial
        target_shape = (h, w)
    else:
        # (H, W) format
        target_shape = maps[0].shape[-2:]
        h, w = target_shape
    
    device = maps[0].device
    dtype = maps[0].dtype
    
    # Resize all maps to same shape
    maps_resized = []
    for m in maps:
        if m.shape[-2:] != target_shape:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        maps_resized.append(m.to(device=device, dtype=dtype))
    
    # Separate background if present
    bg_map = None
    bg_key = None
    concept_names = []
    concept_maps = []
    
    for name, m in zip(names, maps_resized):
        if name in ["__background__", "__bg__"]:
            bg_map = m
            bg_key = name
        else:
            concept_names.append(name)
            concept_maps.append(m)
    
    if not concept_maps:
        return {}
    
    # Stack concept maps: (C, H, W)
    stacked = torch.stack(concept_maps, dim=0)
    C = len(concept_maps)
    
    if debug:
        print(f"[generate_masks] {C} concepts, shape={target_shape}")
        for i, name in enumerate(concept_names):
            print(f"  {name}: min={stacked[i].min():.6f}, max={stacked[i].max():.6f}")
    
    if method == "stabilized":
        # Use the sophisticated stabilized_balanced_argmax from diffusers
        # Reshape to (B, C, N) format
        N = h * w
        logits = stacked.view(1, C, N)  # (1, C, N)
        
        # Add background channel if we have background similarity map
        if bg_map is not None and include_background:
            bg_channel = bg_map.view(1, 1, N) * bg_scale
            logits_with_bg = torch.cat([logits, bg_channel], dim=1)  # (1, C+1, N)
            
            # Determine foreground vs background first
            bg_argmax = logits_with_bg.argmax(dim=1)  # (1, N)
            not_background_mask = (bg_argmax != C).float()  # (1, N) - C is bg index
            
            if use_morphological_cleaning:
                not_background_mask = morphological_clean_mask(
                    not_background_mask, h, w,
                    opening_kernel_size=2,
                    closing_kernel_size=2
                )
        else:
            not_background_mask = torch.ones(1, N, device=device, dtype=dtype)
        
        # Run stabilized balanced argmax on concept logits only
        max_indices = stabilized_balanced_argmax(logits, h, w, max_iter=max_iter, debug=debug)  # (1, N)
        
        # Create masks with background exclusion
        result = {}
        for idx, name in enumerate(concept_names):
            concept_mask = (max_indices == idx).float()  # (1, N)
            concept_mask = concept_mask * not_background_mask  # Apply foreground mask
            result[name] = concept_mask[0].view(h, w)  # (H, W)
        
        # Add background mask with consistent naming
        if include_background:
            bg_mask = 1.0 - not_background_mask[0].view(h, w)
            # Use original key if it was __background__, otherwise use _background_
            bg_result_key = bg_key if bg_key else "_background_"
            result[bg_result_key] = bg_mask
                
    elif method == "balanced":
        # Legacy balanced argmax
        if include_background:
            if bg_map is not None:
                bg = bg_map.unsqueeze(0) * bg_scale
            else:
                bg = torch.zeros_like(stacked[0:1])
            stacked = torch.cat([stacked, bg], dim=0)
            concept_names = concept_names + ["_background_"]
        
        mask_tensor = balanced_argmax(stacked)
        
        result = {}
        for i, name in enumerate(concept_names):
            result[name] = mask_tensor[i]
            
    else:
        # Simple argmax
        if include_background:
            if bg_map is not None:
                bg = bg_map.unsqueeze(0) * bg_scale
            else:
                bg = torch.zeros_like(stacked[0:1])
            stacked = torch.cat([stacked, bg], dim=0)
            concept_names = concept_names + ["_background_"]
        
        assignment = stacked.argmax(dim=0)
        result = {}
        for i, name in enumerate(concept_names):
            result[name] = (assignment == i).float()
    
    return result


def resize_mask(mask, target_size):
    """Resize mask to target size."""
    if mask.shape[-2:] == target_size:
        return mask
    
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode='nearest'
    ).squeeze()
