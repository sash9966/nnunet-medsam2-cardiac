import numpy as np
import torch
import SimpleITK as sitk

def run_medsam2_with_mask_prompts(ct_volume, positive_mask, negative_mask, 
                                  sam2_model, device):
    """
    Run MedSAM2 with positive and negative mask prompts.
    
    Args:
        ct_volume: 3D CT image (D, H, W)
        positive_mask: 3D binary mask of target region (D, H, W)
        negative_mask: 3D binary mask of regions to EXCLUDE (D, H, W)
        sam2_model: Loaded MedSAM2 model
        device: torch device
    """
    import torch
    
    # Initialize predictor
    with torch.inference_mode():
        # Start video/3D session
        inference_state = sam2_model.init_state(
            ct_volume, 
            device=device
        )
        
        # Select a representative slice (middle of the structure)
        z_coords = np.where(positive_mask.sum(axis=(1,2)) > 0)[0]
        if len(z_coords) == 0:
            return None
            
        mid_slice_idx = z_coords[len(z_coords) // 2]
        
        # Get 2D masks for the middle slice
        pos_mask_2d = positive_mask[mid_slice_idx]  # (H, W)
        neg_mask_2d = negative_mask[mid_slice_idx]  # (H, W)
        
        # Add positive mask prompt
        _, out_obj_ids, out_mask_logits = sam2_model.add_new_mask(
            inference_state=inference_state,
            frame_idx=mid_slice_idx,
            obj_id=1,  # Object ID
            mask=torch.from_numpy(pos_mask_2d).to(device).unsqueeze(0),  # (1, H, W)
        )
        
        # Add negative mask prompt if present
        if neg_mask_2d.sum() > 0:
            # MedSAM2/SAM2 way to add negative prompts:
            # Use point prompts with label=0 (negative) at locations in negative mask
            neg_coords = np.argwhere(neg_mask_2d > 0)
            
            # Sample points from negative mask (don't use all, too many)
            if len(neg_coords) > 50:  # Limit to 50 negative points
                indices = np.random.choice(len(neg_coords), 50, replace=False)
                neg_coords = neg_coords[indices]
            
            if len(neg_coords) > 0:
                # Add negative point prompts
                points = torch.from_numpy(neg_coords[:, [1, 0]]).float().to(device)  # (N, 2) [x,y]
                labels = torch.zeros(len(points), dtype=torch.int32).to(device)  # 0 = negative
                
                _, _, _ = sam2_model.add_new_points(
                    inference_state=inference_state,
                    frame_idx=mid_slice_idx,
                    obj_id=1,
                    points=points.unsqueeze(0),  # (1, N, 2)
                    labels=labels.unsqueeze(0),  # (1, N)
                )
        
        # Propagate through entire 3D volume
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    # Reconstruct 3D volume
    output_volume = np.zeros_like(ct_volume, dtype=bool)
    for frame_idx in sorted(video_segments.keys()):
        if 1 in video_segments[frame_idx]:  # obj_id = 1
            output_volume[frame_idx] = video_segments[frame_idx][1][0]  # (H, W)
    
    return output_volume.astype(np.uint8)