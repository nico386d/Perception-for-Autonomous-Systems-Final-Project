#!/usr/bin/env python3
"""
Load and visualize 3D bounding boxes from SAM3 AABB files for tracking.

Usage:
    1. Copy your 3dsam folder to the project directory
    2. Run: python load_sam3_boxes.py --data_dir /path/to/3dsam/sequences/seq_01
"""

import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add Tracking to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import load_detections_from_aabb
from detection import Detection3D


def load_sam3_sequence(data_dir: Path, visualize=False):
    """
    Load all frames of 3D detections from SAM3 AABB format.
    
    Expected structure:
        data_dir/
            image_02/
                000000/
                    boxes3d_aabb.txt, mask_full.png, ...
                000001/
                    boxes3d_aabb.txt, mask_full.png, ...
    
    Args:
        data_dir: Path to sequence directory (e.g., 3dsam/sequences/seq_01)
        visualize: Whether to save visualizations
        
    Returns:
        dict with frame data and statistics
    """
    print(f"\n{'='*70}")
    print(f"Loading SAM3 3D Bounding Boxes")
    print(f"{'='*70}\n")
    
    image_02_dir = data_dir / "image_02"
    
    if not image_02_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_02_dir}")
    
    # Get all frame folders (000000, 000001, etc.)
    frame_folders = sorted([d for d in image_02_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(frame_folders)} frame folders")
    
    # Setup visualization
    if visualize:
        viz_dir = Path(__file__).resolve().parent / "pipeline_outputs" / "sam3_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}\n")
    
    # Get original images for visualization
    orig_img_dir = data_dir.parent.parent.parent / "Classification" / "Data" / "34759_final_project_raw" / data_dir.name / "image_02" / "data"
    has_orig_images = orig_img_dir.exists()
    if visualize and not has_orig_images:
        print(f"Warning: Original images not found at {orig_img_dir}")
    
    # Load all frames
    all_frames = []
    total_detections = 0
    class_counts = {0: 0, 1: 0, 2: 0}  # car, pedestrian, cyclist
    
    for frame_idx, frame_folder in enumerate(frame_folders):
        print(f"Processing frame {frame_idx}/{len(frame_folders)-1}", end='\r')
        
        # Load detections from boxes3d_aabb.txt
        box_path = frame_folder / "boxes3d_aabb.txt"
        if not box_path.exists():
            print(f"\nWarning: No boxes file at {box_path}")
            continue
        
        detections = load_detections_from_aabb(box_path, min_conf=0.0)
        
        frame_data = {
            'frame_idx': frame_idx,
            'detections': detections,
            'num_detections': len(detections),
        }
        
        total_detections += len(detections)
        
        # Count classes
        for det in detections:
            if det.class_id in class_counts:
                class_counts[det.class_id] += 1
        
        all_frames.append(frame_data)
        
        # Visualization
        if visualize and len(detections) > 0:
            # Try to load original image or mask
            if has_orig_images:
                img_path = list(orig_img_dir.glob(f"{frame_folder.name}.*"))[0]
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Use the mask_full.png from SAM3 output
                mask_path = frame_folder / "mask_full.png"
                if mask_path.exists():
                    img = cv2.imread(str(mask_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    continue
            
            # Draw 3D boxes (projected to image plane would require camera params)
            # For now, just draw center points and labels
            for det in detections:
                # Use simple projection (you'll need proper camera matrix for real projection)
                # For now, just show detection info as text
                class_names = {0: "car", 1: "pedestrian", 2: "cyclist"}
                cls_name = class_names.get(det.class_id, f"cls_{det.class_id}")
                
                # Draw text with detection info
                text = f"{cls_name} {det.confidence:.2f} z={det.position[2]:.1f}m"
                cv2.putText(img, text, (10, 30 + 30*detections.index(det)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_text = f"Frame {frame_idx} | Detections: {len(detections)}"
            cv2.putText(img, info_text, (10, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            plt.imsave(viz_dir / f"{frame_idx:06d}.png", img)
    
    print(f"\n\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Total frames:         {len(all_frames)}")
    print(f"Total detections:     {total_detections}")
    print(f"Avg detections/frame: {total_detections / len(all_frames):.2f}")
    print(f"\nClass distribution:")
    print(f"  Cars:        {class_counts[0]}")
    print(f"  Pedestrians: {class_counts[1]}")
    print(f"  Cyclists:    {class_counts[2]}")
    
    # Save to pickle
    output_dir = Path(__file__).resolve().parent / "pipeline_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sam3_detections.pkl"
    
    result = {
        'num_frames': len(all_frames),
        'frames': all_frames,
        'total_detections': total_detections,
        'class_counts': class_counts,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\nSaved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Load 3D bounding boxes from SAM3 AABB format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to sequence directory (e.g., 3dsam/sequences/seq_01)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease copy your 3dsam folder to the project and provide the correct path.")
        print("Example: python load_sam3_boxes.py --data_dir /path/to/3dsam/sequences/seq_01")
        return
    
    result = load_sam3_sequence(data_dir, visualize=args.visualize)
    
    print("\n✓ Done! You can now use these detections for 3D tracking.")


if __name__ == "__main__":
    main()
