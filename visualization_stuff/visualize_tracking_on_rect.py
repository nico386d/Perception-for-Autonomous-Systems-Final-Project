#!/usr/bin/env python3
"""
Overlay tracking results on rectified images.
Loads tracking results pkl and draws on rectified images from Classification/Data.
"""
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import cv2

# Color palette for tracks
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (192, 192, 192)
]

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def project_3d_to_2d(point_3d):
    """Project 3D point to image coordinates."""
    fx, fy = 721.5377, 721.5377
    cx, cy = 609.5593, 172.8540
    x, y, z = point_3d
    if z <= 0:
        return None
    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)
    return (u, v)


def draw_track_label(img, bbox, track_id):
    """Draw track ID and label at center."""
    center_2d = project_3d_to_2d(bbox['center'])
    if center_2d is None:
        return
    
    color = COLORS[track_id % len(COLORS)]
    
    # Draw center point
    cv2.circle(img, center_2d, 8, color, -1)
    cv2.circle(img, center_2d, 10, (255, 255, 255), 2)
    
    # Draw label
    class_name = CLASS_NAMES.get(bbox['class_id'], "Unknown")
    label = f"ID:{track_id} {class_name}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    label_pos = (center_2d[0] - 30, center_2d[1] - 15)
    cv2.rectangle(img, 
                  (label_pos[0] - 5, label_pos[1] - text_h - 5),
                  (label_pos[0] + text_w + 5, label_pos[1] + 5),
                  color, -1)
    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw distance
    dist = np.linalg.norm(bbox['center'])
    cv2.putText(img, f"{dist:.1f}m", (center_2d[0] - 20, center_2d[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_trajectory(img, track_id, history):
    """Draw trajectory line for a track."""
    if len(history) < 2:
        return
    
    color = COLORS[track_id % len(COLORS)]
    
    points_2d = []
    for h in history[-20:]:  # Last 20 positions
        pt_2d = project_3d_to_2d(h['bbox']['center'])
        if pt_2d:
            points_2d.append(pt_2d)
    
    # Draw trajectory line
    for i in range(len(points_2d) - 1):
        cv2.line(img, points_2d[i], points_2d[i+1], color, 2)


def main():
    parser = argparse.ArgumentParser(description="Overlay tracking on rectified images")
    parser.add_argument("--sequence", type=str, default="seq_01",
                       choices=["seq_01", "seq_02", "seq_03"])
    parser.add_argument("--results", type=str, 
                       help="Path to tracking results pkl file (optional)")
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    
    # Load tracking results
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = Path(__file__).resolve().parent / "pipeline_outputs" / f"{args.sequence}_tracking_results.pkl"
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run tracking first with run_tracking_sam3.py")
        return
    
    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    track_histories = results['track_histories']
    
    # Rectified images directory
    img_dir = project_root / "Classification" / "Data" / "34759_final_project_rect" / args.sequence / "image_02" / "data"
    
    if not img_dir.exists():
        print(f"Error: Rectified image directory not found: {img_dir}")
        return
    
    # Output directory
    output_dir = Path(__file__).resolve().parent / "pipeline_outputs" / f"{args.sequence}_rect_track"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    img_files = sorted(img_dir.glob("*.png"))
    
    print(f"\n{'='*70}")
    print(f"Overlaying tracking on rectified images: {args.sequence}")
    print(f"{'='*70}")
    print(f"Images: {len(img_files)}")
    print(f"Tracks: {len(track_histories)}")
    print(f"Output: {output_dir}")
    print()
    
    # Organize tracks by frame
    tracks_by_frame = {}
    for track_id, history in track_histories.items():
        for entry in history:
            frame = entry['frame']
            if frame not in tracks_by_frame:
                tracks_by_frame[frame] = []
            tracks_by_frame[frame].append({
                'track_id': track_id,
                'bbox': entry['bbox']
            })
    
    # Process each frame
    for frame_idx, img_file in enumerate(img_files):
        print(f"Frame {frame_idx}/{len(img_files)-1}", end='\r')
        
        # Load rectified image
        img = cv2.imread(str(img_file))
        
        # Draw trajectories
        for track_id, history in track_histories.items():
            # Only draw trajectory if track is active in current or recent frames
            if any(h['frame'] <= frame_idx <= h['frame'] + 5 for h in history):
                # Get history up to current frame
                current_history = [h for h in history if h['frame'] <= frame_idx]
                draw_trajectory(img, track_id, current_history)
        
        # Draw track labels for tracks in this frame
        if frame_idx in tracks_by_frame:
            for track_info in tracks_by_frame[frame_idx]:
                draw_track_label(img, track_info['bbox'], track_info['track_id'])
        
        # Info overlay
        num_tracks = len(tracks_by_frame.get(frame_idx, []))
        info = [
            f"Frame: {frame_idx}/{len(img_files)-1}",
            f"Active Tracks: {num_tracks}",
            "Rectified Images"
        ]
        
        y = 30
        for line in info:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
        
        # Save
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}.png"), img)
    
    print()
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")
    print(f"Visualizations saved: {output_dir}")
    print(f"\nCreate video:")
    print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{output_dir}/*.png' \\")
    print(f"         -c:v libx264 -pix_fmt yuv420p {args.sequence}_rect_tracking.mp4")


if __name__ == "__main__":
    main()
