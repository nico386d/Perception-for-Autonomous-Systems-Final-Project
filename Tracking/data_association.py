import numpy as np
from scipy.optimize import linear_sum_assignment

def associate_detections_to_tracks(detections, tracks, gate_threshold=9.21):
    
    # Associate detections to tracks using Mahalanobis distance and Hungarian algorithm
    
    # inputs:
    # detections: List of Detection objects
    # tracks: List of Track objects
    # gate_threshold: Maximum Mahalanobis distance to consider a valid match
    
    # outputs:
    # matches: List of (detection_idx, track_idx) tuples
    # unmatched_detections: List of detection indices
    # unmatched_tracks: List of track indices
    
    
    # Handle empty cases
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # Cost matrix (Mahalanobis distances) calculation
    cost_matrix = np.zeros((len(detections), len(tracks)))
    
    for d, detection in enumerate(detections):
        measurement = detection.to_measurement()
        for t, track in enumerate(tracks):
            distance = track.kf.gating_mahalanobis(measurement)
            cost_matrix[d, t] = distance
    
    # Apply gating by setting gated pairs to large cost
    gated_mask = cost_matrix > gate_threshold
    cost_matrix[gated_mask] = 1e6 
    
    # Hungarian algorithm for optimal assignment
    detection_indices, track_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out assignments that were gated
    matches = []
    for d_idx, t_idx in zip(detection_indices, track_indices):
        if cost_matrix[d_idx, t_idx] < 1e6:  # Not gated
            matches.append((d_idx, t_idx))
    
    # Find unmatched detections and tracks
    matched_detection_indices = set([m[0] for m in matches])
    matched_track_indices = set([m[1] for m in matches])
    
    unmatched_detections = [d for d in range(len(detections)) 
                           if d not in matched_detection_indices]
    unmatched_tracks = [t for t in range(len(tracks)) 
                       if t not in matched_track_indices]
    
    return matches, unmatched_detections, unmatched_tracks
