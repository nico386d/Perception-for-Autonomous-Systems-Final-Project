import numpy as np
from track import Track
from data_association import associate_detections_to_tracks

class MultiObjectTracker:
    
    # Multi-object tracker using Kalman filtering and Hungarian data association
    
    # Manages multiple tracks over time and handles:
    # 1) Track creation from new detections
    # 2) Track update with matched detections
    # 3) Track deletion when lost
    
    
    def __init__(self, dt=0.1, max_age=3, min_hits=3, gate_threshold=14.07):

        # Initialize tracker parameters
        self.dt = dt
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_threshold = gate_threshold
        
        self.tracks = []
        self.frame_count = 0
        
        # stats
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def update(self, detections):
    
        # Update tracker with new detections
        self.frame_count += 1
        
        # Predict all existing tracks
        for track in self.tracks:
            track.predict(self.dt)
        
        # Associate detections to tracks
        matches, unmatched_detections, unmatched_tracks = \
            associate_detections_to_tracks(
                detections, 
                self.tracks, 
                self.gate_threshold
            )
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx], dt=self.dt)
            self.tracks.append(new_track)
            self.total_tracks_created += 1
        
        # Delete old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]
        
        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed(self.min_hits)]
        
        return confirmed_tracks
    
    def get_all_tracks(self):

        # Get all active tracks
        return self.tracks
    
    def get_confirmed_tracks(self):

        # Get only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
    
    def get_track_states(self):

        # Get states of all confirmed tracks
        confirmed = self.get_confirmed_tracks()
        return [track.get_state() for track in confirmed]
    
    def get_track_bboxes(self):

        # Get 3D bounding boxes of all confirmed tracks
        confirmed = self.get_confirmed_tracks()
        return [track.get_bbox() for track in confirmed]
    
    def reset(self):

        # Reset tracker to initial state
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def get_statistics(self):
        
        # Get tracking statistics
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracks),
            'confirmed_tracks': len(self.get_confirmed_tracks()),
            'total_created': self.total_tracks_created,
            'total_deleted': self.total_tracks_deleted,
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"MultiObjectTracker(frame={stats['frame_count']}, "
                f"active={stats['active_tracks']}, "
                f"confirmed={stats['confirmed_tracks']})")


class OnlineTracker(MultiObjectTracker):
    # Online tracker with history storage for evaluation
    # Stores track history for computing metrics of tracking performance
    
    def __init__(self, dt=0.1, max_age=3, min_hits=3, gate_threshold=14.07):
        super().__init__(dt, max_age, min_hits, gate_threshold)
        self.track_history = {}  # track_id -> list of states per frame
    
    def update(self, detections):
        # Update and store history
        confirmed_tracks = super().update(detections)
        
        # Store states for this frame
        for track in confirmed_tracks:
            if track.id not in self.track_history:
                self.track_history[track.id] = []
            self.track_history[track.id].append({
                'frame': self.frame_count,
                'state': track.get_state(),
                'bbox': track.get_bbox()
            })
        
        return confirmed_tracks
    
    def get_track_history(self, track_id):

        # Get history for specific track
        return self.track_history.get(track_id, [])
    
    def get_all_track_histories(self):

        # Get history for all tracks
        return self.track_history
