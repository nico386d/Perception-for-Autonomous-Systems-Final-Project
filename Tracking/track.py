import numpy as np
from kalman_3d import Kalman3D

class Track:

    # Single tracked object over time
    
    # id: Unique track identifier
    # kf: Kalman3D filter instance
    # age: Total number of frames since track creation
    # hits: Number of times track was matched with a detection
    # time_since_update: Frames since last detection match
    # class_id: Object class (0=Car, 3=Pedestrian, 5=Cyclist)
    # confidence: Detection confidence at creation
    
    
    # Class variable for generating unique IDs
    _next_id = 1
    
    def __init__(self, detection, dt=0.1, init_velocity=0.0):

        # Initialize new track with first detection
        self.id = Track._next_id
        Track._next_id += 1
        
        # Initialize Kalman filter
        self.kf = Kalman3D(dt=dt)
        measurement = detection.to_measurement()
        self.kf.set_state_from_measurement(
            measurement, 
            init_v=init_velocity,
            init_yaw_rate=0.0,
            P_pos=1.0,
            P_vel=10.0
        )
        
        # Track metadata
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        
        # Store state history for debugging/visualization
        self.history = [self.get_state()]
    
    def predict(self, dt=None):

        # Predict next state using kf
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection):
    
        # Update track with matched detection
        measurement = detection.to_measurement()
        self.kf.update(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection.confidence
        
        # Store state
        self.history.append(self.get_state())
    
    def get_state(self):
    
        # Get current state estimate
        x = self.kf.x
        return {
            'position': x[0:3, 0].copy(),      # (px, py, pz)
            'velocity': x[3:6, 0].copy(),      # (vx, vy, vz)
            'yaw': x[6, 0],                    # yaw angle
            'yaw_rate': x[7, 0],               # angular velocity
            'dimensions': x[8:11, 0].copy(),   # (w, h, l)
        }
    
    def get_bbox(self):
    
        # Get current 3D bounding box for visualization
        state = self.get_state()
        return {
            'center': state['position'],
            'dimensions': state['dimensions'],
            'yaw': state['yaw'],
            'class_id': self.class_id,
            'track_id': self.id,
            'confidence': self.confidence
        }
    
    def is_confirmed(self, min_hits=3):

        # Check if track is reliable
        return self.hits >= min_hits
    
    def is_deleted(self, max_age=3):
    
        # Check if track should be deleted due to inactivity
        return self.time_since_update >= max_age
    
    def mark_missed(self):
        # Mark that no detection was associated this frame
        self.time_since_update += 1
    
    def __repr__(self):
        state = self.get_state()
        return (f"Track(id={self.id}, class={self.class_id}, "
                f"age={self.age}, hits={self.hits}, "
                f"pos={state['position']}, "
                f"time_since_update={self.time_since_update})")
