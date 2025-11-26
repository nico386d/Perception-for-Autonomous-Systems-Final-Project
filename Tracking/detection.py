import numpy as np

class Detection3D:
    
    def __init__(self, position, yaw, dimensions, class_id, confidence, frame_id=None):
        
        # Initialize 3D detection
        # position: (px, py, pz)
        # yaw: Orientation angle
        # dimensions: (width, height, length)
        # class_id: Object class (0=Car, 3=Pedestrian, 5=Cyclist)
        # confidence: Detection confidence score (0, 1)
        # frame_id: Optional frame number
        
        self.position = np.asarray(position, dtype=np.float64)
        self.yaw = float(yaw)
        self.dimensions = np.asarray(dimensions, dtype=np.float64)
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.frame_id = frame_id
        
        # Validate dimensions
        assert self.position.shape == (3,) # (px, py, pz)
        assert self.dimensions.shape == (3,) # (w, h, l)
        assert 0 <= self.confidence <= 1 
    
    def to_measurement(self):  # Convert to Kalman filter measurements

        return np.array([
            self.position[0],    # px
            self.position[1],    # py
            self.position[2],    # pz
            self.yaw,            # yaw
            self.dimensions[0],  # width
            self.dimensions[1],  # height
            self.dimensions[2],  # length
        ])
    
    @classmethod
    def from_measurement(cls, measurement, class_id=0, confidence=1.0, frame_id=None): # Create Detection3D from measurement vector.
       
        measurement = np.asarray(measurement)
        return cls(
            position=measurement[0:3],
            yaw=measurement[3],
            dimensions=measurement[4:7],
            class_id=class_id,
            confidence=confidence,
            frame_id=frame_id
        )
    
    def get_corners(self): # Get 8 corners of the 3D bbox for visualization
        
        w, h, l = self.dimensions
        px, py, pz = self.position
        
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        corners = np.array([x_corners, y_corners, z_corners])
        
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate
        corners = R @ corners
        corners[0, :] += px
        corners[1, :] += py
        corners[2, :] += pz
        
        return corners.T
    
    def __repr__(self):
        return (f"Detection3D(pos={self.position}, yaw={self.yaw:.2f}, "
                f"dim={self.dimensions}, class={self.class_id}, "
                f"conf={self.confidence:.2f})")
    
