import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class Kalman3D:
    def __init__(self, dt=0.1,
                 process_noise_position=1.0,
                 process_noise_velocity=1.0,
                 process_noise_yaw=0.1,
                 process_noise_yaw_rate=0.1,
                 process_noise_dim=1e-3,
                 meas_noise_position=1.0,
                 meas_noise_yaw=0.1,
                 meas_noise_dim=1e-2): # Tuning when we get the pipeline working

        self.dt = dt # time step
        self.m = 7 # Measurement dimension of 3d_bbox (px,py,pz, yaw, width, height, length)
        self.n = 11 # State dimension: (px,py,pz, vx,vy,vz, yaw, yaw_rate, width, height, length)

        # Initialize state and covariances
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n) * 1.0

        # State transition matrix F
        self._build_F(dt)

        # Measurement matrix H: maps state to measurement
        self.H = np.zeros((self.m, self.n))
        # position
        self.H[0, 0] = 1.0  # px
        self.H[1, 1] = 1.0  # py
        self.H[2, 2] = 1.0  # pz
        self.H[3, 6] = 1.0  # yaw
        self.H[4, 8] = 1.0  # width
        self.H[5, 9] = 1.0  # height 
        self.H[6,10] = 1.0  # length

        # Measurement noise R
        self.R = np.diag([
            meas_noise_position,  # px
            meas_noise_position,  # py
            meas_noise_position,  # pz
            meas_noise_yaw,       # yaw
            meas_noise_dim,       # width
            meas_noise_dim,       # height
            meas_noise_dim        # length
        ])

        # Process noise Q
        q = np.zeros((self.n, self.n))
        q[0:3, 0:3] = np.eye(3) * process_noise_position
        q[3:6, 3:6] = np.eye(3) * process_noise_velocity
        q[6,6]   = process_noise_yaw
        q[7,7]   = process_noise_yaw_rate
        q[8,8]   = process_noise_dim
        q[9,9]   = process_noise_dim
        q[10,10] = process_noise_dim
        self.Q = q

    def _build_F(self, dt):

        self.F = np.eye(self.n)
        # position updated by velocity
        self.F[0,3] = dt
        self.F[1,4] = dt
        self.F[2,5] = dt
        # yaw updated by yaw_rate
        self.F[6,7] = dt
        # px, py, pz assumed constant
        self.dt = dt

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        else:
            self._build_F(dt)

        # linear predict
        self.x = self.F @ self.x
        # covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Normalize yaw component
        self.x[6,0] = wrap_angle(self.x[6,0])
        return self.x.copy(), self.P.copy()

    def update(self, z, R_override=None):

        z = np.asarray(z).reshape((self.m,1))
        R = self.R if R_override is None else R_override

        # Innovation
        y = z - (self.H @ self.x)
        y[3,0] = wrap_angle(y[3,0])

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[6,0] = wrap_angle(self.x[6,0])

        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy(), self.P.copy()

    def gating_mahalanobis(self, z):
        # Compute Mahalanobis distance for gating
        z = np.asarray(z).reshape((self.m,1))
        z_pred = self.H @ self.x
        y = z - z_pred
        y[3,0] = wrap_angle(y[3,0])
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        return d2

    def set_state_from_measurement(self, z, init_v=0.0, init_yaw_rate=0.0, P_pos=1.0, P_vel=10.0):
        # Initialize state from first measurement
        z = np.asarray(z).reshape((self.m,1))
        self.x = np.zeros((self.n,1))
        self.x[0:3,0] = z[0:3,0]
        self.x[3:6,0] = init_v  # set initial linear velocity
        self.x[6,0] = z[3,0]    # yaw
        self.x[7,0] = init_yaw_rate # yaw rate
        self.x[8:11,0] = z[4:7,0]  # width, height, length

        # Initialize covariance with larger uncertainty for velocities
        self.P = np.eye(self.n)
        self.P[0:3,0:3] *= P_pos
        self.P[3:6,3:6] *= P_vel
        self.P[6,6] *= 0.5
        self.P[7,7] *= 1.0
        self.P[8:11,8:11] *= 0.1
