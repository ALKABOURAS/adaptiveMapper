class SimpleKalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=0.1, estimated_error=1.0):
        self.Q = process_noise       # Process covariance
        self.R = measurement_noise   # Measurement covariance
        self.P = estimated_error     # Estimation error covariance
        self.x = 0.0                 # Value estimate

    def update(self, measurement):
        # Prediction update
        self.P = self.P + self.Q

        # Measurement update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x