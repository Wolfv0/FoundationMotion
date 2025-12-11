import numpy as np

class KalmanFilterBB:
    def __init__(self, dt=1.0):
        """
        Initializes a Kalman filter for bounding box tracking.
        The state vector is defined as:
          [center_x, center_y, velocity_x, velocity_y, width, height, velocity_width, velocity_height]
        """
        self.dt = dt
        self.x = np.zeros((8, 1))  # initial state vector
        self.P = np.eye(8) * 1000  # initial state covariance
        
        # State transition matrix for constant velocity model
        self.F = np.array([
            [1, 0, dt, 0,  0, 0,  0, 0],
            [0, 1, 0, dt,  0, 0,  0, 0],
            [0, 0, 1,  0,  0, 0,  0, 0],
            [0, 0, 0,  1,  0, 0,  0, 0],
            [0, 0, 0,  0,  1, 0, dt, 0],
            [0, 0, 0,  0,  0, 1,  0, dt],
            [0, 0, 0,  0,  0, 0,  1, 0],
            [0, 0, 0,  0,  0, 0,  0, 1]
        ])
        
        # Measurement matrix: we measure [center_x, center_y, width, height]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # Measurement noise covariance (tunable)
        self.R = np.eye(4) * 10
        
        # Process noise covariance (tunable)
        self.Q = np.eye(8)

    def predict(self):
        """
        Predicts the next state and error covariance.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, measurement):
        """
        Updates the state with a new measurement.
        :param measurement: list or array [center_x, center_y, width, height]
        """
        z = np.array(measurement).reshape((4, 1))
        y = z - np.dot(self.H, self.x)  # measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        
        self.x = self.x + np.dot(K, y)  # updated state estimate
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)  # updated covariance
        return self.x

def bbox_to_measurement(bbox):
    """
    Converts a bounding box [left, top, right, bottom] to a measurement vector:
    [center_x, center_y, width, height]
    """
    left, top, right, bottom = bbox
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    w = right - left
    h = bottom - top
    return [cx, cy, w, h]

def measurement_to_bbox(state):
    """
    Converts the state vector back to a bounding box [left, top, right, bottom].
    Uses the state: center (first two elements) and dimensions (elements 5 and 6).
    """
    cx, cy, _, _, w, h, _, _ = state.flatten()
    left = cx - w / 2.0
    top = cy - h / 2.0
    right = cx + w / 2.0
    bottom = cy + h / 2.0
    return [left, top, right, bottom]

def smooth_bounding_boxes(bboxes, dt=1.0):
    """
    Smooths a list of bounding boxes using a Kalman filter.
    If a bounding box is None, its value remains None.
    
    :param bboxes: List of bounding boxes in [left, top, right, bottom] format, 
                   with possible None entries.
    :param dt: Time step between frames (default is 1.0).
    :return: List of smoothed bounding boxes (or None for missing detections).
    """
    kf = KalmanFilterBB(dt)
    smoothed_bboxes = []
    for bbox in bboxes:
        # If bbox is None, simply predict and output None.
        if bbox is None:
            kf.predict()  # Advance the state without an update.
            smoothed_bboxes.append(None)
            continue
        
        # Otherwise, smooth the valid bounding box.
        measurement = bbox_to_measurement(bbox)
        kf.predict()            # Predict the next state.
        kf.update(measurement)  # Update with the current measurement.
        smooth_bbox = measurement_to_bbox(kf.x)
        smoothed_bboxes.append(smooth_bbox)
    
    return smoothed_bboxes

def smooth_all_bboxes(data, dt=1.0):
    """
    Smooths all bounding box lists in the given dictionary using the Kalman filter.
    It traverses the dictionary and applies smoothing to:
      - The person's overall 'bbox'
      - 'left_hand' and 'right_hand' bounding boxes
      - In-hand object bounding boxes: 'left_hand_1st_obj' and 'right_hand_1st_obj'
    If a bounding box value is None, it remains None.
    
    :param data: Dictionary structured as shown in the example.
    :param dt: Time step between frames (default is 1.0).
    :return: The dictionary with smoothed bounding boxes.
    """
    for person_id, person_data in data.items():
        # Smooth person's overall bounding box list.
        if 'bbox' in person_data and isinstance(person_data['bbox'], list):
            person_data['bbox'] = smooth_bounding_boxes(person_data['bbox'], dt)
        
        # Process left_hand and right_hand entries.
        for hand in ['left_hand', 'right_hand']:
            if hand in person_data and isinstance(person_data[hand], dict):
                # Smooth the hand bounding boxes.
                if 'bbox' in person_data[hand] and isinstance(person_data[hand]['bbox'], list):
                    person_data[hand]['bbox'] = smooth_bounding_boxes(person_data[hand]['bbox'], dt)
                # Smooth the in-hand object bounding boxes.
                key_obj = f"{hand}_1st_obj"
                if key_obj in person_data[hand] and isinstance(person_data[hand][key_obj], list):
                    person_data[hand][key_obj] = smooth_bounding_boxes(person_data[hand][key_obj], dt)
    return data

# --- Example usage ---

if __name__ == '__main__':
    # Example dictionary mimicking the provided structure.
    # (Here, actual numerical values should replace left, top, right, bottom.)
    sample_data = {
        'person_00': {
            'bbox': [
                [100, 200, 150, 250],
                None,
                [105, 208, 155, 258]
            ],
            'left_hand': {
                'bbox': [
                    [90, 190, 130, 230],
                    None,
                    [92, 192, 132, 232]
                ],
                'left_hand_1st_obj': [
                    None, None, [95, 195, 125, 225]
                ]
            },
            'right_hand': {
                'bbox': [
                    [110, 210, 160, 260],
                    [112, 212, 162, 262],
                    None
                ],
                'right_hand_1st_obj': [
                    [115, 215, 165, 265],
                    None,
                    [118, 218, 168, 268]
                ]
            }
        },
        'person_01': {
            'bbox': [
                [200, 300, 250, 350],
                [202, 302, 252, 352],
                None
            ],
            'left_hand': {
                'bbox': [
                    [190, 290, 230, 330],
                    [192, 292, 232, 332],
                    [194, 294, 234, 334]
                ],
                'left_hand_1st_obj': [
                    None, [195, 295, 235, 335], None
                ]
            },
            'right_hand': {
                'bbox': [
                    [210, 310, 260, 360],
                    None,
                    [212, 312, 262, 362]
                ],
                'right_hand_1st_obj': [
                    [215, 315, 265, 365],
                    [217, 317, 267, 367],
                    None
                ]
            }
        }
    }
    
    # Apply the smoothing function to the entire dictionary.
    smoothed_data = smooth_all_bboxes(sample_data, dt=1.0)
    
    # Print the resulting smoothed bounding boxes.
    import pprint
    pprint.pprint(smoothed_data)
