import numpy as np

# --- Kalman Filter Implementation (same as before) ---
class KalmanFilterBB:
    def __init__(self, dt=1.0):
        """
        Initializes a Kalman filter for bounding box tracking.
        State vector:
          [center_x, center_y, velocity_x, velocity_y, width, height, velocity_width, velocity_height]
        """
        self.dt = dt
        self.x = np.zeros((8, 1))
        self.P = np.eye(8) * 1000
        
        # Constant velocity model.
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
        
        # We measure [center_x, center_y, width, height]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        self.R = np.eye(4) * 10  # measurement noise
        self.Q = np.eye(8)       # process noise

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, measurement):
        z = np.array(measurement).reshape((4, 1))
        y = z - np.dot(self.H, self.x)  # innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x

def bbox_to_measurement(bbox):
    """
    Converts a bounding box [left, top, right, bottom] into
    a measurement vector [center_x, center_y, width, height].
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
    Uses center (first two elements) and dimensions (elements 5 and 6).
    """
    cx, cy, _, _, w, h, _, _ = state.flatten()
    left = cx - w / 2.0
    top = cy - h / 2.0
    right = cx + w / 2.0
    bottom = cy + h / 2.0
    return [left, top, right, bottom]

def smooth_bounding_boxes(bboxes, dt=1.0):
    """
    Smooths a list of bounding boxes with a Kalman filter.
    Missing detections (None) remain None.
    :param bboxes: List of bounding boxes ([left, top, right, bottom]) or None.
    :param dt: Time step between frames.
    :return: List of smoothed bounding boxes.
    """
    kf = KalmanFilterBB(dt)
    smoothed_bboxes = []
    for bbox in bboxes:
        if bbox is None:
            kf.predict()  # Advance state when detection is missing.
            smoothed_bboxes.append(None)
        else:
            measurement = bbox_to_measurement(bbox)
            kf.predict()
            kf.update(measurement)
            smooth_bbox = measurement_to_bbox(kf.x)
            smoothed_bboxes.append(smooth_bbox)
    return smoothed_bboxes

# --- Reformatting Functions ---

def reformat_res_ls_to_person_format(res_ls):
    """
    Converts the dictionary of frame results (with keys as frame indexes) into a per-person dictionary.
    For each person (e.g. 'person_00'), every attribute ('bbox', 'left_hand', etc.) becomes a list
    over the full frame index range from the minimum to maximum frame (inclusive).
    If a frame index is missing in res_ls, that frame is treated as no detection.
    
    :param res_ls: Dictionary with keys as frame indexes and values as frame dictionaries.
    :return: (person_data, all_frame_indices) where person_data is keyed by person id.
    """
    if not res_ls:
        return {}, []
    
    # Determine the full range of frame indices.
    min_frame = min(res_ls.keys())
    max_frame = max(res_ls.keys())
    all_frame_indices = list(range(min_frame, max_frame + 1))
    
    # Determine all person ids appearing in any frame.
    person_ids = set()
    for frame in res_ls.values():
        for pid in frame.keys():
            person_ids.add(pid)
    
    # Initialize the per-person dictionary.
    output = {}
    for pid in person_ids:
        output[pid] = {
            'bbox': [None] * len(all_frame_indices),
            'left_hand': {
                'bbox': [None] * len(all_frame_indices),
                'left_hand_1st_obj': [None] * len(all_frame_indices)
            },
            'right_hand': {
                'bbox': [None] * len(all_frame_indices),
                'right_hand_1st_obj': [None] * len(all_frame_indices)
            }
        }
    
    # Fill the lists using the full frame index range.
    for idx, frame_index in enumerate(all_frame_indices):
        # If the frame index is missing in res_ls, treat it as an empty detection.
        frame = res_ls.get(frame_index, {})
        for pid in person_ids:
            if pid in frame:
                pdata = frame[pid]
                if 'bbox' in pdata:
                    output[pid]['bbox'][idx] = pdata['bbox']
                if 'left_hand' in pdata:
                    lh = pdata['left_hand']
                    if 'bbox' in lh:
                        output[pid]['left_hand']['bbox'][idx] = lh['bbox']
                    if 'left_hand_1st_obj' in lh:
                        output[pid]['left_hand']['left_hand_1st_obj'][idx] = lh['left_hand_1st_obj']
                if 'right_hand' in pdata:
                    rh = pdata['right_hand']
                    if 'bbox' in rh:
                        output[pid]['right_hand']['bbox'][idx] = rh['bbox']
                    if 'right_hand_1st_obj' in rh:
                        output[pid]['right_hand']['right_hand_1st_obj'][idx] = rh['right_hand_1st_obj']
    return output, all_frame_indices

def smooth_all_person_data(person_data, dt=1.0):
    """
    Applies Kalman filter smoothing to each bounding box list in the per-person dictionary.
    Smooths:
      - Person 'bbox'
      - left_hand: 'bbox' and 'left_hand_1st_obj'
      - right_hand: 'bbox' and 'right_hand_1st_obj'
    
    :param person_data: Dictionary keyed by person id, each attribute is a list.
    :param dt: Time step between frames.
    :return: The same dictionary with smoothed bounding box lists.
    """
    for pid, pdata in person_data.items():
        pdata['bbox'] = smooth_bounding_boxes(pdata['bbox'], dt)
        pdata['left_hand']['bbox'] = smooth_bounding_boxes(pdata['left_hand']['bbox'], dt)
        pdata['left_hand']['left_hand_1st_obj'] = smooth_bounding_boxes(pdata['left_hand']['left_hand_1st_obj'], dt)
        pdata['right_hand']['bbox'] = smooth_bounding_boxes(pdata['right_hand']['bbox'], dt)
        pdata['right_hand']['right_hand_1st_obj'] = smooth_bounding_boxes(pdata['right_hand']['right_hand_1st_obj'], dt)
    return person_data

def reformat_person_format_to_res_ls(person_data, all_frame_indices):
    """
    Converts the per-person dictionary (each attribute is a list over frames)
    back to the original res_ls format (a dictionary keyed by frame index).
    
    :param person_data: Dictionary keyed by person id.
    :param all_frame_indices: Full list of frame indexes.
    :return: Dictionary with keys as frame indexes.
    """
    total_frames = len(all_frame_indices)
    res_ls = {}
    for idx in range(total_frames):
        frame_dict = {}
        for pid, pdata in person_data.items():
            frame_dict[pid] = {
                'bbox': pdata['bbox'][idx],
                'left_hand': {
                    'bbox': pdata['left_hand']['bbox'][idx],
                    'left_hand_1st_obj': pdata['left_hand']['left_hand_1st_obj'][idx]
                },
                'right_hand': {
                    'bbox': pdata['right_hand']['bbox'][idx],
                    'right_hand_1st_obj': pdata['right_hand']['right_hand_1st_obj'][idx]
                }
            }
        res_ls[all_frame_indices[idx]] = frame_dict
    return res_ls

# --- Example Usage ---

if __name__ == '__main__':
    # Sample input: res_ls is a dictionary keyed by frame index.
    # Note: Some frame indexes are skipped (e.g., frame 1 is missing).
    res_ls = {
        0: {'person_00': {
                'bbox': [366, 0, 802, 458],
                'left_hand': {
                    'bbox': [618, 219, 772, 398],
                    'left_hand_1st_obj': [634, 193, 748, 331]
                },
                'right_hand': {
                    'bbox': [465, 400, 618, 485],
                    'right_hand_1st_obj': [452, 417, 1108, 670]
                }
            }
        },
        2: {'person_00': {
                'bbox': [365, 0, 802, 457],
                'left_hand': {
                    'bbox': [618, 197, 772, 402],
                    'left_hand_1st_obj': [638, 195, 754, 329]
                },
                'right_hand': {
                    'bbox': [457, 400, 619, 485],
                    'right_hand_1st_obj': [455, 417, 1112, 677]
                }
            }
        },
        4: {'person_00': {
                'bbox': [364, 0, 801, 457],
                'left_hand': {
                    'bbox': [617, 190, 771, 400],
                    'left_hand_1st_obj': [636, 190, 752, 330]
                },
                'right_hand': {
                    'bbox': [456, 395, 618, 483],
                    'right_hand_1st_obj': [454, 415, 1110, 675]
                }
            }
        }
    }
    
    # Step 1: Convert the frame-indexed dictionary into per-person format.
    person_data, all_frame_indices = reformat_res_ls_to_person_format(res_ls)
    
    # Step 2: Smooth all bounding box lists using the Kalman filter.
    smoothed_person_data = smooth_all_person_data(person_data, dt=1.0)
    
    # Step 3: Reformat the per-person dictionary back into the dictionary keyed by frame index.
    smoothed_res_ls = reformat_person_format_to_res_ls(smoothed_person_data, all_frame_indices)
    
    # For demonstration, print the final smoothed dictionary.
    import pprint
    pprint.pprint(smoothed_res_ls)
