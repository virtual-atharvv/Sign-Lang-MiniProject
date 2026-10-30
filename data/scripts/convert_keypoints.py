import numpy as np

class KeypointAdapter:
    """
    Converts raw landmark objects from browser-side MediaPipe Tasks Vision JS 
    into numpy array tensors formatted for OpenHands pretrained models.
    """
    def __init__(self, target_frames: int = 120, num_landmarks: int = 75):
        self.target_frames = target_frames
        self.num_landmarks = num_landmarks

    def convert_frame(self, frame_data: dict) -> np.ndarray:
        """
        Convert a single frame dictionary containing pose, leftHand, rightHand into a (75, 3) matrix.
        - Pose: 33 landmarks
        - Left Hand: 21 landmarks
        - Right Hand: 21 landmarks
        Total = 75 keypoints × (x, y, z)
        """
        coords = np.zeros((self.num_landmarks, 3), dtype=np.float32)

        # 1. Pose landmarks (indices 0 to 32)
        pose_list = frame_data.get("pose", [])
        for i in range(min(len(pose_list), 33)):
            p = pose_list[i]
            coords[i] = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]

        # 2. Left Hand landmarks (indices 33 to 53)
        left_hand = frame_data.get("leftHand", [])
        for i in range(min(len(left_hand), 21)):
            h = left_hand[i]
            coords[33 + i] = [h.get("x", 0.0), h.get("y", 0.0), h.get("z", 0.0)]

        # 3. Right Hand landmarks (indices 54 to 74)
        right_hand = frame_data.get("rightHand", [])
        for i in range(min(len(right_hand), 21)):
            h = right_hand[i]
            coords[54 + i] = [h.get("x", 0.0), h.get("y", 0.0), h.get("z", 0.0)]

        return coords

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Applies scale & translation invariance to landmark sequence.
        Input sequence shape: (T, 75, 3)
        """
        T, N, C = sequence.shape
        if T == 0:
            return np.zeros((self.target_frames, N, C), dtype=np.float32)

        normalized = sequence.copy()

        for t in range(T):
            # Translate relative to shoulder center (indices 11 and 12 in MediaPipe pose)
            left_shoulder = normalized[t, 11]
            right_shoulder = normalized[t, 12]
            shoulder_center = (left_shoulder + right_shoulder) / 2.0

            normalized[t] -= shoulder_center

            # Scale by shoulder distance
            dist = np.linalg.norm(left_shoulder - right_shoulder)
            if dist > 1e-6:
                normalized[t] /= dist

        # Resample sequence along temporal dimension T to fixed target_frames
        if T != self.target_frames:
            indices = np.linspace(0, T - 1, self.target_frames).astype(int)
            normalized = normalized[indices]

        return normalized

    def process_raw_buffer(self, raw_buffer: list) -> np.ndarray:
        """
        Full processing pipeline: list of frame dicts -> (target_frames, 75, 3) tensor
        """
        frames_coords = [self.convert_frame(f) for f in raw_buffer]
        if not frames_coords:
            return np.zeros((self.target_frames, self.num_landmarks, 3), dtype=np.float32)
            
        seq_tensor = np.array(frames_coords, dtype=np.float32)
        return self.normalize_sequence(seq_tensor)
