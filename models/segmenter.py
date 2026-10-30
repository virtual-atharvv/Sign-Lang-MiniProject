import numpy as np

class MotionSegmenter:
    """
    Splits continuous stream of landmark frames into isolated sign clips 
    based on motion energy drop (pause detection) and sliding window fallback.
    """
    def __init__(self, motion_threshold: float = 0.015, pause_frames: int = 5, window_size: int = 64, stride: int = 32):
        self.motion_threshold = motion_threshold
        self.pause_frames = pause_frames
        self.window_size = window_size
        self.stride = stride

    def compute_motion_energy(self, sequence: np.ndarray) -> np.ndarray:
        """
        Computes frame-to-frame L2 norm distance across keypoints.
        Input shape: (T, 75, 3)
        Output shape: (T-1,)
        """
        if len(sequence) < 2:
            return np.array([0.0])

        diffs = np.diff(sequence, axis=0)
        # Average movement per keypoint per frame
        motion_per_frame = np.mean(np.linalg.norm(diffs, axis=2), axis=1)
        return motion_per_frame

    def segment_sequence(self, sequence: np.ndarray) -> list[tuple[int, int]]:
        """
        Detects start and end frame indices for individual signs.
        Returns list of (start_idx, end_idx) tuples.
        """
        T = len(sequence)
        if T <= self.window_size:
            return [(0, T)]

        motion_energy = self.compute_motion_energy(sequence)
        segments = []

        is_moving = False
        start_idx = 0
        pause_count = 0

        for i, energy in enumerate(motion_energy):
            if energy >= self.motion_threshold:
                if not is_moving:
                    is_moving = True
                    start_idx = i
                pause_count = 0
            else:
                if is_moving:
                    pause_count += 1
                    if pause_count >= self.pause_frames:
                        is_moving = False
                        end_idx = i - pause_count + 1
                        if end_idx - start_idx >= 10:  # Minimum sign duration constraint
                            segments.append((start_idx, end_idx))
                        pause_count = 0

        if is_moving and (T - start_idx >= 10):
            segments.append((start_idx, T))

        # Fallback to sliding window if motion segmentation yields no segments
        if not segments:
            for start in range(0, T - self.window_size + 1, self.stride):
                segments.append((start, start + self.window_size))

        return segments
