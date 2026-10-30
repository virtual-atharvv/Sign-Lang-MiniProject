import time
import numpy as np
from collections import deque, Counter
from data.scripts.convert_keypoints import KeypointAdapter
from models.segmenter import MotionSegmenter
from models.sign_recognition.openhands_wrapper import OpenHandsSignRecognizer
from models.grammar_correction.inference import ISLGrammarCorrector


class PredictionSmoother:
    """
    Majority-vote buffer that stabilises predictions over time.

    A new gloss is only "confirmed" and emitted when it wins the vote
    across `vote_window` consecutive predictions AND stays the winner
    for `hold_frames` polls without changing.
    """

    def __init__(self, vote_window: int = 5, hold_frames: int = 3, cooldown_sec: float = 2.0):
        self.vote_window   = vote_window
        self.hold_frames   = hold_frames
        self.cooldown_sec  = cooldown_sec

        self._buffer: deque  = deque(maxlen=vote_window)
        self._confirmed      = None      # last confirmed gloss
        self._confirmed_conf = 0.0
        self._hold_count     = 0
        self._last_emit_time = 0.0
        self._confirmed_glosses: list = []  # rolling sentence accumulator

    def push(self, gloss: str, conf: float) -> dict | None:
        """
        Push a raw prediction. Returns a confirmed result dict or None.
        """
        self._buffer.append((gloss, conf))

        if len(self._buffer) < self.vote_window:
            return None   # not enough votes yet

        # Majority vote
        counter = Counter(g for g, _ in self._buffer)
        winner, win_count = counter.most_common(1)[0]
        mean_conf = float(np.mean([c for g, c in self._buffer if g == winner]))

        # Must win ≥ 60 % of the vote window
        if win_count / self.vote_window < 0.6:
            return None

        # Accumulate hold frames on the same winner
        if winner == self._confirmed:
            self._hold_count += 1
        else:
            self._confirmed      = winner
            self._confirmed_conf = mean_conf
            self._hold_count     = 1

        # Emit only when hold criterion met AND cooldown elapsed
        now = time.time()
        if self._hold_count >= self.hold_frames and (now - self._last_emit_time) >= self.cooldown_sec:
            self._last_emit_time = now
            self._hold_count = 0       # reset so same gloss doesn't keep emitting
            return {"gloss": winner, "confidence": round(mean_conf, 4)}

        return None


class TranslationPipeline:
    """
    Integrates keypoint adaptation, motion segmentation, sign recognition,
    grammar correction, and temporal smoothing into a single inference pipeline.
    """

    # Number of frames the keypoint buffer must contain before we try to predict
    MIN_FRAMES = 8

    # Minimum mean motion energy to bother predicting (avoids idle noise)
    MOTION_GATE = 0.005

    def __init__(self, config_path: str = "data/checkpoints/include/st_gcn/config.yaml"):
        self.adapter          = KeypointAdapter()
        self.segmenter        = MotionSegmenter(
            motion_threshold=0.01,   # lower threshold for quick detection
            pause_frames=5,          # faster sign segment boundary detection
        )
        self.recognizer       = OpenHandsSignRecognizer(config_path=config_path)
        self.grammar_corrector = ISLGrammarCorrector()
        self.smoother         = PredictionSmoother(
            vote_window=3,      # votes over last 3 predictions
            hold_frames=1,      # emit quickly when winner confirmed
            cooldown_sec=0.4    # fast updates between new signs
        )
        # Rolling gloss history for sentence building
        self._gloss_history: list[str] = []
        self._last_sentence: str = ""

    def process(self, raw_frames: list, mode: str = "continuous") -> dict:
        t0 = time.time()

        empty = {
            "glosses":            [],
            "sentence":           "",
            "confidence":         0.0,
            "processing_time_ms": 0.0
        }

        if not raw_frames or len(raw_frames) < self.MIN_FRAMES:
            return empty

        # 1. Convert to numpy  (T, 75, 3)
        raw_dicts  = [f.dict() if hasattr(f, "dict") else f for f in raw_frames]
        seq_matrix = np.array([self.adapter.convert_frame(f) for f in raw_dicts])

        # 2. Motion gate — skip prediction if hands are idle
        motion_energies = self.segmenter.compute_motion_energy(seq_matrix)
        mean_motion     = float(np.mean(motion_energies))

        if mean_motion < self.MOTION_GATE:
            # Return whatever is already confirmed (don't blank the UI)
            return {
                "glosses":            self._gloss_history[-3:] if self._gloss_history else [],
                "sentence":           self._last_sentence,
                "confidence":         0.0,
                "processing_time_ms": round((time.time() - t0) * 1000, 2)
            }

        # 3. Segment the sequence
        if mode == "continuous":
            segments = self.segmenter.segment_sequence(seq_matrix)
            if not segments:
                segments = [(max(0, len(seq_matrix) - 120), len(seq_matrix))]
        else:
            segments = [(0, len(seq_matrix))]

        # 4. Recognise each segment, push through smoother
        confirmed_result = None
        raw_conf         = 0.0

        for start, end in segments[-1:]:
            clip = seq_matrix[start:end]
            if len(clip) < 8:
                continue

            # Pass the raw clip directly — openhands_wrapper._neural_predict owns
            # the pixel-scaling, PoseSelect, PoseUniformSubsampling(120), and
            # CenterAndScaleNormalize transform chain. Calling normalize_sequence
            # here would double-normalize and corrupt the scale heuristic.

            res  = self.recognizer.predict(clip)

            gloss = res.get("gloss", "UNKNOWN")
            conf  = res.get("confidence", 0.0)
            raw_conf = conf

            if gloss in ("UNKNOWN", "STANDBY") or conf < 0.15:
                continue

            confirmed_result = self.smoother.push(gloss, conf)

        # 5. If smoother confirms a new gloss, update history & rebuild sentence
        if confirmed_result:
            g = confirmed_result["gloss"]
            if not self._gloss_history or self._gloss_history[-1] != g:
                self._gloss_history.append(g)
                if len(self._gloss_history) > 6:
                    self._gloss_history = self._gloss_history[-6:]

            self._last_sentence = self.grammar_corrector.translate(self._gloss_history)

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return {
            "glosses":            list(self._gloss_history[-3:]),
            "sentence":           self._last_sentence,
            "confidence":         round(raw_conf, 4),
            "processing_time_ms": elapsed_ms
        }
