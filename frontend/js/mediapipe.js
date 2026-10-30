/**
 * MediaPipeExtractor — uses legacy @mediapipe/holistic (stable CDN, no ES module).
 * Extracts pose + hand landmarks from webcam and buffers them for inference.
 */
class MediaPipeExtractor {
    constructor(videoElement, canvasElement) {
        this.video       = videoElement;
        this.canvas      = canvasElement;
        this.ctx         = canvasElement.getContext('2d');
        this.holistic    = null;
        this.camera      = null;
        this.isRunning   = false;
        this.keypointBuffer = [];
        this.maxBufferSize  = 120;  // Match PoseUniformSubsampling(num_frames=120) in the model
        this.onFrameProcessed = null;
        this.fps            = 0;
        this._frameCount    = 0;
        this._fpsTimer      = Date.now();
        this.mediapipeReady = false;
        this._lastResults   = null;
    }

    /**
     * Initialise MediaPipe Holistic. Uses window.Holistic exposed by CDN script.
     */
    async initialize() {
        return new Promise((resolve) => {
            // Wait up to 6 s for CDN script to define window.Holistic
            let waited = 0;
            const check = setInterval(() => {
                if (typeof window.Holistic !== 'undefined') {
                    clearInterval(check);
                    this._buildHolistic();
                    resolve(true);
                } else if ((waited += 300) > 6000) {
                    clearInterval(check);
                    console.warn('MediaPipe Holistic CDN timed out — landmark overlay disabled.');
                    resolve(false);
                }
            }, 300);
        });
    }

    _buildHolistic() {
        this.holistic = new window.Holistic({
            locateFile: (file) =>
                `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });

        this.holistic.setOptions({
            modelComplexity:          1,
            smoothLandmarks:          true,
            enableSegmentation:       false,
            smoothSegmentation:       false,
            refineFaceLandmarks:      false,
            minDetectionConfidence:   0.5,
            minTrackingConfidence:    0.5
        });

        this.holistic.onResults((results) => {
            this._lastResults = results;
            this._drawResults(results);
            this._extractFrameKeypoints(results);
        });

        this.mediapipeReady = true;
        console.log('✅ MediaPipe Holistic ready.');
    }

    /**
     * Start webcam stream and MediaPipe camera utility loop.
     */
    async startCamera() {
        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                console.error('getUserMedia not supported.');
                return false;
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: { ideal: 30 } },
                audio: false
            });
            this.video.srcObject = stream;

            await new Promise((resolve, reject) => {
                this.video.onloadedmetadata = () =>
                    this.video.play().then(resolve).catch(reject);
                this.video.onerror = reject;
                setTimeout(() => reject(new Error('Video load timeout')), 8000);
            });

            this.canvas.width  = this.video.videoWidth  || 640;
            this.canvas.height = this.video.videoHeight || 480;
            this.isRunning = true;

            if (this.mediapipeReady && this.holistic) {
                // Use MediaPipe Camera utility — it drives the holistic model each frame
                this.camera = new window.Camera(this.video, {
                    onFrame: async () => {
                        await this.holistic.send({ image: this.video });
                    },
                    width:  640,
                    height: 480
                });
                this.camera.start();
            } else {
                // Fallback: manual RAF loop; push empty frames so buffer fills
                this._fallbackLoop();
            }

            console.log(`📷 Camera started at ${this.video.videoWidth}×${this.video.videoHeight}`);
            return true;

        } catch (err) {
            console.error('Camera error:', err.name, '-', err.message);
            if (this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(t => t.stop());
                this.video.srcObject = null;
            }
            return false;
        }
    }

    _fallbackLoop() {
        if (!this.isRunning) return;
        this._extractFrameKeypoints({ poseLandmarks: null, leftHandLandmarks: null, rightHandLandmarks: null });
        requestAnimationFrame(() => this._fallbackLoop());
    }

    stopCamera() {
        this.isRunning = false;
        if (this.camera) { this.camera.stop(); this.camera = null; }
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }
        this.keypointBuffer = [];
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // ── Keypoint extraction ──────────────────────────────────────────────────

    _extractFrameKeypoints(results) {
        // FPS counter
        this._frameCount++;
        const now = Date.now();
        if (now - this._fpsTimer >= 1000) {
            this.fps = this._frameCount;
            this._frameCount = 0;
            this._fpsTimer = now;
        }

        const lmToObj = (lm) => ({ x: lm.x, y: lm.y, z: lm.z ?? 0 });

        const frameData = {
            pose:      results.poseLandmarks       ? results.poseLandmarks.map(lmToObj)      : [],
            leftHand:  results.leftHandLandmarks   ? results.leftHandLandmarks.map(lmToObj)  : [],
            rightHand: results.rightHandLandmarks  ? results.rightHandLandmarks.map(lmToObj) : []
        };

        this.keypointBuffer.push(frameData);
        if (this.keypointBuffer.length > this.maxBufferSize)
            this.keypointBuffer.shift();

        if (this.onFrameProcessed) this.onFrameProcessed(this.keypointBuffer);
    }

    // ── Drawing ──────────────────────────────────────────────────────────────

    _drawResults(results) {
        this.ctx.save();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Mirror-flip canvas to match video
        this.ctx.translate(this.canvas.width, 0);
        this.ctx.scale(-1, 1);

        // Draw connections if drawingUtils is available
        if (window.drawConnectors && window.POSE_CONNECTIONS) {
            if (results.poseLandmarks) {
                window.drawConnectors(this.ctx, results.poseLandmarks, window.POSE_CONNECTIONS,
                    { color: '#6366f1', lineWidth: 2 });
                window.drawLandmarks(this.ctx, results.poseLandmarks,
                    { color: '#818cf8', lineWidth: 1, radius: 3 });
            }
            if (results.leftHandLandmarks) {
                window.drawConnectors(this.ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS,
                    { color: '#10b981', lineWidth: 2 });
                window.drawLandmarks(this.ctx, results.leftHandLandmarks,
                    { color: '#34d399', lineWidth: 1, radius: 3 });
            }
            if (results.rightHandLandmarks) {
                window.drawConnectors(this.ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS,
                    { color: '#f59e0b', lineWidth: 2 });
                window.drawLandmarks(this.ctx, results.rightHandLandmarks,
                    { color: '#fbbf24', lineWidth: 1, radius: 3 });
            }
        } else {
            // Minimal fallback dots
            this._dots(results.poseLandmarks, '#6366f1', 3);
            this._dots(results.leftHandLandmarks, '#10b981', 4);
            this._dots(results.rightHandLandmarks, '#f59e0b', 4);
        }

        this.ctx.restore();
    }

    _dots(lms, color, r) {
        if (!lms) return;
        this.ctx.fillStyle = color;
        for (const lm of lms) {
            this.ctx.beginPath();
            this.ctx.arc(lm.x * this.canvas.width, lm.y * this.canvas.height, r, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
}

window.MediaPipeExtractor = MediaPipeExtractor;
