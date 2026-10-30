document.addEventListener('DOMContentLoaded', async () => {
    const video               = document.getElementById('webcam');
    const canvas              = document.getElementById('output-canvas');
    const cameraPlaceholder   = document.getElementById('camera-placeholder');
    const cameraStatusMsg     = document.getElementById('camera-status-msg');
    const camSpinner          = document.getElementById('cam-spinner');

    const statusIndicator     = document.getElementById('status-indicator');
    const statusText          = document.getElementById('status-text');

    const translatedSentenceEl = document.getElementById('translated-sentence');
    const glossContainer      = document.getElementById('gloss-container');
    const confidenceBadge     = document.getElementById('confidence-badge');

    const metricFps           = document.getElementById('metric-fps');
    const metricLatency       = document.getElementById('metric-latency');

    const btnToggleCam        = document.getElementById('btn-toggle-cam');
    const btnToggleTts        = document.getElementById('btn-toggle-tts');
    const btnSpeakAgain       = document.getElementById('btn-speak-again');
    const modeSelect          = document.getElementById('mode-select');

    let extractor   = null;
    let isPredicting = false;
    let lastSentence = '';
    let ttsEnabled   = true;
    let cameraActive = false;

    // ── Helpers ─────────────────────────────────────────────────────────────
    function setStatus(text, cls) {
        statusIndicator.className = 'status-badge ' + cls;
        statusText.textContent = text;
    }

    // ── Build extractor ──────────────────────────────────────────────────────
    extractor = new window.MediaPipeExtractor(video, canvas);

    extractor.initialize().then(ok => {
        console.log(ok ? '✅ MediaPipe Holistic ready' : '⚠️ MediaPipe not loaded — raw motion mode');
    });

    // ── Camera toggle ────────────────────────────────────────────────────────
    async function handleToggleCamera() {
        if (cameraActive) {
            extractor.stopCamera();
            cameraActive = false;
            cameraPlaceholder.style.display = 'flex';
            if (camSpinner) camSpinner.style.display = 'none';
            cameraStatusMsg.textContent = 'Click "Start Camera" to begin live ISL translation';
            btnToggleCam.innerHTML = '<span class="icon">📷</span> Start Camera';
            setStatus('Standby', 'connecting');
            return;
        }

        btnToggleCam.disabled = true;
        if (camSpinner) camSpinner.style.display = 'block';
        cameraStatusMsg.textContent = 'Requesting camera access…';
        setStatus('Connecting…', 'connecting');

        const ok = await extractor.startCamera();
        btnToggleCam.disabled = false;

        if (ok) {
            cameraActive = true;
            cameraPlaceholder.style.display = 'none';
            btnToggleCam.innerHTML = '<span class="icon">⏹️</span> Stop Camera';
            setStatus('Camera Live', 'ready');
        } else {
            if (camSpinner) camSpinner.style.display = 'none';
            cameraStatusMsg.textContent = '⚠️ Camera access denied or no webcam found. Allow permission and retry.';
            setStatus('Camera Unavailable', 'error');
            btnToggleCam.innerHTML = '<span class="icon">🔄</span> Retry Camera';
        }
    }

    // ── Motion detector (client-side) ────────────────────────────────────────
    // Returns mean frame-to-frame displacement of key landmarks
    function computeBufferMotion(buffer) {
        if (!buffer || buffer.length < 4) return 0;

        let totalMotion = 0;
        let count = 0;
        const recent = buffer.slice(-20);   // look at last 20 frames

        for (let t = 1; t < recent.length; t++) {
            const prev = recent[t - 1];
            const curr = recent[t];
            let frameMotion = 0;
            let kpCount = 0;

            // Use pose landmarks (available even without full MediaPipe)
            const maxL = Math.min(prev.pose.length, curr.pose.length);
            for (let k = 0; k < maxL; k++) {
                const dx = (curr.pose[k].x || 0) - (prev.pose[k].x || 0);
                const dy = (curr.pose[k].y || 0) - (prev.pose[k].y || 0);
                frameMotion += Math.sqrt(dx * dx + dy * dy);
                kpCount++;
            }

            // Weight hand motion more heavily
            const maxLH = Math.min(prev.leftHand.length, curr.leftHand.length);
            for (let k = 0; k < maxLH; k++) {
                const dx = (curr.leftHand[k].x || 0) - (prev.leftHand[k].x || 0);
                const dy = (curr.leftHand[k].y || 0) - (prev.leftHand[k].y || 0);
                frameMotion += Math.sqrt(dx * dx + dy * dy) * 2;
                kpCount += 2;
            }
            const maxRH = Math.min(prev.rightHand.length, curr.rightHand.length);
            for (let k = 0; k < maxRH; k++) {
                const dx = (curr.rightHand[k].x || 0) - (prev.rightHand[k].x || 0);
                const dy = (curr.rightHand[k].y || 0) - (prev.rightHand[k].y || 0);
                frameMotion += Math.sqrt(dx * dx + dy * dy) * 2;
                kpCount += 2;
            }

            if (kpCount > 0) {
                totalMotion += frameMotion / kpCount;
                count++;
            }
        }
        return count > 0 ? totalMotion / count : 0;
    }

    // ── Signing state machine ────────────────────────────────────────────────
    const MOTION_THRESHOLD = 0.0008;  // min motion to count as "signing"
    const IDLE_THRESHOLD   = 0.0004;  // below this → idle
    let signingState = 'idle';       // 'idle' | 'signing' | 'settling'
    let settleFrames = 0;
    const SETTLE_FRAMES = 4;         // frames to wait after motion stops

    function updateSigningState(motionLevel) {
        if (motionLevel > MOTION_THRESHOLD) {
            signingState = 'signing';
            settleFrames = 0;
            setStatus('🤟 Signing…', 'ready');
        } else if (signingState === 'signing') {
            settleFrames++;
            if (settleFrames >= SETTLE_FRAMES) {
                signingState = 'idle';
                setStatus('Camera Live', 'ready');
            }
        }
    }

    // ── Inference polling ────────────────────────────────────────────────────
    async function triggerPrediction() {
        if (!cameraActive || isPredicting) return;
        const buf = extractor.keypointBuffer;
        if (!buf || buf.length < 8) return;

        const motionLevel = computeBufferMotion(buf);
        updateSigningState(motionLevel);

        // Send to backend whenever active or settling
        if (signingState === 'idle' && settleFrames === 0) return;

        isPredicting = true;
        const mode = modeSelect ? modeSelect.value : 'continuous';

        try {
            const response = await window.apiClient.predict(buf.slice(), mode);
            if (response) {
                updateUI(response);
                if (metricLatency) metricLatency.textContent = `${response.clientLatencyMs} ms`;
            }
        } catch (e) {
            console.warn('Prediction error:', e);
        }

        if (metricFps) metricFps.textContent = extractor.fps || '—';
        isPredicting = false;
    }

    // Poll every 200 ms for fast detection (~5 FPS)
    setInterval(triggerPrediction, 200);

    // ── UI updater ───────────────────────────────────────────────────────────
    function updateUI(data) {
        // ── Gloss chips
        if (glossContainer) {
            if (data.glosses && data.glosses.length > 0) {
                // Only re-render if glosses actually changed
                const chipText = data.glosses.join('|');
                if (glossContainer.dataset.lastGlosses !== chipText) {
                    glossContainer.dataset.lastGlosses = chipText;
                    glossContainer.innerHTML = '';
                    data.glosses.forEach(g => {
                        const chip = document.createElement('span');
                        chip.className = 'chip';
                        chip.textContent = g;
                        glossContainer.appendChild(chip);
                    });
                }
            }
        }

        // ── Sentence (only update when backend confirms a new one)
        if (data.sentence && data.sentence.trim() !== '' && data.sentence !== lastSentence) {
            lastSentence = data.sentence;
            translatedSentenceEl.textContent = data.sentence;
            translatedSentenceEl.classList.remove('placeholder-text');

            // Animate the sentence box
            translatedSentenceEl.style.opacity = '0';
            requestAnimationFrame(() => {
                translatedSentenceEl.style.transition = 'opacity 0.4s ease';
                translatedSentenceEl.style.opacity = '1';
            });

            if (ttsEnabled) window.tts.speak(data.sentence);
        }

        // ── Confidence
        if (confidenceBadge && data.confidence > 0) {
            confidenceBadge.textContent = `Conf: ${Math.round(data.confidence * 100)}%`;
        }
    }

    // ── Event listeners ──────────────────────────────────────────────────────
    btnToggleCam.addEventListener('click', handleToggleCamera);

    if (btnToggleTts) {
        btnToggleTts.addEventListener('click', () => {
            ttsEnabled = !ttsEnabled;
            window.tts.setEnabled(ttsEnabled);
            btnToggleTts.innerHTML = ttsEnabled
                ? '<span class="icon">🔊</span> Speech: ON'
                : '<span class="icon">🔇</span> Speech: OFF';
            btnToggleTts.classList.toggle('active', ttsEnabled);
        });
    }

    if (btnSpeakAgain) {
        btnSpeakAgain.addEventListener('click', () => {
            const t = translatedSentenceEl.textContent;
            if (t && !translatedSentenceEl.classList.contains('placeholder-text'))
                window.tts.speak(t);
        });
    }

    // ── Initial state ────────────────────────────────────────────────────────
    if (camSpinner) camSpinner.style.display = 'none';
    cameraStatusMsg.textContent = 'Click "Start Camera" to begin live ISL translation';
    setStatus('Ready', 'ready');
});
