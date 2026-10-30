class TTSController {
    constructor() {
        this.synth = window.speechSynthesis;
        this.enabled = true;
        this.lastSpoken = "";
    }

    speak(text) {
        if (!this.enabled || !text || text === this.lastSpoken) return;
        if (!this.synth) return;

        if (this.synth.speaking) {
            this.synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;

        this.synth.speak(utterance);
        this.lastSpoken = text;
    }

    setEnabled(status) {
        this.enabled = status;
        if (!status && this.synth && this.synth.speaking) {
            this.synth.cancel();
        }
    }
}

window.tts = new TTSController();
