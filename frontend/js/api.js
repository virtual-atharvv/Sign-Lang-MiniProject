class SignBridgeAPI {
    constructor() {
        this.backendUrl = window.location.origin; // auto-detect: works both locally and deployed
    }

    async predict(keypointBuffer, mode = "continuous") {
        const url = `${this.backendUrl}/predict`;
        const startTime = performance.now();

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    keypoints: keypointBuffer,
                    mode: mode
                })
            });

            const latency = Math.round(performance.now() - startTime);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            data.clientLatencyMs = latency;
            return data;
        } catch (error) {
            console.warn("API request failed:", error.message);
            return null;
        }
    }
}

window.apiClient = new SignBridgeAPI();
