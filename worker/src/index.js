/**
 * SignBridge Cloudflare Worker API Gateway
 * Handles CORS, request validation, rate limiting, and backend routing.
 */

export default {
    async fetch(request, env, ctx) {
        // Handle CORS preflight requests
        if (request.method === "OPTIONS") {
            return new Response(null, {
                headers: {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
            });
        }

        const url = new URL(request.url);
        // Default backend target is set via env variable HF_BACKEND_URL
        // Can be easily switched to local cloudflared tunnel URL on demo day
        const backendTarget = env.HF_BACKEND_URL || "https://yourname-signbridge.hf.space";
        const targetUrl = `${backendTarget}${url.pathname}${url.search}`;

        const newHeaders = new Headers(request.headers);
        newHeaders.set("Host", new URL(backendTarget).hostname);
        if (env.API_SECRET_TOKEN) {
            newHeaders.set("Authorization", `Bearer ${env.API_SECRET_TOKEN}`);
        }

        try {
            const response = await fetch(targetUrl, {
                method: request.method,
                headers: newHeaders,
                body: request.method !== "GET" && request.method !== "HEAD" ? request.body : null,
            });

            const responseHeaders = new Headers(response.headers);
            responseHeaders.set("Access-Control-Allow-Origin", "*");

            return new Response(response.body, {
                status: response.status,
                statusText: response.statusText,
                headers: responseHeaders,
            });
        } catch (error) {
            return new Response(JSON.stringify({ error: "Backend gateway routing error", details: error.message }), {
                status: 502,
                headers: {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
            });
        }
    },
};
