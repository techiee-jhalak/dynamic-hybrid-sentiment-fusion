/**
 * Dynamic Hybrid Sentiment Fusion
 * Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Configuration
    // Allows overriding via global window object or fallback to localhost
    const API_BASE_URL = window.APP_CONFIG?.API_BASE_URL || 'http://localhost:8000/api';

    // DOM Elements
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const resultsDashboard = document.getElementById('results-dashboard');

    // UI Updates - Input
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCount.textContent = `${len} / 2048`;
        if (len > 2048) {
            charCount.style.color = 'var(--danger-color)';
        } else {
            charCount.style.color = 'var(--text-secondary)';
        }
    });

    // Action: Clear
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        textInput.dispatchEvent(new Event('input'));
        
        hideError();
        resultsDashboard.classList.add('hidden');
        textInput.focus();
    });

    // Action: Analyze
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        
        if (!text) {
            showError("Please enter some text to analyze.");
            return;
        }

        if (text.length > 2048) {
            showError("Text exceeds the 2048 character limit.");
            return;
        }

        // Prepare UI for loading
        hideError();
        resultsDashboard.classList.add('hidden');
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle structured error from API
                const errorMsg = data.detail || "An unexpected error occurred.";
                throw new Error(errorMsg);
            }

            // Success - Populate Dashboard
            populateDashboard(data);
            
            // Show Dashboard
            resultsDashboard.classList.remove('hidden');

        } catch (err) {
            console.error("API Error:", err);
            // Don't expose raw Fetch stack traces, just message
            showError(err.message === 'Failed to fetch' 
                ? "Network error: Cannot connect to the API server." 
                : err.message);
        } finally {
            setLoading(false);
        }
    });

    // DOM Update Functions
    function setLoading(isLoading) {
        if (isLoading) {
            loadingState.classList.remove('hidden');
            analyzeBtn.disabled = true;
            clearBtn.disabled = true;
            textInput.disabled = true;
        } else {
            loadingState.classList.add('hidden');
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
            textInput.disabled = false;
        }
    }

    function showError(message) {
        errorState.textContent = message;
        errorState.classList.remove('hidden');
    }

    function hideError() {
        errorState.classList.add('hidden');
        errorState.textContent = '';
    }

    function populateDashboard(data) {
        // 1. Core Result
        const sentimentBadge = document.getElementById('sentiment-badge');
        sentimentBadge.textContent = data.sentiment;
        sentimentBadge.className = `badge ${data.sentiment.toLowerCase()}`;
        
        document.getElementById('final-score').textContent = formatNumber(data.final_score, 3);

        // 2. Dynamic Fusion Details
        document.getElementById('vader-score').textContent = formatNumber(data.vader_score, 3);
        document.getElementById('distilbert-score').textContent = formatNumber(data.distilbert_score, 3);
        document.getElementById('alpha-value').textContent = formatNumber(data.alpha, 3);

        // 3. Noise Analysis
        const noise = data.noise_features;
        
        // Noise Band Badge
        const bandBadge = document.getElementById('noise-band');
        bandBadge.textContent = noise.noise_band;
        bandBadge.className = `badge noise-badge ${noise.noise_band.toLowerCase()}`;
        
        // Noise Composite Progress Bar
        const composite = data.noise_score;
        document.getElementById('composite-noise').textContent = formatNumber(composite, 3);
        document.getElementById('noise-progress').style.width = `${Math.min(100, composite * 100)}%`;
        
        // Adjust progress bar color based on band
        const progressFill = document.getElementById('noise-progress');
        if (noise.noise_band === 'LOW') progressFill.style.backgroundColor = 'var(--success-color)';
        else if (noise.noise_band === 'MODERATE') progressFill.style.backgroundColor = 'var(--warning-color)';
        else progressFill.style.backgroundColor = 'var(--danger-color)';

        // Noise Components
        document.getElementById('emoji-density').textContent = formatNumber(noise.emoji_density, 3);
        document.getElementById('repetition-ratio').textContent = formatNumber(noise.repetition_ratio, 3);
        document.getElementById('codemix-intensity').textContent = formatNumber(noise.codemix_intensity, 3);
        document.getElementById('symbol-density').textContent = formatNumber(noise.symbol_density, 3);

        // 4. Router Explanation
        document.getElementById('router-text').textContent = data.router.explanation;

        // 5. Technical Details
        document.getElementById('token-length').textContent = data.token_length;
    }

    function formatNumber(num, decimals) {
        if (num === null || num === undefined) return "0.000";
        return Number(num).toFixed(decimals);
    }
});
