/**
 * Dynamic Hybrid Sentiment Fusion
 * Frontend Application Logic — SentiMix 3-Class Production Pipeline
 */

document.addEventListener('DOMContentLoaded', () => {
    // Configuration
    // Allows overriding via global window object or fallback to origin / localhost
    const API_BASE_URL = window.APP_CONFIG?.API_BASE_URL || (
        window.location.origin.startsWith('http')
            ? window.location.origin + '/api'
            : 'http://localhost:8000/api'
    );

    // DOM Elements
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const resultsDashboard = document.getElementById('results-dashboard');

    // UI Updates - Input character counter
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
                const errorMsg = data.detail || (Array.isArray(data) ? data[0]?.msg : "An unexpected error occurred.");
                throw new Error(errorMsg);
            }

            // Success - Populate Dashboard
            populateDashboard(data);
            
            // Show Dashboard
            resultsDashboard.classList.remove('hidden');

        } catch (err) {
            console.error("API Error:", err);
            showError(err.message === 'Failed to fetch' 
                ? "Network error: Cannot connect to the API server. Please check that the server is running." 
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

    function setVectorBars(prefix, vector) {
        if (!vector) return;
        const pos = Number(vector.positive || 0);
        const neg = Number(vector.negative || 0);
        const neu = Number(vector.neutral || 0);

        const posPercent = (pos * 100).toFixed(1) + '%';
        const negPercent = (neg * 100).toFixed(1) + '%';
        const neuPercent = (neu * 100).toFixed(1) + '%';

        const posElem = document.getElementById(`${prefix}-pos`);
        const negElem = document.getElementById(`${prefix}-neg`);
        const neuElem = document.getElementById(`${prefix}-neu`);

        if (posElem) posElem.textContent = posPercent;
        if (negElem) negElem.textContent = negPercent;
        if (neuElem) neuElem.textContent = neuPercent;

        const posBar = document.getElementById(`bar-${prefix}-pos`);
        const negBar = document.getElementById(`bar-${prefix}-neg`);
        const neuBar = document.getElementById(`bar-${prefix}-neu`);

        if (posBar) posBar.style.width = `${Math.min(100, pos * 100)}%`;
        if (negBar) negBar.style.width = `${Math.min(100, neg * 100)}%`;
        if (neuBar) neuBar.style.width = `${Math.min(100, neu * 100)}%`;
    }

    function populateDashboard(data) {
        // 1. Core Sentiment Result
        const sentimentBadge = document.getElementById('sentiment-badge');
        const rawLabel = (data.predicted_label || data.sentiment || 'neutral').toLowerCase();
        sentimentBadge.textContent = rawLabel.toUpperCase();
        sentimentBadge.className = `badge ${rawLabel}`;
        
        // Confidence
        const confidence = data.confidence !== undefined ? data.confidence : (data.final_score || 0);
        const confPercent = (Number(confidence) * 100).toFixed(1) + '%';
        document.getElementById('confidence-value').textContent = confPercent;

        // 2. Model Contributions (3-Class Probability Distributions)
        setVectorBars('fused', data.fused_vector);
        setVectorBars('distil', data.distilbert_vector);
        setVectorBars('vader', data.vader_vector);

        // Dynamic Weights
        const alpha = Number(data.alpha !== undefined ? data.alpha : 0.02);
        const bertWeight = 1.0 - alpha;

        const alphaElem = document.getElementById('alpha-value');
        if (alphaElem) alphaElem.textContent = alpha.toFixed(3);

        const vaderContrib = document.getElementById('vader-contrib');
        if (vaderContrib) vaderContrib.textContent = (alpha * 100).toFixed(1) + '%';

        const distilContrib = document.getElementById('distil-contrib');
        if (distilContrib) distilContrib.textContent = (bertWeight * 100).toFixed(1) + '%';

        const vaderWeightLabel = document.getElementById('vader-weight-label');
        if (vaderWeightLabel) vaderWeightLabel.textContent = `Weight: α = ${alpha.toFixed(3)}`;

        const distilWeightLabel = document.getElementById('distilbert-weight-label');
        if (distilWeightLabel) distilWeightLabel.textContent = `Weight: 1−α = ${bertWeight.toFixed(3)}`;

        // 3. Noise Analysis
        const noise = data.noise || data.noise_features || {};
        const band = (noise.band || noise.noise_band || 'LOW').toUpperCase();
        
        // Noise Band Badge
        const bandBadge = document.getElementById('noise-band');
        bandBadge.textContent = band;
        bandBadge.className = `badge noise-badge ${band.toLowerCase()}`;
        
        // Noise Composite Progress Bar
        const composite = Number(noise.composite_noise !== undefined ? noise.composite_noise : (data.noise_score || 0));
        document.getElementById('composite-noise').textContent = formatNumber(composite, 3);
        
        const progressFill = document.getElementById('noise-progress');
        progressFill.style.width = `${Math.min(100, composite * 100)}%`;
        if (band === 'LOW') progressFill.style.backgroundColor = 'var(--success-color)';
        else if (band === 'MODERATE') progressFill.style.backgroundColor = 'var(--warning-color)';
        else progressFill.style.backgroundColor = 'var(--danger-color)';

        // Noise Components
        document.getElementById('emoji-density').textContent = formatNumber(noise.emoji_density, 3);
        document.getElementById('repetition-ratio').textContent = formatNumber(noise.repetition_score ?? noise.repetition_ratio, 3);
        document.getElementById('codemix-intensity').textContent = formatNumber(noise.code_mixing_ratio ?? noise.codemix_intensity, 3);
        document.getElementById('symbol-density').textContent = formatNumber(noise.symbol_density, 3);

        // 4. Router Explanation
        document.getElementById('router-text').textContent = data.explanation || data.router?.explanation || '--';

        // 5. Technical Details
        document.getElementById('token-length').textContent = data.token_length ?? '--';
    }

    function formatNumber(num, decimals) {
        if (num === null || num === undefined) return "0.000";
        return Number(num).toFixed(decimals);
    }
});
