/**
 * Dynamic Hybrid Sentiment Fusion
 * Dreamy Winter AI — Frontend Application
 * Snow canvas system · Multi-depth particles · Live API integration · Observatory tabs
 */

/* ═══════════════════════════════════════════════════
   1. DIGITAL SNOW & SPARKLE CANVAS SYSTEM
═══════════════════════════════════════════════════ */
(function initSnowSystem() {
  const canvas = document.getElementById('snow-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Honor reduced motion
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    canvas.style.display = 'none';
    return;
  }

  const isMobile = () => window.innerWidth <= 768;

  function getParticleCounts() {
    const mobile = isMobile();
    return {
      bg: mobile ? 25 : 55,   // Background: tiny, slow, low opacity
      mid: mobile ? 14 : 32,  // Midground: medium, soft drift
      fg: mobile ? 6 : 14     // Foreground: larger, glowing stars/sparkles
    };
  }

  let W = 0, H = 0;
  let particles = [];
  let raf = null;

  class Particle {
    constructor(layer) {
      this.layer = layer; // 'bg' | 'mid' | 'fg'
      this.reset(true);
    }

    reset(initial = false) {
      const xRange = W || window.innerWidth;
      const yRange = H || window.innerHeight;

      this.x = Math.random() * xRange;
      this.y = initial ? Math.random() * yRange : -12;

      if (this.layer === 'bg') {
        this.r     = 0.5 + Math.random() * 0.9;
        this.vx    = (Math.random() - 0.5) * 0.14;
        this.vy    = 0.12 + Math.random() * 0.20;
        this.base  = 0.10 + Math.random() * 0.16;
        this.isStar = false;
        this.sparkle = false;
      } else if (this.layer === 'mid') {
        this.r     = 1.1 + Math.random() * 1.3;
        this.vx    = (Math.random() - 0.5) * 0.22;
        this.vy    = 0.24 + Math.random() * 0.32;
        this.base  = 0.18 + Math.random() * 0.22;
        this.isStar = Math.random() < 0.15;
        this.sparkle = Math.random() < 0.25;
      } else { // 'fg'
        this.r     = 1.6 + Math.random() * 1.8;
        this.vx    = (Math.random() - 0.5) * 0.32;
        this.vy    = 0.42 + Math.random() * 0.55;
        this.base  = 0.25 + Math.random() * 0.30;
        this.isStar = Math.random() < 0.50; // 50% render as delicate ✦ sparkle
        this.sparkle = Math.random() < 0.65;
      }

      // Horizontal wave drift
      this.waveOffset = Math.random() * Math.PI * 2;
      this.waveFreq   = 0.003 + Math.random() * 0.005;
      this.waveAmp    = 0.20 + Math.random() * 0.35;

      // Sparkle pulse timing (opacity 0.2 -> 0.9 -> 0.2)
      this.sparklePeriod = 1400 + Math.random() * 2200;
      this.sparklePhase  = Math.random() * this.sparklePeriod;
      this.opacity       = this.base;
    }

    update(t, dt) {
      this.x += this.vx + Math.sin(t * this.waveFreq + this.waveOffset) * this.waveAmp * (dt * 0.06);
      this.y += this.vy * (dt * 0.06);

      // Smooth sparkle breathing
      if (this.sparkle) {
        const phase = ((t + this.sparklePhase) % this.sparklePeriod) / this.sparklePeriod;
        const breath = Math.max(0, Math.sin(phase * Math.PI * 2));
        this.opacity = this.base + breath * (this.layer === 'fg' ? 0.65 : 0.35);
      }

      // Screen wrapping
      if (this.x < -14) this.x = W + 14;
      if (this.x > W + 14) this.x = -14;

      // Reset when falling beyond screen
      if (this.y > H + 18) this.reset(false);
    }

    draw(ctx) {
      ctx.save();
      ctx.globalAlpha = Math.min(1, Math.max(0, this.opacity));

      if (this.isStar) {
        // Star sparkle ✦
        ctx.fillStyle = this.opacity > 0.55 ? '#E9EEFF' : '#8FB8FF';
        ctx.font = `${this.r * 2.6}px serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (this.opacity > 0.45) {
          ctx.shadowColor = '#A99AF5';
          ctx.shadowBlur  = this.r * 4;
        }
        ctx.fillText('✦', this.x, this.y);
      } else {
        // Soft glowing snow dot
        const grad = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.r * 2);
        grad.addColorStop(0, '#FFFFFF');
        grad.addColorStop(0.5, '#A8C4FF');
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        if (this.layer === 'fg' && this.opacity > 0.35) {
          ctx.shadowColor = '#8FB8FF';
          ctx.shadowBlur  = this.r * 3;
        }
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }
  }

  function initParticles() {
    particles = [];
    const counts = getParticleCounts();
    for (let i = 0; i < counts.bg;  i++) particles.push(new Particle('bg'));
    for (let i = 0; i < counts.mid; i++) particles.push(new Particle('mid'));
    for (let i = 0; i < counts.fg;  i++) particles.push(new Particle('fg'));
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    initParticles();
  }

  let lastTime = 0;
  function loop(t) {
    const dt = Math.min(t - lastTime, 50);
    lastTime = t;

    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < particles.length; i++) {
      particles[i].update(t, dt);
      particles[i].draw(ctx);
    }
    raf = requestAnimationFrame(loop);
  }

  window.addEventListener('resize', resize, { passive: true });
  resize();
  raf = requestAnimationFrame(loop);

  // ── Sparkle burst on AI completion ──
  window.triggerSparkBurst = function(cx, cy) {
    if (prefersReduced) return;
    const burstParticles = [];
    const count = 10;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
      const speed = 1.5 + Math.random() * 2.8;
      burstParticles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        r: 2.0 + Math.random() * 2.0,
        life: 1.0,
        decay: 0.020 + Math.random() * 0.015
      });
    }

    let start = null;
    function animBurst(t) {
      if (!start) start = t;
      let alive = false;
      ctx.save();
      for (let i = 0; i < burstParticles.length; i++) {
        const p = burstParticles[i];
        if (p.life <= 0) continue;
        alive = true;
        p.x += p.vx;
        p.y += p.vy;
        p.life -= p.decay;

        ctx.globalAlpha = Math.max(0, p.life * 0.9);
        ctx.fillStyle = '#E9EEFF';
        ctx.font = `${p.r * 2.5}px serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = '#A99AF5';
        ctx.shadowBlur = p.r * 5;
        ctx.fillText('✦', p.x, p.y);
      }
      ctx.restore();
      if (alive) requestAnimationFrame(animBurst);
    }
    requestAnimationFrame(animBurst);
  };
})();

/* ═══════════════════════════════════════════════════
   2. MAIN APPLICATION LOGIC
═══════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  // Resolve API Base endpoint
  const API_BASE = window.APP_CONFIG?.API_BASE_URL || '/api';

  // DOM Elements
  const textInput       = document.getElementById('text-input');
  const charCount       = document.getElementById('char-count');
  const analyzeBtn      = document.getElementById('analyze-btn');
  const clearBtn        = document.getElementById('clear-btn');
  const loadingState    = document.getElementById('loading-state');
  const loadingLabel    = document.getElementById('loading-label');
  const errorState      = document.getElementById('error-state');
  const resultsSection  = document.getElementById('results');
  const aiCore          = document.getElementById('ai-core');
  const sampleChips     = document.querySelectorAll('.chip-btn');
  const tabButtons      = document.querySelectorAll('.obs-tab-btn');
  const tabPanels       = document.querySelectorAll('.tab-panel');

  // ── Character Counter ──
  textInput.addEventListener('input', () => {
    const len = textInput.value.length;
    charCount.textContent = `${len} / 2048`;
    charCount.style.color = len > 2048 ? 'var(--neg)' : 'var(--t3)';
  });

  // ── Sample Prompt Chips ──
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-text');
      if (text) {
        textInput.value = text;
        textInput.dispatchEvent(new Event('input'));
        textInput.focus();
      }
    });
  });

  // ── Clear Action ──
  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    textInput.dispatchEvent(new Event('input'));
    hideError();
    resultsSection.classList.add('hidden');
    textInput.focus();
  });

  // ── Observatory Tab Switching ──
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });

  // ── Loading Sequence with Pipeline Stages ──
  const PIPELINE_STAGES = [
    'Analyzing signal…',
    'Reading language…',
    'Measuring noise field…',
    'Routing adaptive signals…',
    'Fusing predictions…'
  ];
  let loadInterval = null;

  function startLoading() {
    setFormBusy(true);
    loadingState.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    hideError();

    let stageIdx = 0;
    if (loadingLabel) loadingLabel.textContent = PIPELINE_STAGES[0];
    loadInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % PIPELINE_STAGES.length;
      if (loadingLabel) loadingLabel.textContent = PIPELINE_STAGES[stageIdx];
    }, 850);

    if (aiCore) aiCore.classList.add('loading-active');
  }

  function stopLoading() {
    clearInterval(loadInterval);
    loadInterval = null;
    loadingState.classList.add('hidden');
    if (aiCore) aiCore.classList.remove('loading-active');
    setFormBusy(false);
  }

  function setFormBusy(busy) {
    analyzeBtn.disabled = busy;
    clearBtn.disabled   = busy;
    textInput.disabled  = busy;
  }

  // ── Error Messaging ──
  function showError(msg) {
    errorState.textContent = msg;
    errorState.classList.remove('hidden');
  }
  function hideError() {
    errorState.classList.add('hidden');
    errorState.textContent = '';
  }

  // ── Analyze Submission ──
  analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
      showError('Please enter some text to analyze.');
      textInput.focus();
      return;
    }
    if (text.length > 2048) {
      showError('Text length exceeds the 2048-character limit.');
      return;
    }

    startLoading();

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      if (!res.ok) {
        const errorMsg = data.detail || (Array.isArray(data) ? data[0]?.msg : 'Analysis failed.');
        throw new Error(errorMsg);
      }

      // Populate interface with real API data
      populateDashboard(data);

      // Unhide results
      resultsSection.classList.remove('hidden');

      // Trigger sparkle burst around AI Core
      if (window.triggerSparkBurst && aiCore) {
        const rect = aiCore.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        window.triggerSparkBurst(cx, cy);
      }

      // Smoothly scroll down to results
      const resultsTop = resultsSection.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({ top: resultsTop, behavior: 'smooth' });

    } catch (err) {
      console.error('API Error:', err);
      showError(
        err.message === 'Failed to fetch'
          ? 'Network error: unable to connect to the backend server. Please check your connection or service status.'
          : err.message
      );
    } finally {
      stopLoading();
    }
  });

  // ── Populate Dashboard with Real Output ──
  function populateDashboard(data) {
    // 1. Predicted Label & Sentiment Badge
    const rawLabel = (data.predicted_label || data.sentiment || 'neutral').toLowerCase();
    const badge = document.getElementById('sentiment-badge');
    if (badge) {
      badge.textContent = rawLabel.toUpperCase();
      badge.className = `sentiment-badge ${rawLabel}`;
    }

    // 2. Confidence & Confidence Ring
    const conf = Number(data.confidence ?? data.final_score ?? 0);
    const confPct = (conf * 100).toFixed(1) + '%';
    setElementText('confidence-pct', confPct);

    const qualityBadge = document.getElementById('confidence-quality');
    if (qualityBadge) {
      qualityBadge.textContent =
        conf >= 0.75 ? 'High Confidence (Robust)' :
        conf >= 0.50 ? 'Moderate Confidence' :
        'Uncertain Signal';
    }

    // SVG Ring progress: circumference = 2 * PI * 58 ≈ 364.42
    const CIRC = 364.42;
    const ringFill = document.getElementById('ring-fill');
    if (ringFill) {
      const offset = CIRC - (conf * CIRC);
      ringFill.style.strokeDashoffset = Math.max(0, offset);

      const colorMap = {
        positive: '#6EDCB4',
        negative: '#F4907A',
        neutral:  '#93B4F5'
      };
      ringFill.style.stroke = colorMap[rawLabel] || '#A99AF5';
    }

    // 3. Fused Signal Distribution Streams
    const fv = data.fused_vector || {};
    setStreamBar('fused-pos', fv.positive ?? 0);
    setStreamBar('fused-neg', fv.negative ?? 0);
    setStreamBar('fused-neu', fv.neutral  ?? 0);

    // Individual Sub-Vectors
    const dv = data.distilbert_vector || {};
    setSubBar('distil-pos', dv.positive ?? 0);
    setSubBar('distil-neg', dv.negative ?? 0);
    setSubBar('distil-neu', dv.neutral  ?? 0);

    const vv = data.vader_vector || {};
    setSubBar('vader-pos', vv.positive ?? 0);
    setSubBar('vader-neg', vv.negative ?? 0);
    setSubBar('vader-neu', vv.neutral  ?? 0);

    // 4. Noise Field Telemetry
    const noise = data.noise || data.noise_features || {};
    const compositeScore = Number(noise.composite_noise ?? data.noise_score ?? 0);
    const bandName = (noise.band || noise.noise_band || 'LOW').toLowerCase();

    setElementText('noise-value', compositeScore.toFixed(3));
    const noiseBandBadge = document.getElementById('noise-band');
    if (noiseBandBadge) {
      noiseBandBadge.textContent = bandName.toUpperCase();
      noiseBandBadge.className = `noise-band-badge band-${bandName}`;
    }

    setTelemetryMeter('emoji',   noise.emoji_density ?? 0);
    setTelemetryMeter('rep',     noise.repetition_score ?? noise.repetition_ratio ?? 0);
    setTelemetryMeter('codemix', noise.code_mixing_ratio ?? noise.codemix_intensity ?? 0);
    setTelemetryMeter('symbol',  noise.symbol_density ?? 0);

    // 5. Fusion Architecture Telemetry
    const alpha = Number(data.alpha ?? 0.02);
    const beta  = 1 - alpha;
    setElementText('alpha-label',   alpha.toFixed(3));
    setElementText('beta-label',    beta.toFixed(3));
    setElementText('routing-alpha', alpha.toFixed(3));
    setElementText('vader-weight',  (alpha * 100).toFixed(1) + '%');
    setElementText('distil-weight', (beta * 100).toFixed(1) + '%');
    setElementText('token-len',     data.token_length ?? '--');

    // 6. Model Interpretation
    const explanation = data.explanation || data.router?.explanation || '--';
    setElementText('router-text', explanation);
  }

  // ── Helper Utilities ──
  function setElementText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function formatPercent(v) {
    return (Number(v) * 100).toFixed(1) + '%';
  }

  function setStreamBar(prefix, val) {
    const bar = document.getElementById(`bar-${prefix}`);
    const num = document.getElementById(`val-${prefix}`);
    const widthPct = Math.min(100, Math.max(0, Number(val) * 100));
    if (bar) bar.style.width = `${widthPct}%`;
    if (num) num.textContent = formatPercent(val);
  }

  function setSubBar(prefix, val) {
    const bar = document.getElementById(`bar-${prefix}`);
    const num = document.getElementById(`val-${prefix}`);
    const widthPct = Math.min(100, Math.max(0, Number(val) * 100));
    if (bar) bar.style.width = `${widthPct}%`;
    if (num) num.textContent = `${Math.round(Number(val) * 100)}%`;
  }

  function setTelemetryMeter(key, val) {
    const fill = document.getElementById(`meter-${key}`);
    const num  = document.getElementById(`val-${key}`);
    const widthPct = Math.min(100, Math.max(0, Number(val) * 100));
    if (fill) fill.style.width = `${widthPct}%`;
    if (num)  num.textContent  = formatPercent(val);
  }
});
