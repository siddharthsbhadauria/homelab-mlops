// ==========================================================================
// Homelab MLOps & Telemetry Anomaly Detection Platform - Frontend Engine
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSimulator();
});

// --------------------------------------------------------------------------
// 1. Dual Theme Engine
// --------------------------------------------------------------------------
function initTheme() {
  const savedTheme = localStorage.getItem('theme');
  const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  const initialTheme = savedTheme || (prefersLight ? 'light' : 'dark');

  document.documentElement.setAttribute('data-theme', initialTheme);

  const themeToggleBtn = document.getElementById('themeToggle');
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      showToast(newTheme === 'dark' ? '🌙 Dark Mode Activated' : '☀️ Light Mode Activated');
    });
  }
}

function showToast(message) {
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toastMessage');
  if (!toast || !toastMsg) return;

  toastMsg.textContent = message;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 3000);
}

// --------------------------------------------------------------------------
// 2. Interactive Telemetry Anomaly Simulator
// --------------------------------------------------------------------------
function initSimulator() {
  const cpuRange = document.getElementById('cpuRange');
  const ramRange = document.getElementById('ramRange');
  const diskRange = document.getElementById('diskRange');
  const spikeRange = document.getElementById('spikeRange');

  const cpuVal = document.getElementById('cpuVal');
  const ramVal = document.getElementById('ramVal');
  const diskVal = document.getElementById('diskVal');
  const spikeVal = document.getElementById('spikeVal');

  const inferenceBadge = document.getElementById('inferenceBadge');
  const statusIcon = document.getElementById('statusIcon');
  const statusTitle = document.getElementById('statusTitle');
  const statusSub = document.getElementById('statusSub');
  const decisionScore = document.getElementById('decisionScore');
  const scoreBar = document.getElementById('scoreBar');
  const jsonPayload = document.getElementById('jsonPayload');

  function evaluateInference() {
    if (!cpuRange || !ramRange || !diskRange || !spikeRange) return;

    const cpu = parseFloat(cpuRange.value);
    const ram = parseFloat(ramRange.value);
    const disk = parseFloat(diskRange.value);
    const spike = parseFloat(spikeRange.value);

    // Update label text
    if (cpuVal) cpuVal.textContent = `${cpu.toFixed(1)}%`;
    if (ramVal) ramVal.textContent = `${ram.toFixed(1)}%`;
    if (diskVal) diskVal.textContent = `${disk.toFixed(1)}%`;
    if (spikeVal) spikeVal.textContent = `${spike >= 0 ? '+' : ''}${spike.toFixed(1)}%`;

    // Isolation Forest decision score approximation
    const cpuPenalty = Math.max(0, (cpu - 70) / 30) * 0.45;
    const ramPenalty = Math.max(0, (ram - 80) / 20) * 0.35;
    const diskPenalty = Math.max(0, (disk - 85) / 15) * 0.30;
    const spikePenalty = (Math.abs(spike) > 20 ? (Math.abs(spike) - 20) / 30 : 0) * 0.40;

    const baseScore = 0.22;
    const penalty = cpuPenalty + ramPenalty + diskPenalty + spikePenalty;
    const score = baseScore - penalty;

    const isAnomaly = score < 0.0 || cpu > 90 || ram > 90 || Math.abs(spike) > 35;

    // Update decision score UI
    if (decisionScore) decisionScore.textContent = (score >= 0 ? '+' : '') + score.toFixed(4);

    // Scale progress bar (-0.5 to +0.5 -> 0% to 100%)
    if (scoreBar) {
      const clampedScore = Math.max(-0.5, Math.min(0.5, score));
      const barPct = ((clampedScore + 0.5) / 1.0) * 100;
      scoreBar.style.width = `${barPct}%`;
      scoreBar.style.backgroundColor = isAnomaly ? 'var(--accent-red)' : 'var(--accent-green)';
    }

    if (inferenceBadge && statusTitle && statusSub && statusIcon) {
      if (isAnomaly) {
        inferenceBadge.className = 'inference-status anomaly-alert';
        statusIcon.textContent = '🚨';
        statusTitle.textContent = 'ANOMALY DETECTED';
        statusTitle.style.color = 'var(--accent-red)';
        statusSub.textContent = 'Telemetry vector breached IsolationForest boundary! Incident dispatched to GitHub Issues.';
      } else {
        inferenceBadge.className = 'inference-status';
        statusIcon.textContent = '🟢';
        statusTitle.textContent = 'OPERATIONAL HEALTHY';
        statusTitle.style.color = 'var(--text-main)';
        statusSub.textContent = 'Inference evaluated within nominal bounds (5% contamination envelope).';
      }
    }

    if (jsonPayload) {
      const payload = {
        timestamp: new Date().toISOString(),
        system: {
          cpu_percent: cpu,
          ram_percent: ram,
          disk_percent: disk
        },
        rate_of_change: {
          cpu_roc: spike
        },
        anomaly: isAnomaly,
        anomaly_score: parseFloat(score.toFixed(4)),
        model_type: "IsolationForest",
        model_version: "v1.0-prod"
      };
      jsonPayload.textContent = JSON.stringify(payload, null, 2);
    }
  }

  // Bind slider input events
  [cpuRange, ramRange, diskRange, spikeRange].forEach(slider => {
    if (slider) slider.addEventListener('input', evaluateInference);
  });

  // Preset Buttons
  const btnPresetAnomaly = document.getElementById('btnPresetAnomaly');
  const btnPresetNormal = document.getElementById('btnPresetNormal');

  if (btnPresetAnomaly) {
    btnPresetAnomaly.addEventListener('click', () => {
      cpuRange.value = 94.5;
      ramRange.value = 88.0;
      diskRange.value = 45.0;
      spikeRange.value = 42.0;
      evaluateInference();
      showToast('🚨 Simulated Heavy CPU Spike');
    });
  }

  if (btnPresetNormal) {
    btnPresetNormal.addEventListener('click', () => {
      cpuRange.value = 16.5;
      ramRange.value = 26.0;
      diskRange.value = 24.0;
      spikeRange.value = 0.8;
      evaluateInference();
      showToast('✅ Reset to Nominal Homelab Telemetry');
    });
  }

  // Copy JSON Payload
  const btnCopyJson = document.getElementById('btnCopyJson');
  if (btnCopyJson) {
    btnCopyJson.addEventListener('click', () => {
      if (jsonPayload) {
        navigator.clipboard.writeText(jsonPayload.textContent).then(() => {
          btnCopyJson.textContent = 'Copied! ✓';
          showToast('📋 Payload Copied to Clipboard');
          setTimeout(() => {
            btnCopyJson.textContent = 'Copy JSON';
          }, 2000);
        });
      }
    });
  }

  // Run initial evaluation
  evaluateInference();
}
