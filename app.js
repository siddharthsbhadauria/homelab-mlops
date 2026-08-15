// Interactive Client-Side MLOps Engine Simulator
document.addEventListener('DOMContentLoaded', () => {
  // Theme Toggle
  const themeToggle = document.getElementById('themeToggle');
  const themeIcon = document.getElementById('themeIcon');
  const body = document.body;

  const savedTheme = localStorage.getItem('mlops_theme') || 'dark';
  if (savedTheme === 'light') {
    body.classList.remove('dark-theme');
    body.classList.add('light-theme');
    themeIcon.textContent = '🌙';
  }

  themeToggle.addEventListener('click', () => {
    if (body.classList.contains('dark-theme')) {
      body.classList.remove('dark-theme');
      body.classList.add('light-theme');
      themeIcon.textContent = '🌙';
      localStorage.setItem('mlops_theme', 'light');
    } else {
      body.classList.remove('light-theme');
      body.classList.add('dark-theme');
      themeIcon.textContent = '☀️';
      localStorage.setItem('mlops_theme', 'dark');
    }
  });

  // Slider Elements
  const cpuRange = document.getElementById('cpuRange');
  const ramRange = document.getElementById('ramRange');
  const diskRange = document.getElementById('diskRange');
  const spikeRange = document.getElementById('spikeRange');

  const cpuVal = document.getElementById('cpuVal');
  const ramVal = document.getElementById('ramVal');
  const diskVal = document.getElementById('diskVal');
  const spikeVal = document.getElementById('spikeVal');

  // Outputs
  const inferenceBadge = document.getElementById('inferenceBadge');
  const statusIcon = document.getElementById('statusIcon');
  const statusTitle = document.getElementById('statusTitle');
  const statusSub = document.getElementById('statusSub');
  const decisionScore = document.getElementById('decisionScore');
  const scoreBar = document.getElementById('scoreBar');
  const jsonPayload = document.getElementById('jsonPayload');

  // Simulation Evaluation (Mimics IsolationForest decision function with learned boundaries)
  function evaluateInference() {
    const cpu = parseFloat(cpuRange.value);
    const ram = parseFloat(ramRange.value);
    const disk = parseFloat(diskRange.value);
    const spike = parseFloat(spikeRange.value);

    // Update labels
    cpuVal.textContent = `${cpu.toFixed(1)}%`;
    ramVal.textContent = `${ram.toFixed(1)}%`;
    diskVal.textContent = `${disk.toFixed(1)}%`;
    spikeVal.textContent = `${spike >= 0 ? '+' : ''}${spike.toFixed(1)}%`;

    // Decision function formula approximating trained multi-variate isolation trees
    // Nominal baseline: cpu ~15%, ram ~30%, disk ~25%, spike ~0%
    const cpuPenalty = Math.max(0, (cpu - 70) / 30) * 0.45;
    const ramPenalty = Math.max(0, (ram - 80) / 20) * 0.35;
    const diskPenalty = Math.max(0, (disk - 85) / 15) * 0.30;
    const spikePenalty = (Math.abs(spike) > 20 ? (Math.abs(spike) - 20) / 30 : 0) * 0.40;

    const baseScore = 0.22;
    const penalty = cpuPenalty + ramPenalty + diskPenalty + spikePenalty;
    const score = baseScore - penalty;

    const isAnomaly = score < 0.0 || cpu > 90 || ram > 90 || Math.abs(spike) > 35;

    // Update UI elements
    decisionScore.textContent = (score >= 0 ? '+' : '') + score.toFixed(4);

    // Scale progress bar (-0.5 to +0.5 -> 0% to 100%)
    const clampedScore = Math.max(-0.5, Math.min(0.5, score));
    const barPct = ((clampedScore + 0.5) / 1.0) * 100;
    scoreBar.style.width = `${barPct}%`;

    if (isAnomaly) {
      inferenceBadge.className = 'inference-status anomaly-alert';
      statusIcon.textContent = '🚨';
      statusTitle.textContent = 'ANOMALY DETECTED';
      statusTitle.style.color = 'var(--accent-red)';
      statusSub.textContent = 'Telemetry vector breached IsolationForest boundary! Incident dispatched to GitHub Issues.';
      scoreBar.style.backgroundColor = 'var(--accent-red)';
    } else {
      inferenceBadge.className = 'inference-status';
      statusIcon.textContent = '🟢';
      statusTitle.textContent = 'OPERATIONAL HEALTHY';
      statusTitle.style.color = 'var(--text-primary)';
      statusSub.textContent = 'Inference evaluated within nominal bounds (5% contamination envelope).';
      scoreBar.style.backgroundColor = 'var(--accent-green)';
    }

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

  // Event Listeners
  [cpuRange, ramRange, diskRange, spikeRange].forEach(el => {
    el.addEventListener('input', evaluateInference);
  });

  // Presets
  const btnPresetAnomaly = document.getElementById('btnPresetAnomaly');
  const btnPresetNormal = document.getElementById('btnPresetNormal');

  btnPresetAnomaly.addEventListener('click', () => {
    cpuRange.value = 94.5;
    ramRange.value = 88.0;
    diskRange.value = 45.0;
    spikeRange.value = 42.0;
    evaluateInference();
  });

  btnPresetNormal.addEventListener('click', () => {
    cpuRange.value = 16.5;
    ramRange.value = 26.0;
    diskRange.value = 24.0;
    spikeRange.value = 0.8;
    evaluateInference();
  });

  // Copy JSON
  const btnCopyJson = document.getElementById('btnCopyJson');
  btnCopyJson.addEventListener('click', () => {
    navigator.clipboard.writeText(jsonPayload.textContent).then(() => {
      btnCopyJson.textContent = 'Copied! ✓';
      setTimeout(() => {
        btnCopyJson.textContent = 'Copy JSON';
      }, 2000);
    });
  });

  // Initialize on load
  evaluateInference();
});
