"use strict";

const $ = (id) => document.getElementById(id);
let last = null;
let currentPid = "demo-person-001";

async function run() {
  currentPid = $("pid").value.trim() || "demo-person-001";
  const age = parseFloat($("age").value) || 49;
  const duration = parseInt($("duration").value) || 600;
  $("status").textContent = "Running twin...";
  try {
    const res = await fetch(
      `/api/twin/demo?person_id=${encodeURIComponent(currentPid)}&age=${age}&duration_s=${duration}`
    );
    last = await res.json();
    render();
    $("status").textContent = `${last.snapshots.length} windows processed`;
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  }
}

function render() {
  if (!last || !last.snapshots.length) return;
  const s = last.snapshots[last.snapshots.length - 1];

  $("stage").textContent = pretty(s.most_likely_stage);
  $("confidence").textContent =
    `P = ${(s.confidence * 100).toFixed(1)}%  (capped below certainty)`;

  renderPosterior(s.stage_posterior);
  renderIndex(s.menopause_index_z);
  renderField(s.latest_named);
  renderEvents(last.events);
  drawTimeline(last.snapshots);
}

function pretty(id) {
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderPosterior(dist) {
  const el = $("posterior");
  el.innerHTML = "";
  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  for (const [k, v] of entries) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span>${pretty(k)}</span>
      <div class="bar"><span style="width:${Math.min(100, v * 100).toFixed(1)}%"></span></div>
      <span>${(v * 100).toFixed(0)}%</span>`;
    el.appendChild(row);
  }
}

function renderIndex(z) {
  const el = $("mindex");
  el.textContent = (z >= 0 ? "+" : "") + z.toFixed(2);
  el.style.color = z > 0.5 ? "var(--danger)" : z > 0 ? "var(--warn)" : "var(--accent)";
}

function renderField(named) {
  const el = $("field");
  el.innerHTML = "";
  const show = ["vasomotor_index", "temp_gradient", "eda_peaks", "hrv_low", "stress", "fatigue"];
  for (const key of show) {
    if (named[key] === undefined) continue;
    const li = document.createElement("li");
    li.innerHTML = `<span>${pretty(key)}</span><span class="temp">${named[key].toFixed(3)}</span>`;
    el.appendChild(li);
  }
}

function renderEvents(events) {
  const el = $("events");
  el.innerHTML = "";
  if (!events.length) {
    el.innerHTML = "<li>No vasomotor events detected in window.</li>";
    return;
  }
  for (const e of events) {
    const li = document.createElement("li");
    li.innerHTML = `t=${e.onset_s.toFixed(0)}s &rarr; ${e.end_s.toFixed(0)}s &bull; &Delta;T peak ${e.peak_delta_c.toFixed(2)}&deg;C &bull; ` +
      `spread ${e.propagation_sites.length} sites (delay ${e.spatial_delay_s.toFixed(0)}s)`;
    el.appendChild(li);
  }
}

function drawTimeline(snaps) {
  const c = $("timeline");
  const ctx = c.getContext("2d");
  const W = (c.width = c.clientWidth);
  const H = (c.height = 90);
  ctx.clearRect(0, 0, W, H);

  const xs = snaps.map((_, i) => i);
  const maxX = Math.max(1, xs.length - 1);
  const plot = (getter, color, fill) => {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    snaps.forEach((s, i) => {
      const x = (i / maxX) * (W - 20) + 10;
      const y = H - 10 - Math.min(1, Math.max(0, getter(s))) * (H - 24);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (fill) {
      ctx.lineTo(W - 10, H - 10);
      ctx.lineTo(10, H - 10);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
  };

  plot((s) => s.vasomotor_index || 0, "#ff6b81", "rgba(255,107,129,0.12)");
  plot((s) => s.stress || 0, "#6ea8fe", null);
  plot((s) => Math.min(1, Math.abs(s.menopause_index_z) / 3), "#c58bff", null);

  ctx.fillStyle = "#93a0c4";
  ctx.font = "11px Segoe UI";
  ctx.fillText("vasomotor", 12, 14);
  ctx.fillStyle = "#6ea8fe";
  ctx.fillText("stress", 84, 14);
  ctx.fillStyle = "#c58bff";
  ctx.fillText("menopause z", 140, 14);
}

async function submitCorrection() {
  const payload = {
    variable: $("var").value,
    predicted: parseFloat($("pred").value),
    observed: parseFloat($("obs").value),
    t: last && last.snapshots.length ? last.snapshots[last.snapshots.length - 1].t : 0,
  };
  try {
    const res = await fetch(`/api/twin/${encodeURIComponent(currentPid)}/correct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    $("corrOut").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("corrOut").textContent = "Error: " + e.message;
  }
}

$("run").addEventListener("click", run);
$("correct").addEventListener("click", submitCorrection);
window.addEventListener("load", run);
