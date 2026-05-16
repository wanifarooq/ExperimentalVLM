#!/usr/bin/env python3
"""Build a single self-contained interactive HTML teaser for the paper.

Reads saved run data (per_sample.jsonl + 2D filters) for a handful of curated
(image, perturbation) pairs, regenerates the perturbed images, and emits one
HTML file with everything inlined as base64 so it can be opened anywhere.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/farooq/Public/vlm-robustness")
sys.path.insert(0, str(ROOT))

from frequency_alignment.perturbations import build_perturbation_suite

RUN = ROOT / "frequency_alignment_outputs_server_2d_100_20260512_023014"
PER_SAMPLE = RUN / "exp1" / "per_sample.jsonl"
FILTERS_2D = RUN / "exp2" / "filters_2d"
IMAGE_CACHE = ROOT / ".hf_cache" / "gqa" / "images"
OUT_HTML = ROOT / "paper" / "teaser" / "index.html"

SAMPLES = [
    {
        "image_id": "2361294",
        "perturbation": "JPEG(70)|sev1",
        "terse_level": "L2_MEDIUM",
        "wordy_level": "L6_WORDY_MEDIUM",
        "label": "Lighthouse · JPEG compression",
        "caption": "Same compressed image. With a bare colour question the model's confidence in the correct answer collapses by 13.7 log-likelihood; with a padded version it loses only 0.2.",
    },
    {
        "image_id": "2374740",
        "perturbation": "JPEG(70)|sev1",
        "terse_level": "L3_FINE",
        "wordy_level": "L7_WORDY_FINE",
        "label": "Comforter · JPEG compression",
        "caption": "A spatial yes/no question. Compression knocks 9.3 log-likelihood off the correct answer when the question is terse; the wordy version drifts essentially zero.",
    },
    {
        "image_id": "2400998",
        "perturbation": "JPEG(70)|sev1",
        "terse_level": "L4_VERY_FINE",
        "wordy_level": "L8_WORDY_VERY_FINE",
        "label": "Pillow · JPEG compression",
        "caption": "Fine-grained referring question. Same image, same JPEG quality — drift 10.3 (terse) vs 0.03 (wordy). Both predictions are correct; only the model's confidence margin differs.",
    },
]

FILTER_POSITION = "last_2"


def load_per_sample(image_ids: set[str]):
    out = {}
    with open(PER_SAMPLE) as f:
        for line in f:
            d = json.loads(line)
            if d["image_id"] in image_ids:
                out[d["image_id"]] = d
    return out


def radial_average(W2d: np.ndarray) -> np.ndarray:
    h, w = W2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(W2d.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    counts = np.bincount(r.ravel())[: r_max + 1]
    sums = np.bincount(r.ravel(), W2d.ravel())[: r_max + 1]
    rad = sums / np.maximum(counts, 1)
    return np.maximum(rad, 0)


def filter_stats(image_id: str, level: str) -> tuple[float, list[float]]:
    p = FILTERS_2D / f"{image_id}_{level}_{FILTER_POSITION}.npz"
    arr = np.load(p)
    W = arr[arr.files[0]]
    rad = radial_average(W)
    rad_n = rad / (rad.sum() + 1e-12)
    ipr = float(1.0 / np.sum(rad_n ** 2))
    return ipr, rad_n.tolist()


def parse_severity(pert_name: str) -> int:
    if "|sev" in pert_name:
        return int(pert_name.split("|sev")[-1])
    return 1


def regenerate_perturbed(image_id: str, pert_name: str) -> Image.Image:
    img_path = IMAGE_CACHE / f"{image_id}.jpg"
    clean = Image.open(img_path).convert("RGB")
    sev = parse_severity(pert_name)
    suite = build_perturbation_suite(clean, severity_levels=[sev], suppress_dc=False)
    for r in suite:
        if r.name == pert_name:
            return r.perturbed_image
    raise RuntimeError(f"Perturbation {pert_name!r} not found in suite (sev {sev})")


def img_to_b64(img: Image.Image, max_side: int = 480) -> str:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_options(question_type: str, prompt: str) -> list[tuple[str, str]]:
    # We need actual option strings. Look them up from candidate_details.md is too brittle;
    # instead read from the saved candidate JSON if available, otherwise default to A=yes B=no
    # for binary, and pull from a fallback table.
    pass  # filled by main


DATASET_CACHE = ROOT / ".hf_cache" / "gqa" / "granularity_cache" / "gqa_multilevel_v6_val_seed42_max100_allowdl1.json"


def build_options_table() -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Recover option strings (A=..., B=..., ...) from the cached GQA dataset
    used by the 100-sample run."""
    out: dict[str, dict[str, list[tuple[str, str]]]] = {}
    data = json.loads(DATASET_CACHE.read_text())
    samples = data.get("samples", data) if isinstance(data, dict) else data
    for s in samples:
        iid = s.get("image_id")
        if not iid:
            continue
        out[iid] = {}
        for lvl, payload in s.get("levels", {}).items():
            opts = payload.get("options") or {}
            out[iid][lvl] = [(k, opts[k]) for k in sorted(opts)]
    return out


def main():
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    image_ids = {s["image_id"] for s in SAMPLES}
    per_sample = load_per_sample(image_ids)
    options_table = build_options_table()

    payload = {"samples": []}

    for s in SAMPLES:
        img_id = s["image_id"]
        pert = s["perturbation"]
        terse_lvl = s["terse_level"]
        wordy_lvl = s["wordy_level"]
        d = per_sample[img_id]
        l2 = d["levels"][terse_lvl]
        l6 = d["levels"][wordy_lvl]
        p_l2 = next(p for p in l2["perturbations"] if p["name"] == pert)
        p_l6 = next(p for p in l6["perturbations"] if p["name"] == pert)

        opts_l2 = options_table.get(img_id, {}).get(terse_lvl) or []
        opts_l6 = options_table.get(img_id, {}).get(wordy_lvl) or opts_l2
        if not opts_l2:
            opts_l2 = [(k, k) for k in sorted(l2["clean"]["scores"].keys())]
            opts_l6 = opts_l2

        bw_l2, rad_l2 = filter_stats(img_id, terse_lvl)
        bw_l6, rad_l6 = filter_stats(img_id, wordy_lvl)

        clean_img = Image.open(IMAGE_CACHE / f"{img_id}.jpg").convert("RGB")
        pert_img = regenerate_perturbed(img_id, pert)

        sample_entry = {
            "image_id": img_id,
            "label": s["label"],
            "caption": s["caption"],
            "perturbation": pert,
            "terse_level": terse_lvl,
            "wordy_level": wordy_lvl,
            "answer_label": l2["answer_label"],
            "terse_question": l2["question"],
            "wordy_question": l6["question"],
            "options_terse": opts_l2,
            "options_wordy": opts_l6,
            "clean_b64": img_to_b64(clean_img),
            "perturbed_b64": img_to_b64(pert_img),
            "clean_terse": {
                "predicted": l2["clean"]["predicted"],
                "correct": l2["clean"]["correct"],
                "scores": l2["clean"]["scores"],
            },
            "clean_wordy": {
                "predicted": l6["clean"]["predicted"],
                "correct": l6["clean"]["correct"],
                "scores": l6["clean"]["scores"],
            },
            "pert_terse": {
                "predicted": p_l2["predicted"],
                "correct": p_l2["correct"],
                "scores": p_l2["scores"],
                "loglik_drift": p_l2["loglik_drift"],
            },
            "pert_wordy": {
                "predicted": p_l6["predicted"],
                "correct": p_l6["correct"],
                "scores": p_l6["scores"],
                "loglik_drift": p_l6["loglik_drift"],
            },
            "bandwidth_terse": bw_l2,
            "bandwidth_wordy": bw_l6,
            "filter_radial_terse": rad_l2,
            "filter_radial_wordy": rad_l6,
        }
        payload["samples"].append(sample_entry)
        print(f"  packed {img_id} / {pert}")

    html = render_html(payload)
    OUT_HTML.write_text(html)
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"\nWrote {OUT_HTML} ({size_kb:.1f} KB)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM Robustness Teaser — Clean and perturbed log-likelihoods, terse vs wordy</title>
<style>
:root {
  --ink: #1a1a1a;
  --muted: #6b6b6b;
  --line: #d8d8d8;
  --bg: #fafaf7;
  --card: #ffffff;
  --red: #b03a2e;
  --green: #117a3d;
  --amber: #b07a16;
  --teal: #1f6b66;
  --accent: #2e4a7f;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
.wrap { max-width: 1480px; margin: 0 auto; padding: 28px 24px 60px; }
header h1 { margin: 0 0 6px; font-size: 26px; font-weight: 600; letter-spacing: -0.01em; }
header p.lede { margin: 0 0 18px; color: var(--muted); font-size: 15px; max-width: 920px; line-height: 1.45; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 24px; align-items: center; }
.controls .lbl { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-right: 6px; }
.controls button { background: var(--card); border: 1px solid var(--line); padding: 8px 14px; border-radius: 999px;
  font-size: 13px; cursor: pointer; color: var(--ink); transition: all .15s ease; }
.controls button:hover { border-color: var(--accent); color: var(--accent); }
.controls button.active { background: var(--accent); color: white; border-color: var(--accent); }

.grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 14px 14px;
  display: flex; flex-direction: column; gap: 10px; position: relative; }
.panel .corner-tag { position: absolute; top: -10px; left: 14px; background: var(--card); padding: 0 8px;
  font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; }
.panel.clean-terse .corner-tag { color: var(--green); border: 1px solid var(--green); border-radius: 4px; }
.panel.pert-terse  .corner-tag { color: var(--red);   border: 1px solid var(--red);   border-radius: 4px; }
.panel.clean-wordy .corner-tag { color: var(--teal);  border: 1px solid var(--teal);  border-radius: 4px; }
.panel.pert-wordy  .corner-tag { color: var(--amber); border: 1px solid var(--amber); border-radius: 4px; }
.panel h2 { margin: 4px 0 0; font-size: 13px; font-weight: 600; }

.qbox { background: #f3efe4; border-radius: 8px; padding: 8px 12px; font-size: 12px; line-height: 1.35;
  color: var(--ink); white-space: pre-wrap; min-height: 44px; }
.qbox.terse { background: #fde9e3; }
.qbox.wordy { background: #fff3da; }

.image-frame { background: #f3f1ed; border-radius: 6px; padding: 6px; display: flex; justify-content: center; }
.image-frame img { max-width: 100%; height: auto; max-height: 200px; border-radius: 3px; display: block; }

.opt-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.opt-table th { text-align: left; font-weight: 500; color: var(--muted); padding: 3px 5px; border-bottom: 1px solid var(--line); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; }
.opt-table td { padding: 4px 5px; border-bottom: 1px solid #eee; }
.opt-table td.num { font-variant-numeric: tabular-nums; text-align: right; font-size: 11.5px; }
.opt-table tr.correct { background: rgba(17,122,61,0.10); }
.opt-table tr.predicted-wrong { background: rgba(176,58,46,0.08); }
.opt-table .check { color: var(--green); font-weight: 600; }
.opt-table .cross { color: var(--red); font-weight: 600; }

.summary-row { font-size: 11.5px; padding: 4px 0 0; }
.summary-row .k { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
.summary-row .v { font-variant-numeric: tabular-nums; font-weight: 600; }

/* drift annotations sit BETWEEN cells in the same row */
.drift-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; margin: 6px 0 18px; }
.drift-cell { font-size: 12px; color: var(--ink); padding: 8px 12px; border-radius: 6px; text-align: center; line-height: 1.4; }
.drift-cell.empty { background: transparent; }
.drift-cell.span-terse { grid-column: 1 / span 2; background: rgba(176,58,46,0.08); border: 1px dashed var(--red); }
.drift-cell.span-wordy { grid-column: 3 / span 2; background: rgba(176,122,22,0.10); border: 1px dashed var(--amber); }
.drift-cell .label { color: var(--muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; }
.drift-cell .val   { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.drift-cell.span-terse .val { color: var(--red); }
.drift-cell.span-wordy .val { color: var(--amber); }
.drift-cell .formula { font-size: 11px; color: var(--muted); margin-top: 2px; font-family: ui-monospace, monospace; }

.filter-strip { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
.filter-panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 14px 10px; }
.filter-panel h3 { margin: 0 0 6px; font-size: 13px; font-weight: 600; }
.filter-panel h3 .ipr { color: var(--accent); font-variant-numeric: tabular-nums; }
.filter-panel svg { width: 100%; height: 90px; background: #fbf9f1; border-radius: 4px; }
.filter-panel .desc { font-size: 11.5px; color: var(--muted); margin-top: 4px; line-height: 1.4; }

footer.notes { color: var(--muted); font-size: 12px; margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--line); line-height: 1.5; }
.caption-line { font-size: 13.5px; color: var(--ink); margin: 4px 0 14px; max-width: 1200px; line-height: 1.45; }
.swap-note { display: inline-block; margin-left: 10px; padding: 2px 8px; background: #f3efe4; border-radius: 4px; font-size: 11px; color: var(--muted); }

.confound-box { background: #fbfaf3; border: 1px solid #e6dfc7; border-radius: 8px; padding: 14px 16px; margin: 16px 0 0; font-size: 12.5px; line-height: 1.5; color: var(--ink); }
.confound-box h4 { margin: 0 0 6px; font-size: 13px; }
.confound-box code { font-size: 11.5px; background: rgba(0,0,0,0.04); padding: 1px 5px; border-radius: 3px; }

@media (max-width: 1180px) {
  .grid, .drift-row { grid-template-columns: 1fr 1fr; }
  .drift-cell.span-terse, .drift-cell.span-wordy { grid-column: auto / span 2; }
  .filter-strip { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .grid, .drift-row { grid-template-columns: 1fr; }
  .drift-cell.span-terse, .drift-cell.span-wordy { grid-column: auto; }
}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Clean and perturbed log-likelihoods, terse vs wordy.</h1>
  <p class="lede">
    Four columns, same image. The first two share the bare question (clean image vs JPEG-perturbed image);
    the last two share the wordy-padded question (clean vs same perturbation). The drift label spans the
    two columns it is computed from &mdash; <em>each level's drift uses that level's own clean baseline.</em>
    The wordy clean log&nbsp;p almost always sits closer to&nbsp;0 than the terse clean log&nbsp;p, so the
    smaller wordy drift you see is partly a structural property of the conditioning, not just the
    spectral filter <em>W<sub>t</sub></em>. The <em>standardised partial regression</em> in the paper is
    what isolates the filter effect; this panel is the visual hook.
  </p>
</header>

<div class="controls">
  <span class="lbl">Pick sample</span>
  <span id="sample-buttons"></span>
  <span class="swap-note">Click to switch — numbers from the saved 100-sample Qwen3-VL-8B run, no fitting.</span>
</div>

<div id="caption" class="caption-line"></div>

<div class="grid">

  <section class="panel clean-terse">
    <span class="corner-tag">① Clean · Terse</span>
    <h2>Untouched image · bare prompt</h2>
    <div class="qbox terse" id="q-terse-1"></div>
    <div class="image-frame"><img id="img-clean-1" alt="clean image" /></div>
    <table class="opt-table"><thead>
      <tr><th>Opt</th><th>Choice</th><th class="num">log p</th><th>Result</th></tr>
    </thead><tbody id="opts-clean-terse"></tbody></table>
    <div class="summary-row"><span class="k">clean log p(gold) =</span> <span class="v" id="cl-terse"></span></div>
  </section>

  <section class="panel pert-terse">
    <span class="corner-tag">② Perturbed · Terse</span>
    <h2>JPEG hit · bare prompt</h2>
    <div class="qbox terse" id="q-terse-2"></div>
    <div class="image-frame"><img id="img-pert-1" alt="perturbed image" /></div>
    <table class="opt-table"><thead>
      <tr><th>Opt</th><th>Choice</th><th class="num">log p</th><th>Result</th></tr>
    </thead><tbody id="opts-pert-terse"></tbody></table>
    <div class="summary-row"><span class="k">perturbed log p(gold) =</span> <span class="v" id="pt-terse"></span></div>
  </section>

  <section class="panel clean-wordy">
    <span class="corner-tag">③ Clean · Wordy</span>
    <h2>Untouched image · padded prompt</h2>
    <div class="qbox wordy" id="q-wordy-1"></div>
    <div class="image-frame"><img id="img-clean-2" alt="clean image" /></div>
    <table class="opt-table"><thead>
      <tr><th>Opt</th><th>Choice</th><th class="num">log p</th><th>Result</th></tr>
    </thead><tbody id="opts-clean-wordy"></tbody></table>
    <div class="summary-row"><span class="k">clean log p(gold) =</span> <span class="v" id="cl-wordy"></span></div>
  </section>

  <section class="panel pert-wordy">
    <span class="corner-tag">④ Perturbed · Wordy</span>
    <h2>Same JPEG hit · padded prompt</h2>
    <div class="qbox wordy" id="q-wordy-2"></div>
    <div class="image-frame"><img id="img-pert-2" alt="perturbed image" /></div>
    <table class="opt-table"><thead>
      <tr><th>Opt</th><th>Choice</th><th class="num">log p</th><th>Result</th></tr>
    </thead><tbody id="opts-pert-wordy"></tbody></table>
    <div class="summary-row"><span class="k">perturbed log p(gold) =</span> <span class="v" id="pt-wordy"></span></div>
  </section>

</div>

<div class="drift-row">
  <div class="drift-cell span-terse">
    <div class="label">terse drift (① − ②)</div>
    <div class="val" id="drift-terse-val"></div>
    <div class="formula" id="drift-terse-formula"></div>
  </div>
  <div class="drift-cell span-wordy">
    <div class="label">wordy drift (③ − ④)</div>
    <div class="val" id="drift-wordy-val"></div>
    <div class="formula" id="drift-wordy-formula"></div>
  </div>
</div>

<div class="filter-strip">
  <div class="filter-panel">
    <h3>Terse filter W<sub>t</sub><sup>terse</sup> · IPR bandwidth = <span class="ipr" id="bw-terse"></span></h3>
    <svg id="filter-svg-terse"></svg>
    <div class="desc">Radially averaged cross-modal attention spectrum at the second-to-last decoder layer (<code>last_2</code>). Narrower ⇒ a single perturbed band can dominate ⟨W<sub>t</sub>, ΔF⟩.</div>
  </div>
  <div class="filter-panel">
    <h3>Wordy filter W<sub>t</sub><sup>wordy</sup> · IPR bandwidth = <span class="ipr" id="bw-wordy"></span></h3>
    <svg id="filter-svg-wordy"></svg>
    <div class="desc">Same layer, padded prompt. A flatter distribution ⇒ no single band of ΔF dominates ⇒ smaller predicted volatility.</div>
  </div>
</div>

<div class="confound-box">
  <h4>Why are the wordy clean log-probs already so close to&nbsp;0?</h4>
  Two confounds are worth disentangling before reading too much into the drift gap:
  <ul style="margin: 6px 0 6px 18px; padding: 0;">
    <li><strong>Concern A</strong> (reviewer-style): <em>maybe the wordy variant already started lower on the clean baseline ("pre-drifted"), so a small drift just means there is no room left to fall.</em>
      Empirically falsified in <strong>~99% of cells</strong>: wordy clean log&nbsp;p sits <em>higher</em> than terse clean log&nbsp;p, not lower. See <code>scripts/verify_drift_baseline_concerns.py</code>.</li>
    <li><strong>Concern B</strong> (ceiling-style): <em>maybe wordy starts so close to log&nbsp;p&nbsp;=&nbsp;0 that any perturbation is mechanically compressed.</em>
      Real, but addressed by the paper's response variable being <em>within-cell volatility</em> (shift-invariant std of drift) and the regression reporting <em>z-scored partial slopes</em> against an <em>attention-derived</em> predictor (first-order overlap or IPR bandwidth) &mdash; the slope has no log-prob units at all, so multiplicative compression of drift cancels.</li>
  </ul>
  Bottom line: the drift gap shown in this panel is the visual hook; the paper's claim rests on the standardised regression, not raw drift.
</div>

<footer class="notes">
  <strong>How to read.</strong> <em>log p</em> are next-token log-likelihoods of each option's text under the model, as logged during inference. <em>Drift</em> is (clean log p) − (perturbed log p) on the gold answer for the <em>same</em> conditioning column &mdash; terse drift uses ① − ②, wordy drift uses ③ − ④. <em>IPR bandwidth</em> = inverse participation ratio of the radially averaged cross-modal attention spectrum at <code>last_2</code>; higher means flatter (broader) frequency support. All numbers come from the saved Qwen3-VL-8B 100-sample run with <code>suppress_dc:&nbsp;false</code>.
</footer>

</div>

<script>
const PAYLOAD = __PAYLOAD__;

function fmt(v, d=2) {
  if (v === null || v === undefined) return "—";
  if (!isFinite(v)) return v.toString();
  return v.toFixed(d);
}

function renderFilter(svgEl, radial, color) {
  const w = svgEl.clientWidth || 320, h = svgEl.clientHeight || 70;
  svgEl.innerHTML = "";
  const pad = 6;
  const n = radial.length;
  const max = Math.max(...radial);
  const pts = radial.map((v, i) => {
    const x = pad + (w - 2*pad) * i / (n - 1);
    const y = h - pad - (h - 2*pad) * v / (max || 1);
    return [x, y];
  });
  // baseline
  const base = document.createElementNS("http://www.w3.org/2000/svg", "line");
  base.setAttribute("x1", pad); base.setAttribute("x2", w - pad);
  base.setAttribute("y1", h - pad); base.setAttribute("y2", h - pad);
  base.setAttribute("stroke", "#ccc"); base.setAttribute("stroke-width", "1");
  svgEl.appendChild(base);
  // fill area
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  let d = `M ${pad} ${h-pad}`;
  pts.forEach(([x,y]) => d += ` L ${x} ${y}`);
  d += ` L ${w-pad} ${h-pad} Z`;
  path.setAttribute("d", d);
  path.setAttribute("fill", color);
  path.setAttribute("fill-opacity", "0.18");
  path.setAttribute("stroke", color);
  path.setAttribute("stroke-width", "1.5");
  svgEl.appendChild(path);
  // x-axis label
  const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
  lbl.textContent = "low → high frequency band";
  lbl.setAttribute("x", w / 2); lbl.setAttribute("y", h - 1);
  lbl.setAttribute("text-anchor", "middle");
  lbl.setAttribute("font-size", "9"); lbl.setAttribute("fill", "#888");
  svgEl.appendChild(lbl);
}

function renderOptions(tbody, options, scores, predicted, gold) {
  tbody.innerHTML = "";
  options.forEach(([k, label]) => {
    const tr = document.createElement("tr");
    if (k === gold && k === predicted) tr.classList.add("correct");
    else if (k === predicted && k !== gold) tr.classList.add("predicted-wrong");
    const v = scores[k];
    let resultText = "";
    if (k === predicted && k === gold) resultText = '<span class="check">✓ predicted</span>';
    else if (k === predicted) resultText = '<span class="cross">✗ predicted</span>';
    else if (k === gold) resultText = '<span class="check">gold</span>';
    tr.innerHTML = `
      <td><strong>${k}</strong></td>
      <td>${label}</td>
      <td class="num">${fmt(v, 2)}</td>
      <td>${resultText}</td>
    `;
    tbody.appendChild(tr);
  });
}

function setSample(idx) {
  const s = PAYLOAD.samples[idx];
  document.querySelectorAll("#sample-buttons button").forEach((b, i) => {
    b.classList.toggle("active", i === idx);
  });
  const lvlPair = `${s.terse_level} ↔ ${s.wordy_level}`;
  document.getElementById("caption").innerHTML =
    `<strong>${s.label}.</strong> Perturbation: <code>${s.perturbation}</code> · ` +
    `Level pair: <code>${lvlPair}</code> · Gold = <strong>${s.answer_label}</strong>. ${s.caption}`;

  // Questions
  document.getElementById("q-terse-1").textContent = `“${s.terse_question}”`;
  document.getElementById("q-terse-2").textContent = `“${s.terse_question}”`;
  document.getElementById("q-wordy-1").textContent = `“${s.wordy_question}”`;
  document.getElementById("q-wordy-2").textContent = `“${s.wordy_question}”`;

  // Images
  const cleanSrc = "data:image/png;base64," + s.clean_b64;
  const pertSrc  = "data:image/png;base64," + s.perturbed_b64;
  document.getElementById("img-clean-1").src = cleanSrc;
  document.getElementById("img-pert-1").src  = pertSrc;
  document.getElementById("img-clean-2").src = cleanSrc;
  document.getElementById("img-pert-2").src  = pertSrc;

  // Option tables (clean and perturbed for each prompt)
  renderOptions(document.getElementById("opts-clean-terse"),
    s.options_terse, s.clean_terse.scores, s.clean_terse.predicted, s.answer_label);
  renderOptions(document.getElementById("opts-pert-terse"),
    s.options_terse, s.pert_terse.scores,  s.pert_terse.predicted,  s.answer_label);
  renderOptions(document.getElementById("opts-clean-wordy"),
    s.options_wordy, s.clean_wordy.scores, s.clean_wordy.predicted, s.answer_label);
  renderOptions(document.getElementById("opts-pert-wordy"),
    s.options_wordy, s.pert_wordy.scores,  s.pert_wordy.predicted,  s.answer_label);

  // Summary log p(gold) on each panel
  const goldT = s.answer_label;
  const goldW = s.answer_label;
  const cl_t = s.clean_terse.scores[goldT];
  const pt_t = s.pert_terse.scores[goldT];
  const cl_w = s.clean_wordy.scores[goldW];
  const pt_w = s.pert_wordy.scores[goldW];
  document.getElementById("cl-terse").textContent = fmt(cl_t, 2);
  document.getElementById("pt-terse").textContent = fmt(pt_t, 2);
  document.getElementById("cl-wordy").textContent = fmt(cl_w, 2);
  document.getElementById("pt-wordy").textContent = fmt(pt_w, 2);

  // Drift annotations
  const dT = s.pert_terse.loglik_drift;
  const dW = s.pert_wordy.loglik_drift;
  document.getElementById("drift-terse-val").textContent = (dT >= 0 ? "+" : "") + fmt(dT, 2);
  document.getElementById("drift-wordy-val").textContent = (dW >= 0 ? "+" : "") + fmt(dW, 2);
  document.getElementById("drift-terse-formula").textContent =
    `${fmt(cl_t, 2)} − (${fmt(pt_t, 2)}) on option ${goldT}`;
  document.getElementById("drift-wordy-formula").textContent =
    `${fmt(cl_w, 2)} − (${fmt(pt_w, 2)}) on option ${goldW}`;

  // Bandwidth and filter SVGs
  document.getElementById("bw-terse").textContent = fmt(s.bandwidth_terse, 2);
  document.getElementById("bw-wordy").textContent = fmt(s.bandwidth_wordy, 2);
  renderFilter(document.getElementById("filter-svg-terse"), s.filter_radial_terse, "#b03a2e");
  renderFilter(document.getElementById("filter-svg-wordy"), s.filter_radial_wordy, "#b07a16");
}

function init() {
  const btnHost = document.getElementById("sample-buttons");
  PAYLOAD.samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.textContent = s.label;
    b.addEventListener("click", () => setSample(i));
    btnHost.appendChild(b);
  });
  setSample(0);
  window.addEventListener("resize", () => setSample(PAYLOAD.samples.findIndex((_, i) => document.querySelectorAll("#sample-buttons button")[i].classList.contains("active"))));
}
document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    js_payload = json.dumps(payload, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__PAYLOAD__", js_payload)


if __name__ == "__main__":
    main()
