"""
Gait Pattern Analysis — course prototype interface (EXSU 500)

Run locally:   python3 -m streamlit run app.py

Three screens: Welcome -> Input -> Results.
The model plugs in through model_slot.py (see that file). Until a trained model
is saved at model/model.joblib, the app runs in DEMO MODE with placeholder results.
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from features import FEATURE_INFO, FEATURE_ORDER, GaitFileError, compute_features, load_ts
from model_slot import THRESHOLD, load_model, load_model_info, load_reference, predict

APP_DIR = Path(__file__).parent
EXAMPLES_DIR = APP_DIR / "examples"

st.set_page_config(page_title="Gait Pattern Analysis — course prototype", page_icon="🦶",
                   layout="wide")

# ----------------------------------------------------------------- styling --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Public Sans', system-ui, sans-serif; }
.stApp { background: #F5F6F3; color: #17201E; }
.block-container { max-width: 1120px; padding-top: 4.5rem; }
h1, h2 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important; color: #17201E; }
.disclaimer { background: #0E3B38; color: #fff; padding: 12px 20px; border-radius: 10px;
  text-align: center; font-size: 14px; line-height: 1.4; margin-bottom: 8px; }
.topbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
  gap: 12px; padding: 10px 0 18px; }
.brand { font-weight: 700; font-size: 17px; display: flex; align-items: center; gap: 10px; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }
.pill { padding: 6px 12px; border-radius: 999px; border: 1px solid #D9DDD8; background: #fff; color: #4A5552; }
.pill.on { background: #0E5C57; color: #fff; border-color: #0E5C57; font-weight: 600; }
.pill.done { background: #E3EFED; color: #0E5C57; border-color: #E3EFED; font-weight: 600; }
.card { background: #fff; border: 1px solid #D9DDD8; border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.card h3 { margin: 0 0 6px; font-size: 18px; }
.muted { color: #4A5552; font-size: 14px; line-height: 1.55; }
.eyebrow { font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #4A5552; margin: 0 0 8px; }
.demo { background: #FFF7DB; border: 1px solid #E8D27A; color: #5C4A05; border-radius: 10px;
  padding: 12px 16px; font-size: 14px; margin-bottom: 16px; }
.feat-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 12px 0; border-top: 1px solid #ECEEEA; }
.feat-name { flex: 1 1 180px; min-width: 0; }
.feat-name b { display: block; font-size: 15px; }
.feat-name span { font-size: 13px; color: #4A5552; }
.track { flex: 999 1 200px; min-width: 0; position: relative; height: 24px; }
.track .line { position: absolute; left: 0; right: 0; top: 10px; height: 4px; border-radius: 99px; background: #E6E9E5; }
.track .band { position: absolute; top: 6px; height: 12px; border-radius: 4px; background: #BFD9D4; }
.track .dot { position: absolute; top: 4px; width: 16px; height: 16px; margin-left: -8px; border-radius: 99px; border: 2px solid #fff; box-sizing: border-box; }
.feat-val { flex: 0 0 112px; text-align: right; }
.feat-val b { display: block; font-size: 15px; }
.feat-val span { font-size: 12px; color: #4A5552; }
.flag { color: #A34A0B; }
.rank-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-top: 1px solid #ECEEEA; }
.rank-num { flex: none; width: 28px; height: 28px; border-radius: 99px; display: inline-flex; align-items: center;
  justify-content: center; font-size: 13px; font-weight: 700; background: #E3EFED; color: #0E5C57; }
.rank-num.top { background: #0E5C57; color: #fff; }
.rank-body { flex: 1; min-width: 0; font-size: 14px; }
.rank-head { display: flex; justify-content: space-between; gap: 8px; }
.rank-bar { height: 6px; border-radius: 99px; background: #F1F2EF; margin-top: 6px; overflow: hidden; }
.rank-bar div { height: 100%; border-radius: 99px; }
.prob-track { position: relative; height: 28px; margin: 8px 0 4px; }
.prob-track .line { position: absolute; left: 0; right: 0; top: 10px; height: 8px; border-radius: 99px; background: #E6E9E5; }
.prob-track .range { position: absolute; top: 8px; height: 12px; border-radius: 99px; background: #E9B48F; }
.prob-track .thr { position: absolute; left: 50%; top: 2px; width: 2px; height: 24px; background: #4A5552; }
.prob-track .dot { position: absolute; top: 4px; width: 20px; height: 20px; margin-left: -10px; border-radius: 99px;
  border: 3px solid #fff; box-shadow: 0 0 0 1px #A34A0B; background: #A34A0B; box-sizing: border-box; }
.scale { display: flex; justify-content: space-between; font-size: 12px; color: #4A5552; }
.stButton button[kind="primary"] { background: #0E5C57; border-color: #0E5C57; }
.stButton button[kind="primary"]:hover { background: #0A4440; border-color: #0A4440; }
</style>
""", unsafe_allow_html=True)

PAGES = ["welcome", "input", "results"]
LABELS = {"welcome": "1 · Welcome", "input": "2 · Input", "results": "3 · Results"}
st.session_state.setdefault("page", "welcome")


def go(page):
    st.session_state.page = page


def chrome():
    """Disclaimer banner (required on every screen) and progress header."""
    st.markdown('<div class="disclaimer" role="note"><strong>Course prototype for research and '
                'education only.</strong> Not a medical device. Not for clinical use or diagnosis.</div>',
                unsafe_allow_html=True)
    current = PAGES.index(st.session_state.page)
    pills = "".join(
        f'<span class="pill {"on" if i == current else "done" if i < current else ""}">'
        f'{LABELS[p]}{" ✓" if i < current else ""}</span>' for i, p in enumerate(PAGES))
    st.markdown(f'<div class="topbar"><div class="brand">'
                '<svg width="26" height="26" viewBox="0 0 32 32" fill="none" stroke="#0E5C57" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 20h5l3-9 4 15 4-19 3 13h9"/></svg>'
                f'Gait Pattern Analysis</div><nav class="pills" aria-label="Progress">{pills}</nav></div>',
                unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


# ================================================================ WELCOME ==
def page_welcome():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<p class="eyebrow" style="color:#0E5C57">EXSU 500 · Team project</p>', unsafe_allow_html=True)
        st.markdown("# Can the way we walk reveal neurodegenerative disease?")
        st.markdown('<p class="muted" style="font-size:17px">This tool analyses a walking recording from '
                    'pressure-sensing insoles and shows whether the gait pattern more closely resembles healthy '
                    'adults or people with Parkinson’s disease, Huntington’s disease or ALS, and which features '
                    'of walking drive that result.</p>', unsafe_allow_html=True)
        st.button("Start analysis →", type="primary", on_click=go, args=("input",))
    with right:
        st.markdown("""
<div class="card">
<svg role="img" aria-label="Footprints above a steady and an irregular stride-time trace" viewBox="0 0 480 300" style="width:100%;height:auto;display:block">
<g fill="#CFE3DF"><ellipse cx="60" cy="70" rx="15" ry="26"/><ellipse cx="130" cy="40" rx="15" ry="26"/><ellipse cx="200" cy="70" rx="15" ry="26"/><ellipse cx="270" cy="40" rx="15" ry="26"/><ellipse cx="340" cy="70" rx="15" ry="26"/><ellipse cx="410" cy="40" rx="15" ry="26"/></g>
<line x1="24" y1="140" x2="456" y2="140" stroke="#D9DDD8"/>
<text x="24" y="168" font-size="13" fill="#4A5552">Healthy stride rhythm</text>
<polyline fill="none" stroke="#0E5C57" stroke-width="2.5" points="24,200 60,198 96,201 132,199 168,200 204,198 240,201 276,199 312,200 348,198 384,201 420,199 456,200"/>
<text x="24" y="238" font-size="13" fill="#4A5552">Irregular stride rhythm</text>
<polyline fill="none" stroke="#A34A0B" stroke-width="2.5" points="24,272 60,262 96,286 132,266 168,280 204,256 240,284 276,270 312,290 348,258 384,278 420,264 456,282"/>
</svg>
<p class="muted" style="margin:12px 0 0">In healthy walking, each stride takes almost exactly the same time.
Greater stride-to-stride variability and left–right asymmetry are among the gait changes described in
neurodegenerative disease.</p></div>""", unsafe_allow_html=True)

    st.markdown("## How it works")
    steps = [("STEP 1", "Provide a recording", "Upload a stride time-series file (.ts) from force-sensing "
              "insoles, or pick an example recording."),
             ("STEP 2", "We summarise the walk", "Hundreds of steps are condensed into nine gait features: "
              "timing, variability and left–right symmetry."),
             ("STEP 3", "See the result and why", "A probability with its uncertainty range, each feature "
              "compared with healthy adults, and the model’s known limits.")]
    for col, (tag, title, text) in zip(st.columns(3), steps):
        col.markdown(f'<div class="card"><p class="eyebrow" style="color:#0E5C57">{tag}</p>'
                     f'<h3>{title}</h3><p class="muted" style="margin:0">{text}</p></div>',
                     unsafe_allow_html=True)
    st.markdown('<p class="muted">Built on the PhysioNet Gait in Neurodegenerative Disease Database '
                '(64 participants). Uploaded files are processed in memory and never stored.</p>',
                unsafe_allow_html=True)


# ================================================================== INPUT ==
def analyse(raw_bytes, name):
    """Validate, compute features and store everything the results page needs."""
    try:
        df, warnings = load_ts(raw_bytes)
    except GaitFileError as e:
        st.session_state.error = str(e)
        return
    feats = compute_features(df)
    reference = load_reference()
    result = predict(get_model(), df, feats, reference)
    st.session_state.update(error=None, df=df, features=feats, warnings=warnings,
                            filename=name, result=result, page="results")


def page_input():
    st.markdown("# Provide a walking recording")
    st.markdown('<p class="muted" style="font-size:16px">Upload one participant’s stride time-series, '
                'or try an example recording.</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.6, 1], gap="large")
    with col_a:
        st.markdown("### Option A · Upload a file")
        upload = st.file_uploader("Gait recording (.ts)", type=["ts", "txt"],
                                  help="Plain-text PhysioNet gait file: 13 columns, one row per stride.")
        st.markdown(f"""<div class="muted"><b style="color:#17201E">File requirements</b><ul>
<li>13 whitespace-separated columns per row (PhysioNet gait .ts format)</li>
<li>One row per stride; at least 50 strides</li>
<li>Times in seconds; numbers only</li>
<li>A failed foot sensor is detected and reported</li></ul></div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("### Option B · Use an example")
        examples = sorted(EXAMPLES_DIR.glob("*.ts")) if EXAMPLES_DIR.exists() else []
        if examples:
            choice = st.selectbox("Example recording", ["Choose a recording…"] + [p.name for p in examples])
        else:
            choice = None
            st.markdown('<p class="muted">No example recordings yet. Put a few held-out .ts files in the '
                        '<code>examples/</code> folder to enable this option.</p>', unsafe_allow_html=True)

    if st.session_state.get("error"):
        st.error(st.session_state.error, icon="⚠️")

    st.markdown('<p class="muted">🔒 Files are processed in memory and deleted after analysis. '
                'Nothing is stored.</p>', unsafe_allow_html=True)
    back, _, run = st.columns([1, 3, 1.3])
    back.button("Back", on_click=go, args=("welcome",), use_container_width=True)
    if run.button("Analyse gait →", type="primary", use_container_width=True):
        with st.spinner("Analysing gait…"):
            if upload is not None:
                analyse(upload.getvalue(), upload.name)
            elif choice and choice != "Choose a recording…":
                analyse((EXAMPLES_DIR / choice).read_bytes(), choice)
            else:
                st.session_state.error = "Please upload a file or choose an example recording first."
        st.rerun()


# ================================================================ RESULTS ==
def pct(v, lo, hi):
    return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))


def page_results():
    if "result" not in st.session_state:
        go("input")
        st.rerun()
    res, feats = st.session_state.result, st.session_state.features
    ref = load_reference()
    info = load_model_info()

    if res["demo"]:
        st.markdown('<div class="demo" role="status"><strong>Demo mode:</strong> no trained model is '
                    'connected yet, so the probability, its range and the ranking are <b>placeholders</b>. '
                    'The feature values below are real, computed from your file.</div>', unsafe_allow_html=True)
    for w in st.session_state.warnings:
        st.info(w, icon="ℹ️")

    st.markdown(f'<p class="muted" style="margin:0">Recording: {html.escape(st.session_state.filename)} · '
                f'{len(st.session_state.df)} strides analysed</p>', unsafe_allow_html=True)
    st.markdown("# Results")

    # --- prediction + reliability -------------------------------------------
    p, lo, hi = res["prob"], res["prob_low"], res["prob_high"]
    disease = p >= THRESHOLD
    group = "neurodegenerative disease group" if disease else "healthy control group"
    color = "#A34A0B" if disease else "#0E5C57"
    uncertain = lo < THRESHOLD < hi
    c1, c2 = st.columns([1.7, 1], gap="medium")
    with c1:
        st.markdown(f"""<div class="card"><p class="eyebrow">Prediction</p>
<p style="font-family:Newsreader,Georgia,serif;font-size:28px;line-height:1.25;margin:0 0 16px">
This walking pattern more closely resembles the <b style="color:{color}">{group}</b> in our training data.</p>
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><b>Probability of disease-group pattern</b>
<span><b style="font-size:24px">{p:.2f}</b> <span class="muted">(range {lo:.2f}–{hi:.2f})</span></span></div>
<div class="prob-track" role="img" aria-label="Probability {p:.2f}, range {lo:.2f} to {hi:.2f}, threshold {THRESHOLD:.2f}">
<div class="line"></div><div class="range" style="left:{lo*100}%;width:{(hi-lo)*100}%"></div>
<div class="thr"></div><div class="dot" style="left:{p*100}%"></div></div>
<div class="scale"><span>0 · healthy-like</span><span>threshold {THRESHOLD:.2f}</span><span>1 · disease-like</span></div>
<p class="muted" style="margin:14px 0 0"><b style="color:#17201E">What the range means:</b> we re-analysed this
recording many times, each time resampling its strides; 95% of those estimates fell in this range.
{"<b style='color:#A34A0B'>The range crosses the threshold, so this result is uncertain.</b>" if uncertain else ""}</p>
</div>""", unsafe_allow_html=True)
    with c2:
        m = info or {}
        def val(k, fmt="{}"):
            return fmt.format(m[k]) if k in m and m[k] is not None else "[pending]"
        st.markdown(f"""<div class="card"><p class="eyebrow">How reliable is the model?</p>
<p class="muted">Tested once on participants never used in training:</p>
<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px">
<div><div class="muted" style="font-size:13px">Patients detected</div><b style="font-size:22px">{val("sensitivity_pct", "{}%")}</b></div>
<div><div class="muted" style="font-size:13px">Controls cleared</div><b style="font-size:22px">{val("specificity_pct", "{}%")}</b></div>
<div><div class="muted" style="font-size:13px">AUROC (95% CI)</div><b>{val("auroc")} {val("auroc_ci", "({})") if info else ""}</b></div>
<div><div class="muted" style="font-size:13px">Model</div><b>{html.escape(val("model_name"))}</b></div></div>
</div>""", unsafe_allow_html=True)

    # --- feature comparison + ranking ----------------------------------------
    c3, c4 = st.columns([1.7, 1], gap="medium")
    with c3:
        rows = []
        for k in FEATURE_ORDER:
            name, desc, unit, d = FEATURE_INFO[k]
            r, v = ref[k], feats[k]
            band_l, band_r = pct(r["q25"], r["axis_min"], r["axis_max"]), pct(r["q75"], r["axis_min"], r["axis_max"])
            if pd.isna(v):
                dot, val_html = "", '<b class="muted">unavailable</b>'
            else:
                flag = v < r["q25"] or v > r["q75"]
                dc = "#A34A0B" if flag else "#17201E"
                dot = (f'<div class="dot" style="left:{pct(v, r["axis_min"], r["axis_max"])}%;'
                       f'background:{dc};box-shadow:0 0 0 1px {dc}"></div>')
                val_html = f'<b class="{"flag" if flag else ""}">{"⚠ " if flag else ""}{v:.{d}f}{unit}</b>'
            rows.append(f'<div class="feat-row"><div class="feat-name"><b>{name}</b><span>{desc}</span></div>'
                        f'<div class="track" aria-hidden="true"><div class="line"></div>'
                        f'<div class="band" style="left:{band_l}%;width:{max(band_r - band_l, 1)}%"></div>{dot}</div>'
                        f'<div class="feat-val">{val_html}<span>healthy {r["q25"]:.{d}f}–{r["q75"]:.{d}f}{unit}</span></div></div>')
        st.markdown('<div class="card"><h2 style="font-size:24px;margin:0">How this walk compares with healthy adults</h2>'
                    '<p class="muted">Shaded band: middle 50% of healthy controls · dot: this participant · '
                    '<span class="flag">⚠ outside healthy range</span></p>' + "".join(rows) + "</div>",
                    unsafe_allow_html=True)
    with c4:
        drivers = res["drivers"]
        top = max((abs(c) for _, c in drivers), default=1) or 1
        items = []
        for i, (k, c) in enumerate(drivers):
            w = abs(c) / top * 100
            direction = "→ disease" if c > 0 else "→ healthy"
            items.append(f'<div class="rank-row"><span class="rank-num {"top" if i < 3 else ""}">{i + 1}</span>'
                         f'<div class="rank-body"><div class="rank-head"><b>{FEATURE_INFO[k][0]}</b>'
                         f'<span class="muted" style="white-space:nowrap">{direction}</span></div>'
                         f'<div class="rank-bar"><div style="width:{w}%;background:{"#0E5C57" if i < 3 else "#7FAAA4"}"></div>'
                         f'</div></div></div>')
        st.markdown('<div class="card"><h2 style="font-size:24px;margin:0">What drove this result</h2>'
                    '<p class="muted">Features ranked by how strongly they pushed this prediction.</p>'
                    + "".join(items) + "</div>", unsafe_allow_html=True)

    # --- limitations ---------------------------------------------------------
    st.markdown("""<div class="card" style="background:#EEF2F1"><h3>Limitations of this model</h3><ul class="muted" style="font-size:15px">
<li>Trained on only 64 people (48 patients, 16 healthy controls) from one study.</li>
<li>Healthy controls were younger on average (median 35 vs 55.5 years), so age may influence results.</li>
<li>It cannot tell which disease is present, how severe it is, or rule disease out.</li>
<li>Not validated on other sensors, walking conditions or populations (e.g. children, people with injuries or other neurological conditions).</li>
</ul></div>""", unsafe_allow_html=True)

    _, b = st.columns([3, 1.3])
    b.button("Analyse another recording", on_click=go, args=("input",), use_container_width=True)


# ================================================================== ROUTER ==
chrome()
{"welcome": page_welcome, "input": page_input, "results": page_results}[st.session_state.page]()
