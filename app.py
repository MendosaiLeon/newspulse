"""
NewsPulse — AI News Triage & Digest
ISOM5240 Final Project (Fall 2026) · Leon

Three HuggingFace pipelines:
  P1 (fine-tuned)   text-classification    -> news topic (World / Sports / Business / Sci-Tech)
  P2 (pre-trained)  summarization          -> one-sentence digest
  P3 (pre-trained)  token-classification   -> people / organisations / locations
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="NewsPulse — AI News Triage", page_icon="📰", layout="wide")

TOPIC_MODEL = os.environ.get(
    "NEWSPULSE_MODEL",
    "MendosaiLeonye/newspulse-topic-4",   # ⚠ change to your HF username if different
)
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
NER_MODEL = "dslim/bert-base-NER"
TOPICS = ["World", "Sports", "Business", "Sci/Tech"]


# ─────────────────────── CACHED LOADERS ───────────────────────
@st.cache_resource(show_spinner="📰 Loading 3 pipelines…")
def load_pipelines():
    from transformers import pipeline
    return {
        "topic":  pipeline("text-classification", model=TOPIC_MODEL, top_k=1),
        "sum":    pipeline("summarization", model=SUMMARY_MODEL),
        "ner":    pipeline("token-classification", model=NER_MODEL,
                           aggregation_strategy="simple"),
    }


@st.cache_data
def load_demo():
    p = Path(__file__).parent / "demo_articles.json"
    data = json.loads(p.read_text())
    rows = []
    for topic, texts in data.items():
        for t in texts:
            rows.append({"topic": topic, "text": t})
    return pd.DataFrame(rows)


pipes = load_pipelines()
demo_df = load_demo()


def analyze(text, max_len=40, min_len=10):
    """Run all three pipelines on one piece of text."""
    t = pipes["topic"](text)[0]
    s = pipes["sum"](text, max_length=max_len, min_length=min_len, do_sample=False)[0]
    ents = pipes["ner"](text)
    return {
        "topic": t["label"],
        "score": t["score"],
        "summary": s["summary_text"],
        "entities": [(e["word"], e["entity_group"]) for e in ents],
    }


# ─────────────────────────── HEADER ───────────────────────────
st.title("📰 NewsPulse — AI News Triage & Digest")
st.markdown(
    "Paste any news article and get, in one click: its **topic**, a **one-sentence digest**, "
    "and the **key people / organisations / locations** mentioned. "
    "Powered by three HuggingFace pipelines."
)
st.caption(f"P1 `{TOPIC_MODEL}` (fine-tuned) · P2 `{SUMMARY_MODEL}` · P3 `{NER_MODEL}`")

tab1, tab2, tab3 = st.tabs(["🔍 Single article", "📊 Batch dashboard", "⚖️ Compare topics"])

# ══════════════ TAB 1 ══════════════
with tab1:
    st.subheader("Analyse one article")
    txt = st.text_area("Article text", height=150,
                       placeholder="Shares of the regional airline jumped 7% after it reported …")
    if st.button("Triage it", type="primary") and txt.strip():
        with st.spinner("Running 3 pipelines…"):
            r = analyze(txt)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Topic", r["topic"], f"{r['score']*100:.1f}% confidence")
            st.markdown("**Digest**")
            st.info(r["summary"])
        with c2:
            st.markdown("**Entities**")
            if r["entities"]:
                st.dataframe(pd.DataFrame(r["entities"], columns=["entity", "type"]),
                             hide_index=True, use_container_width=True)
            else:
                st.write("(none detected)")

# ══════════════ TAB 2 ══════════════
with tab2:
    st.subheader("Batch triage")
    mode = st.radio("Data source", ["Use 12 demo articles", "Upload my CSV"])
    if mode == "Use 12 demo articles":
        df = demo_df
    else:
        up = st.file_uploader("CSV with columns: topic, text", type=["csv"])
        if up is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        df = pd.read_csv(up)
        if not {"topic", "text"}.issubset(df.columns):
            st.error("CSV must contain columns: topic, text")
            st.stop()

    if st.button("Run batch", type="primary"):
        rows, prog = [], st.progress(0.0, text="Triaging…")
        for i, r in df.iterrows():
            a = analyze(r["text"])
            rows.append({"expected": r["topic"], "predicted": a["topic"],
                         "confidence": round(a["score"], 3),
                         "ok": "✓" if a["topic"] == r["topic"] else "✗",
                         "digest": a["summary"],
                         "entities": ", ".join(w for w, _ in a["entities"]) or "-"})
            prog.progress((i + 1) / len(df))
        prog.empty()
        res = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("Articles", len(res))
        c2.metric("Accuracy", f"{(res['ok']=='✓').mean()*100:.0f}%")
        c3.metric("Avg confidence", f"{res['confidence'].mean()*100:.1f}%")

        st.dataframe(res, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Topic distribution (truth)**")
            st.plotly_chart(px.pie(res, names="expected", hole=0.4),
                            use_container_width=True)
        with col_b:
            st.markdown("**Predicted vs actual**")
            st.dataframe(pd.crosstab(res["expected"], res["predicted"]),
                         use_container_width=True)

# ══════════════ TAB 3 ══════════════
with tab3:
    st.subheader("Compare two topics side by side")
    ca, cb = st.columns(2)
    ta = ca.selectbox("Topic A", TOPICS, index=2)
    tb = cb.selectbox("Topic B", TOPICS, index=1)
    if ta == tb:
        st.warning("Please pick two different topics.")
    else:
        cols = st.columns(2)
        for col, topic in zip(cols, [ta, tb]):
            with col:
                st.markdown(f"### {topic}")
                subset = demo_df[demo_df["topic"] == topic]
                correct, total, conf = 0, 0, 0.0
                for _, r in subset.iterrows():
                    a = analyze(r["text"])
                    total += 1
                    conf += a["score"]
                    if a["topic"] == topic:
                        correct += 1
                    tag = "✓" if a["topic"] == topic else f"✗ → {a['topic']}"
                    with st.expander(f"{a['score']*100:.0f}%  {tag}"):
                        st.write(r["text"])
                        st.caption(a["summary"])
                if total:
                    st.metric(f"{topic} recall", f"{correct/total*100:.0f}%",
                              delta=f"{conf/total*100:.0f}% avg conf")

st.markdown("---")
st.caption("ISOM5240 Fall 2026 · Leon · Built with 🤗 Transformers + Streamlit")
