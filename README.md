# NewsPulse — AI News Triage & Digest

ISOM5240 · Deep Learning Business Applications with Python · Fall 2026 · **Leon**

Paste a news article → get its **topic**, a **one-sentence digest**, and the **key entities**
in one click.

## Business context

A news aggregation platform ingests thousands of articles every day. Editors spend hours
tagging each article with a topic and writing short digests, and readers cannot scan
headlines fast enough. NewsPulse automates both steps.

## The three pipelines

| # | Pipeline | Model | Trained? | Output |
|---|---|---|---|---|
| P1 | `text-classification` | `<hf-user>/newspulse-topic-4` | **Fine-tuned** on ag_news | World / Sports / Business / Sci-Tech |
| P2 | `summarization` | `sshleifer/distilbart-cnn-12-6` | Pre-trained | One-sentence digest |
| P3 | `token-classification` | `dslim/bert-base-NER` | Pre-trained | People / orgs / locations |

## Repository contents

```
app.py                      Streamlit application (3 tabs)
requirements.txt            Pinned dependencies
demo_articles.json          12 hand-written demo articles (3 per topic)
01_finetune_newsclassifier.ipynb   Fine-tune DistilBERT on ag_news  → pushes model to HF Hub
02_pipeline_combined.ipynb         Combine P1 + P2 + P3
experimental_results.xlsx   Model comparison results
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Repository: `your-user/newspulse` · Branch `main` · Main file: `app.py`.
4. Deploy. If your model lives elsewhere, add a secret `NEWSPULSE_MODEL`
   with the value `<hf-user>/newspulse-topic-4`.

## Dataset

`ag_news` (HuggingFace canonical dataset · 120,000 train / 7,600 test · 4 topic classes).
The notebook uses a deliberately minimal balanced subset of **800 train / 200 test** (200 per
class, 1 epoch) so the whole fine-tune finishes in **1-2 minutes even on a CPU-only runtime**.

## Disclaimer

Built for academic purposes as part of the ISOM5240 course at HKUST. Any resemblance to a
commercial product is coincidental.
