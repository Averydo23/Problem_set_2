# SSIM916 — Problem Set 2
## Sentiment Analysis of UK Fashion Brands
### Machine Learning for Social Data Science

**Student:** Phuong Thuy Do · Student ID: 750081635  
**University:** University of Exeter · MSc Business Analytics  
**GitHub:** https://github.com/Averydo23/Problem_set_2

---

## Overview

This project applies NLP techniques to real-world customer reviews scraped from Trustpilot across 12 UK fashion brands. The goal is to classify customer sentiment and identify complaint themes per brand, producing actionable business recommendations through competitor analysis.

**Research Question:**  
*How can automated sentiment analysis of customer reviews help UK fashion brands identify service gaps and enhance their competitive strategy?*

---

## Dataset

| | |
|---|---|
| **Source** | Trustpilot (scraped through custom Python scraper) |
| **Period** | January 2024 – March 2026 |
| **Total reviews** | 2,333 reviews |
| **Brands** | ASOS, Boohoo, H&M, InTheStyle, Missguided, NastyGal, NewLook, Next, PrettyLittleThing, RiverIsland, SHEIN, Zara |
| **Reviews per brand** | ~200 reviews |

**Sentiment Labels:**

| Star Rating | Label | Count | % |
|---|---|---|---|
| 4–5 ★ | Positive | 475 | 20.4% |
| 3 ★ | Neutral | 64 | 2.7% |
| 1–2 ★ | Negative | 1,794 | 76.9% |

---

## Models

| Model | Accuracy | Macro F1 | Negative F1 | Positive F1 |
|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.92 | 0.67 | 0.96 | 0.86 |
| DistilBERT (final) | 0.94 | 0.62 | 0.96 | 0.89 |

**Why DistilBERT as final model?**  
Despite lower Macro F1, DistilBERT captures contextual meaning — understanding sarcasm, negation, and implicit sentiment that TF-IDF misses. This produces cleaner Negative classifications as input for BERTopic competitor analysis, resulting in more reliable per-brand recommendations.

---

## BERTopic Complaint Themes

BERTopic was applied to all 1,794 Negative reviews, identifying 50 topics merged into 6 macro-themes:

| Macro Topic | Description | Reviews |
|---|---|---|
| Refund & Customer Service | Unresolved refund disputes + poor CS response | 524 |
| Poor Customer Service | Unresponsive, unhelpful or hostile staff | 284 |
| Return Costs & Boycott | Costly returns driving brand abandonment | 170 |
| Delivery Issues | Lost parcels, delays, courier failures | 116 |
| Return & Order Issues | Wrong items, order errors, unprocessed returns | 104 |
| Refund Complaints | Standalone refund processing failures | 49 |

---

## Key Findings

- **Refund & Customer Service** is the dominant complaint across all brands (524 reviews)
- **Zara** has the highest refund concentration — 85 reviews in a single topic
- **RiverIsland** has the worst Customer Service score (67 classified complaints)
- **Missguided** shows the highest Return Costs & Boycott signal — worsened post-SHEIN acquisition
- **Next** is the best performer — only 10 classified complaint reviews, no dominant theme

---

## Project Structure

```
Problem_set_2/
├── Problem_Set_2.ipynb    # Main notebook (Colab) — DistilBERT + BERTopic
├── reviews_raw.csv    # Scraped Trustpilot reviews
├── scraper.py             # Trustpilot scraper
└── README.md
```

---

## How to Reproduce

### 1. Scrape data
```bash
pip install requests pandas
python scraper.py
# Output: data/reviews_raw.csv
```

### 2. Run models (Google Colab)
1. Upload `data/reviews_raw.csv` to Google Drive
2. Open `Problem_Set_2.ipynb` in Google Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells in order

### Requirements
```bash
pip install transformers datasets sentence-transformers bertopic \
            pandas scikit-learn matplotlib seaborn
```
