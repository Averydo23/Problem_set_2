SSIM916 — Problem Set 2
Sentiment Analysis of UK Fashion Brands
Machine Learning for Social Data Science
Student: Phuong Thuy Do · Student ID: 750081635
University: University of Exeter · MSc Business Analytics
GitHub: https://github.com/Averydo23/Problem_set_2

Overview
This project applies NLP techniques to real-world customer reviews scraped from Trustpilot across 12 UK fashion brands. The goal is to classify customer sentiment and identify complaint themes per brand, producing actionable business recommendations through competitor analysis.
Research Question:
How can automated sentiment analysis of customer reviews help UK fashion brands identify service gaps and enhance their competitive strategy?

Dataset
SourceTrustpilot (scraped via custom Python scraper)PeriodJanuary 2024 – March 2026Total reviews2,333 reviewsBrandsASOS, Boohoo, H&M, InTheStyle, Missguided, NastyGal, NewLook, Next, PrettyLittleThing, RiverIsland, SHEIN, ZaraReviews per brand~200 reviews
Sentiment Labels:
Star RatingLabelCount%4–5 ★Positive47520.4%3 ★Neutral642.7%1–2 ★Negative1,79476.9%

Models
ModelAccuracyMacro F1Negative F1Positive F1TF-IDF + LogReg (baseline)0.920.670.960.86DistilBERT (final)0.940.620.960.89
Why DistilBERT as final model?
Despite lower Macro F1, DistilBERT captures contextual meaning — understanding sarcasm, negation, and implicit sentiment that TF-IDF misses. This produces cleaner Negative classifications as input for BERTopic competitor analysis, resulting in more reliable per-brand recommendations.

BERTopic Complaint Themes
BERTopic was applied to all 1,794 Negative reviews, identifying 50 topics merged into 6 macro-themes:
Macro TopicDescriptionReviewsRefund & Customer ServiceUnresolved refund disputes + poor CS response524Poor Customer ServiceUnresponsive, unhelpful or hostile staff284Return Costs & BoycottCostly returns driving brand abandonment170Delivery IssuesLost parcels, delays, courier failures116Return & Order IssuesWrong items, order errors, unprocessed returns104Refund ComplaintsStandalone refund processing failures49

Key Findings

Refund & Customer Service is the dominant complaint across all brands (524 reviews)
Zara has the highest refund concentration — 85 reviews in a single topic
RiverIsland has the worst Customer Service score (67 classified complaints)
Missguided shows the highest Return Costs & Boycott signal — worsened post-SHEIN acquisition
Next is the best performer — only 10 classified complaint reviews, no dominant theme


Project Structure
Problem_set_2/
├── Problem_Set_2.ipynb    # Main notebook (Colab) — DistilBERT + BERTopic
├── reviews_raw.csv        # Scraped Trustpilot reviews dataset
├── scraper.py             # Trustpilot scraper
└── README.md

How to Reproduce
1. Scrape data
bashpip install requests pandas
python scraper.py
# Output: data/reviews_raw.csv
2. Run models (Google Colab)

Upload data/reviews_raw.csv to Google Drive
Open Problem_Set_2.ipynb in Google Colab
Runtime → Change runtime type → T4 GPU
Run all cells in order

Requirements
bashpip install transformers datasets sentence-transformers bertopic \
            pandas scikit-learn matplotlib seaborn
