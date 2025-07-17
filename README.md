# DeFi Credit Scoring Engine

## 1. Project Overview

This project provides a robust, single-script solution for generating on-chain credit scores for DeFi wallets based on their historical transaction data from the Aave V2 protocol. The objective is to assign a numerical score between 0 and 1000 to each unique wallet, where a higher score signifies responsible, credit-worthy behavior and a lower score indicates potentially risky, anomalous, or programmatic (bot-like) activity.

The model is designed to be entirely **unsupervised**, meaning it does not require pre-labeled data to function. It derives insights directly from the patterns and structures inherent in the raw transaction data, making it a highly practical tool for risk assessment in a decentralized environment where ground truth is often unavailable.

---

## 2. Architectural Design & Methodology

The scoring engine employs a hybrid unsupervised learning model. This approach was chosen because it combines the strengths of several techniques to produce a nuanced and explainable score. A simple, single-algorithm approach would fail to capture the multi-faceted nature of wallet behavior.

The architecture is built on three core analytical pillars:

1.  **Behavioral Segmentation (Clustering):** We use the **K-Means Clustering** algorithm to group wallets into distinct behavioral archetypes based on their overall transaction profile. This answers the question: "What type of user is this?" (e.g., a "High-Volume Trader," a "Conservative Lender," or an "Infrequent User").

2.  **Anomaly Detection:** We use the **Isolation Forest** algorithm to identify wallets that exhibit unusual or outlier patterns. This is crucial for flagging potentially high-risk actors whose behavior deviates significantly from the norm. This answers the question: "Is this user's activity normal?"

3.  **Heuristic Rule-Based Adjustments:** We apply a set of transparent, domain-specific rules to fine-tune the score. This layer injects expert knowledge into the model, directly rewarding actions indicative of credit-worthiness (e.g., consistent repayments) and penalizing clear risk signals (e.g., liquidations).

The final credit score is a composite of the outputs from these three pillars, creating a balanced and comprehensive assessment.



---

## 3. Data Processing & Scoring Flow

The end-to-end process is executed in a sequential pipeline, encapsulated within the main script.

### Stage 1: Data Ingestion and Preparation (`load_transactions`)

* **Input:** The pipeline begins by loading raw, transaction-level data from the `user-wallet-transactions.json` file.
* **Parsing:** Each JSON record is parsed to extract key fields such as wallet address, action type, timestamp, asset, and amount.
* **Normalization & Cleaning:**
    * Token amounts are normalized to account for varying decimal conventions (e.g., USDC has 6 decimals, WETH has 18). This is critical for comparing transaction values accurately.
    * A `normalizedAmountUSD` value is calculated for each transaction.
    * The data is cleaned to remove null records and transactions with zero value, ensuring data quality for the subsequent stages.

### Stage 2: Feature Engineering (`feature_engineering`)

This is the most critical transformation step. For each unique wallet, a comprehensive set of over 30 features is engineered to create a detailed behavioral fingerprint. These features are grouped into several categories:

* **Volume & Value Metrics:** `total_volume_usd`, `avg_tx_size_usd`, `max_tx_size_usd`, `volume_std`, etc.
* **Frequency & Timing:** `total_transactions`, `days_active`, `tx_frequency`, `avg_time_between_tx`, `burst_activity_ratio`.
* **Protocol Interaction:** `deposit_ratio`, `borrow_ratio`, `repay_ratio`, `liquidation_ratio`, `action_diversity`.
* **Behavioral Indicators:** `is_borrower`, `is_lender`, `borrow_repay_balance`.
* **Asset Profile:** `unique_assets`, `primary_asset_dominance`, `asset_entropy`.
* **Pattern Detection:** `round_amount_ratio` (to detect bot-like precision), `hourly_pattern_entropy` (to measure activity randomness).

### Stage 3: Machine Learning & Score Composition (`create_ml_scores`)

* **Scaling:** All engineered features are scaled using a `QuantileTransformer`. This method is robust to outliers and transforms the feature distributions to be more suitable for the ML models.
* **Dimensionality Reduction:** **Principal Component Analysis (PCA)** is applied to reduce the dimensionality of the feature space while retaining 95% of the variance. This helps improve the performance and stability of the clustering algorithm.
* **Clustering (Base Score):** The PCA-transformed data is fed into a **K-Means** model. Each wallet is assigned to a cluster, and a "goodness" score is calculated for each cluster based on its members' average volume, frequency, and diversity. This cluster-level score is then scaled to a range of **300-700** to serve as the wallet's base credit score.
* **Anomaly Detection (Risk Adjustment):** An **Isolation Forest** model is trained on the scaled feature set. It calculates a `decision_function` score for each wallet, which indicates how "normal" or "anomalous" its behavior is. This score is scaled to a range of **-100 to +100** and acts as a direct adjustment to the base score.
* **Heuristic Overlays:** A final layer of adjustments is made based on specific feature values. For example, a high `liquidation_ratio` incurs a significant penalty, while a positive `borrow_repay_balance` provides a bonus.
* **Final Score Calculation:** The final score is computed as:
    `Score = Base Score + Anomaly Adjustment + Heuristic Impact`
    The result is then clamped to ensure it falls within the **0-1000** range.

### Stage 4: Reporting and Analysis

The script concludes by generating a comprehensive analysis saved in `analysis.md` and `credit_score_analysis.png`. This report provides a statistical summary of the scores, cohort analysis for different score categories, and identifies the key features that influence the credit scores, ensuring full transparency of the model's output.