import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.stats import entropy


def _normalize_amount(row):
    """Normalize token amounts based on common decimal patterns."""
    amount = row['amount']
    symbol = str(row['assetSymbol']).upper()

    decimal_map = {
        'USDC': 6,
        'USDT': 6,
        'DAI': 18,
        'WETH': 18,
        'WMATIC': 18,
        'WBTC': 8,
        'USDC.E': 6
    }
    decimals = decimal_map.get(symbol, 18)

    if amount is None or pd.isna(amount):
        return 0

    return amount / (10**decimals)

def load_transactions(file_path: str) -> pd.DataFrame:
    """Load data from JSON file"""
    print(f"Loading transactions from {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        transactions = []
        for record in data:
            tx = {
                'wallet': record.get('userWallet', ''),
                'network': record.get('network', ''),
                'protocol': record.get('protocol', ''),
                'txHash': record.get('txHash', ''),
                'timestamp': record.get('timestamp', ''),
                'blockNumber': record.get('blockNumber', ''),
                'action': record.get('action', ''),
            }
            action_data = record.get('actionData', {})
            tx.update({
                'amount': action_data.get('amount', 0),
                'assetSymbol': action_data.get('assetSymbol', ''),
                'assetPriceUSD': action_data.get('assetPriceUSD', 0),
                'poolId': action_data.get('poolId', ''),
                'userId': action_data.get('userId', ''),
                'type': action_data.get('type', ''),
            })
            if 'createdAt' in record and isinstance(record['createdAt'], dict):
                if '$date' in record['createdAt']:
                    tx['createdAt'] = record['createdAt']['$date']
            transactions.append(tx)
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')
        df['blockNumber'] = pd.to_numeric(df['blockNumber'], errors='coerce')
        df['amountUSD'] = df['amount'] * df['assetPriceUSD']
        df['normalizedAmount'] = df.apply(_normalize_amount, axis=1)
        df['normalizedAmountUSD'] = df['normalizedAmount'] * df['assetPriceUSD']
        df = df[df['wallet'].notna() & (df['wallet'] != '')]
        df = df[df['timestamp'].notna()]
        df = df[df['amount'] > 0]

        if 'normalizedAmountUSD' in df.columns:
            df = df[df['normalizedAmountUSD'].notna() & (df['normalizedAmountUSD'] > 0)]

        return df
    except Exception as e:
        print(f'Error loading transactions: {e}')
        return pd.DataFrame()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''Feature engineering'''
    print('Performing feature engineering')
    features = []
    for wallet in df['wallet'].unique():
        wallet_df = df[df['wallet'] == wallet].copy()
        wallet_df = wallet_df.sort_values('timestamp')
        _features = {'wallet': wallet}

        amount_usd = wallet_df['normalizedAmountUSD'].dropna()
        amount_usd = amount_usd[amount_usd > 0]

        if len(amount_usd) > 0:
            _features['total_volume_usd'] = amount_usd.sum()
            _features['avg_tx_size_usd'] = amount_usd.mean()
            _features['max_tx_size_usd'] = amount_usd.max()
            _features['min_tx_size_usd'] = amount_usd.min()
            _features['median_tx_size_usd'] = amount_usd.median()
            _features['volume_std'] = amount_usd.std()
            _features['volume_coefficient_variation'] = _features['volume_std'] / max(_features['avg_tx_size_usd'], 1e-9)
            _features['volume_skewness'] = amount_usd.skew()
            _features['volume_kurtosis'] = amount_usd.kurtosis()
            sorted_amounts = amount_usd.sort_values(ascending=False)
            top_10_percent_count = max(1, int(len(sorted_amounts) * 0.1))
            _features['top_10_percent_volume_ratio'] = sorted_amounts.head(top_10_percent_count).sum() / sorted_amounts.sum()
        else:
            _features.update({
                'total_volume_usd': 0, 'avg_tx_size_usd': 0, 'max_tx_size_usd': 0, 'min_tx_size_usd': 0, 'median_tx_size_usd': 0,
                'volume_std': 0, 'volume_coefficient_variation': 0, 'volume_skewness': 0, 'volume_kurtosis': 0,
                'top_10_percent_volume_ratio': 0
            })

        _features['total_transactions'] = len(wallet_df)

        if not wallet_df.empty:
            _features['days_active'] = wallet_df['timestamp'].dt.date.nunique()
        else:
            _features['days_active'] = 0

        _features['tx_frequency'] = _features['total_transactions'] / max(_features['days_active'], 1)

        if len(wallet_df) > 1:
            time_diffs = wallet_df['timestamp'].diff().dt.total_seconds().dropna()
            _features['avg_time_between_tx'] = time_diffs.mean()
            _features['time_regularity'] = time_diffs.std() / max(time_diffs.mean(), 1e-9)
            _features['min_time_gap'] = time_diffs.min()
            _features['max_time_gap'] = time_diffs.max()
            short_gaps = (time_diffs < 3600).sum()
            _features['burst_activity_ratio'] = short_gaps / len(time_diffs) if len(time_diffs) > 0 else 0
        else:
            _features.update({
                'avg_time_between_tx': 0, 'time_regularity': 0, 'min_time_gap': 0,
                'max_time_gap': 0, 'burst_activity_ratio': 0
            })

        #Action composition
        action_counts = wallet_df['action'].value_counts()
        total_actions = len(wallet_df)

        _features['deposit_ratio'] = action_counts.get('deposit', 0) / total_actions if total_actions > 0 else 0
        _features['borrow_ratio'] = action_counts.get('borrow', 0) / total_actions if total_actions > 0 else 0
        _features['repay_ratio'] = action_counts.get('repay', 0) / total_actions if total_actions > 0 else 0
        _features['liquidation_ratio'] = action_counts.get('liquidationcall', 0) / total_actions if total_actions > 0 else 0
        _features['withdraw_ratio'] = action_counts.get('redeemunderlying', 0) / total_actions if total_actions > 0 else 0

        # Behavioral
        _features['action_diversity'] = wallet_df['action'].nunique()
        _features['is_borrower'] = 1 if action_counts.get('borrow', 0) > 0 else 0
        _features['is_lender'] = 1 if action_counts.get('deposit', 0) > 0 else 0
        _features['borrow_repay_balance'] = (action_counts.get('repay', 0) - action_counts.get('borrow', 0)) / max(action_counts.get('borrow', 0), 1)

        #Asset behaviour
        _features['unique_assets'] = wallet_df['assetSymbol'].nunique()
        if _features['unique_assets'] > 0:
            asset_counts = wallet_df['assetSymbol'].value_counts()
            _features['primary_asset_dominance'] = asset_counts.iloc[0] / len(wallet_df)
            asset_probs = asset_counts / len(wallet_df)
            _features['asset_entropy'] = entropy(asset_probs, base=2)
        else:
            _features.update({'primary_asset_dominance': 1, 'asset_entropy': 0})

        coins = ['USDC', 'USDT', 'WETH', 'DAI', 'USDC.e', 'WMATIC', 'WBTC']
        coin_txs = wallet_df[wallet_df['assetSymbol'].isin(coins)]
        _features['coin_preference'] = len(coin_txs) / len(wallet_df) if len(wallet_df) > 0 else 0

        #Pattern detection
        if len(amount_usd) > 0:
            round_amt = amount_usd % 1000 == 0
            _features['round_amount_ratio'] = round_amt.mean()
            unique_amounts = amount_usd.nunique()
            _features['amount_repetition'] = 1 - (unique_amounts / len(amount_usd))
        else:
            _features.update({'round_amount_ratio': 0, 'amount_repetition': 0})

        # Weekly activity pattern
        if len(wallet_df) > 0:
            weekday_counts = wallet_df['timestamp'].dt.dayofweek.value_counts()
            weekday_probs = weekday_counts.reindex(range(7), fill_value=0) / len(wallet_df)
            weekday_entropy = entropy(weekday_probs, base=2)
            _features['weekly_pattern_entropy'] = weekday_entropy / (np.log2(7) if np.log2(7) > 0 else 1)
        else:
            _features['weekly_pattern_entropy'] = 0

        # Hourly activity patterns
        if len(wallet_df) > 0:
            hour_counts = wallet_df['timestamp'].dt.hour.value_counts()
            hour_probs = hour_counts.reindex(range(24), fill_value=0) / len(wallet_df)
            hour_entropy = entropy(hour_probs, base=2)
            _features['hourly_pattern_entropy'] = hour_entropy / (np.log2(24) if np.log2(24) > 0 else 1)
        else:
            _features['hourly_pattern_entropy'] = 0

        # Risk indicators
        if len(wallet_df) > 1:
            time_diffs_for_rapid = wallet_df['timestamp'].diff().dt.total_seconds().dropna()
            rapid_txs = (time_diffs_for_rapid < 300).sum()
            _features['rapid_transaction_ratio'] = rapid_txs / len(time_diffs_for_rapid) if len(time_diffs_for_rapid) > 0 else 0
        else:
            _features['rapid_transaction_ratio'] = 0

        recent_cutoff = wallet_df['timestamp'].max() - timedelta(days=30) if not wallet_df.empty else datetime.min
        recent_activity = wallet_df[wallet_df['timestamp'] >= recent_cutoff]
        _features['recent_activity_ratio'] = len(recent_activity) / len(wallet_df) if len(wallet_df) > 0 else 0

        features.append(_features)

    feature_df = pd.DataFrame(features)
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0)
    feature_df[numeric_columns] = feature_df[numeric_columns].replace([np.inf, -np.inf], 0)

    return feature_df


def create_ml_scores(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, PCA, Dict[str, float]]:
    """Create credit scores"""
    print("Creating credit scores...")

    # Prepare features
    feature_cols = [col for col in feature_df.columns if col != 'wallet']
    X = feature_df[feature_cols]

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Dimensionality reduction
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Cluster analysis to find behavioral groups
    clustering_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = clustering_model.fit_predict(X_pca)
    feature_df['cluster'] = cluster_labels

    # Anomaly detection for risk assessment
    anomaly_detector = IsolationForest(contamination=0.08, random_state=42, n_estimators=200)

    anomaly_scores_raw = anomaly_detector.fit_predict(X_scaled)
    anomaly_decision_scores = anomaly_detector.decision_function(X_scaled)

    # Correctly scale anomaly scores: lower scores (more anomalous) get lower-end range values (-100)
    anomaly_scaler = MinMaxScaler(feature_range=(-100, 100))
    scaled_anomaly_contributions = anomaly_scaler.fit_transform(anomaly_decision_scores.reshape(-1, 1)).flatten()

    # Composite behavioral score
    behavioral_scores = []

    cluster_base_scores = np.zeros(clustering_model.n_clusters)
    for c in range(clustering_model.n_clusters):
        cluster_mask = (cluster_labels == c)
        if cluster_mask.sum() > 0:
            cluster_data = X_scaled.loc[cluster_mask]

            avg_volume = cluster_data['total_volume_usd'].mean()
            avg_tx_freq = cluster_data['tx_frequency'].mean()
            avg_asset_diversity = cluster_data['asset_entropy'].mean()

            cluster_goodness = (
                avg_volume * 0.1
                + avg_tx_freq * 50
                + avg_asset_diversity * 100
            )
            cluster_base_scores[c] = cluster_goodness

    if cluster_base_scores.max() - cluster_base_scores.min() > 1e-9:
        min_score_range = 300
        max_score_range = 700
        normalized_cluster_scores = (cluster_base_scores - cluster_base_scores.min()) / \
                                      (cluster_base_scores.max() - cluster_base_scores.min()) * \
                                      (max_score_range - min_score_range) + min_score_range
    else:
        normalized_cluster_scores = np.full(clustering_model.n_clusters, 500.0)

    for i in range(len(feature_df)):
        current_row = feature_df.iloc[i]

        cluster_idx = cluster_labels[i]
        base_score = normalized_cluster_scores[cluster_idx]

        anomaly_adjustment = scaled_anomaly_contributions[i]

        behavioral_impact = 0

        # Positive behaviors
        if 'is_lender' in current_row: behavioral_impact += current_row['is_lender'] * 50
        if 'is_borrower' in current_row: behavioral_impact += current_row['is_borrower'] * 50
        if 'borrow_repay_balance' in current_row: behavioral_impact += (current_row['borrow_repay_balance'] * 50).clip(0, 50)
        if 'coin_preference' in current_row: behavioral_impact += (current_row['coin_preference'] * 100).clip(0, 100)
        if 'asset_entropy' in current_row: behavioral_impact += (current_row['asset_entropy'] * 20).clip(0, 50)

        # Negative behaviors
        if 'liquidation_ratio' in current_row: behavioral_impact -= (current_row['liquidation_ratio'] * 500).clip(0, 200)
        if 'round_amount_ratio' in current_row: behavioral_impact -= (current_row['round_amount_ratio'] * 100).clip(0, 100)
        if 'burst_activity_ratio' in current_row: behavioral_impact -= (current_row['burst_activity_ratio'] * 150).clip(0, 150)
        if 'time_regularity' in current_row: behavioral_impact -= (current_row['time_regularity'] * 50).clip(0, 50)

        if 'top_10_percent_volume_ratio' in current_row: behavioral_impact -= (current_row['top_10_percent_volume_ratio'] * 50).clip(0, 50)

        # Final score
        final_score = base_score + anomaly_adjustment + behavioral_impact
        final_score = max(0, min(1000, final_score))

        behavioral_scores.append(final_score)

    feature_df['credit_score'] = behavioral_scores

    feature_importance = dict(zip(feature_cols, np.abs(pca.components_[0])))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return feature_df, pca, feature_importance


def categorize_score(score: float) -> str:
    """Categorize credit score into risk levels."""
    if score >= 800:
        return 'Excellent'
    elif score >= 650:
        return 'Good'
    elif score >= 500:
        return 'Fair'
    elif score >= 350:
        return 'Poor'
    else:
        return 'Very Poor'

def generate_results(feature_df: pd.DataFrame) -> pd.DataFrame:
    '''Final results'''
    results = feature_df[['wallet', 'credit_score']].copy()
    results['credit_score'] = results['credit_score'].round().astype(int)
    results['score_category'] = results['credit_score'].apply(categorize_score)

    for col in ['is_borrower', 'is_lender', 'total_volume_usd', 'days_active', 'liquidation_ratio', 'tx_frequency', 'asset_entropy']:
        if col in feature_df.columns:
            results[col] = feature_df[col]
        else:
            results[col] = np.nan

    results['total_volume_usd'] = results['total_volume_usd'].round(2)
    results['liquidation_ratio'] = results['liquidation_ratio'].round(4)

    return results.sort_values('credit_score', ascending=False)

def generate_report(results: pd.DataFrame, pca: PCA, feature_importance: dict) -> str:
    """Generate analysis report"""
    report = []
    report.append(f'Total Wallets Analyzed: {len(results)}')
    report.append("")
    report.append("SCORE DISTRIBUTION:")
    score_dist = results['score_category'].value_counts()
    for category in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
        count = score_dist.get(category, 0)
        percentage = (count / len(results)) * 100
        report.append(f" {category:<12}: {count:4d} ({percentage:5.1f}%)")
    report.append("")
    report.append("SCORE STATISTICS:")
    report.append(f" Mean Score      : {results['credit_score'].mean():.1f}")
    report.append(f" Median Score    : {results['credit_score'].median():.1f}")
    report.append(f" Std Deviation   : {results['credit_score'].std():.1f}")
    report.append(f" Min Score       : {results['credit_score'].min()}")
    report.append(f" Max Score       : {results['credit_score'].max()}")
    report.append("")
    high_scorers = results[results['credit_score'] >= 650]
    if len(high_scorers) > 0:
        report.append(f"High Performers (650+): {len(high_scorers)} wallets")
        report.append(f" - Average volume: ${high_scorers['total_volume_usd'].mean():,.2f}")
        report.append(f" - Lenders         : {(high_scorers['is_lender'].sum()/len(high_scorers)*100):.1f}%")
        report.append(f" - Borrowers       : {(high_scorers['is_borrower'].sum()/len(high_scorers)*100):.1f}%")
        report.append(f" - Average Tx Frequency: {high_scorers['tx_frequency'].mean():.2f}")
        report.append(f" - Average Asset Entropy: {high_scorers['asset_entropy'].mean():.2f}")
    low_scorers = results[results['credit_score'] < 350]
    if len(low_scorers) > 0:
        report.append(f"Low Performers (<350): {len(low_scorers)} wallets")
        report.append(f" - Average volume: ${low_scorers['total_volume_usd'].mean():,.2f}")
        report.append(f" - Average liquidation ratio: {low_scorers['liquidation_ratio'].mean():.4f}")
        report.append(f" - Average Tx Frequency: {low_scorers['tx_frequency'].mean():.2f}")
        report.append(f" - Average Asset Entropy: {low_scorers['asset_entropy'].mean():.2f}")
    report.append("")
    report.append("TOP FEATURE IMPORTANCE (from PCA):")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        report.append(f" {i+1:2d}. {feature:<25}: {importance:.4f}")
    report.append("")
    report.append("PCA ANALYSIS:")
    report.append(f" Components retained     : {pca.n_components_}")
    report.append(f" Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    return "\n".join(report)

def create_visualizations(results: pd.DataFrame):
    seaborn.set_style('whitegrid')
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Score distribution histogram
    axes[0, 0].hist(results['credit_score'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Credit Score Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Credit Score', fontsize=12)
    axes[0, 0].set_ylabel('Number of Wallets', fontsize=12)

    # Score ranges distribution
    score_ranges = pd.cut(results['credit_score'], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                            labels=['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900', '900-1000'],
                            right=False)
    range_counts = score_ranges.value_counts().sort_index()
    axes[0, 1].bar(range_counts.index, range_counts.values, color='lightcoral')
    axes[0, 1].set_title('Score Distribution by Ranges', fontsize=14)
    axes[0, 1].set_xlabel('Score Range', fontsize=12)
    axes[0, 1].set_ylabel('Number of Wallets', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Volume vs Score scatter
    axes[1, 0].scatter(results['total_volume_usd'], results['credit_score'], alpha=0.6, color='green')
    axes[1, 0].set_title('Total Volume USD vs Credit Score', fontsize=14)
    axes[1, 0].set_xlabel('Total Volume USD (Log Scale)', fontsize=12)
    axes[1, 0].set_ylabel('Credit Score', fontsize=12)
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, which="both", ls="--", c='0.7')

    # Category distribution pie chart
    category_counts = results['score_category'].value_counts()
    axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=seaborn.color_palette("pastel"))
    axes[1, 1].set_title('Score Category Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'credit_score_analysis.png'")
    return range_counts

def analyze_wallet_behavior(results: pd.DataFrame, feature_df: pd.DataFrame):
    """Analyze behavior patterns of different score ranges"""
    analysis = {}

    # Define score ranges
    ranges = {
        'Very Low (0-200)': (0, 200),
        'Low (200-400)': (200, 400),
        'Medium (400-600)': (400, 600),
        'High (600-800)': (600, 800),
        'Very High (800-1000)': (800, 1000)
    }

    for range_name, (min_score, max_score) in ranges.items():
        range_wallets = results[(results['credit_score'] >= min_score) & (results['credit_score'] < max_score)]

        if len(range_wallets) > 0:
            # Get corresponding feature data
            range_features = feature_df[feature_df['wallet'].isin(range_wallets['wallet'])]

            analysis[range_name] = {
                'count': len(range_wallets),
                'avg_volume': range_features['total_volume_usd'].mean(),
                'avg_transactions': range_features['total_transactions'].mean(),
                'avg_days_active': range_features['days_active'].mean(),
                'liquidation_ratio': range_features['liquidation_ratio'].mean(),
                'deposit_activity_ratio': range_features['deposit_ratio'].mean(),
                'borrow_activity_ratio': range_features['borrow_ratio'].mean(),
                'asset_diversity': range_features['unique_assets'].mean(),
                'burst_activity_ratio': range_features['burst_activity_ratio'].mean(),
                'lenders_percentage': (range_features['is_lender'].sum() / len(range_features)) * 100,
                'borrowers_percentage': (range_features['is_borrower'].sum() / len(range_features)) * 100
            }
        else:
            analysis[range_name] = {
                'count': 0, 'avg_volume': 0, 'avg_transactions': 0, 'avg_days_active': 0, 'liquidation_ratio': 0,
                'deposit_activity_ratio': 0, 'borrow_activity_ratio': 0, 'asset_diversity': 0,
                'burst_activity_ratio': 0, 'lenders_percentage': 0, 'borrowers_percentage': 0
            }

    return analysis

def create_analysis_markdown(results: pd.DataFrame, feature_df: pd.DataFrame, report: str,
                             behavior_analysis: dict, range_counts: pd.Series):
    """Create detailed analysis markdown file"""

    with open('analysis.md', 'w') as f:
        f.write("# DeFi Credit Score Analysis Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"This report analyzes the credit scores of {len(results)} unique wallet addresses ")
        f.write("based on their DeFi transaction behavior, lending/borrowing patterns, and risk indicators.\n\n")

        f.write("## Key Findings\n\n")
        f.write(f"- **Average Credit Score:** {results['credit_score'].mean():.1f}\n")
        f.write(f"- **Median Credit Score:** {results['credit_score'].median():.1f}\n")
        f.write(f"- **Score Range:** {results['credit_score'].min()} - {results['credit_score'].max()}\n\n")

        f.write("## Score Distribution Analysis\n\n")
        f.write("### Distribution by Score Ranges\n\n")
        f.write("| Score Range | Count | Percentage |\n")
        f.write("|-------------|-------|------------|\n")
        for range_name, count in range_counts.items():
            percentage = (count / len(results)) * 100
            f.write(f"| {range_name} | {count} | {percentage:.1f}% |\n")
        f.write("\n")

        f.write("### Distribution by Risk Categories\n\n")
        score_dist = results['score_category'].value_counts()
        f.write("| Category      | Count | Percentage |\n")
        f.write("|---------------|-------|------------|\n")
        for category in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
            count = score_dist.get(category, 0)
            percentage = (count / len(results)) * 100
            f.write(f"| {category:<13} | {count:5} | {percentage:10.1f}% |\n")
        f.write("\n")

        f.write("## Behavioral Analysis by Score Range\n\n")

        for range_name, metrics in behavior_analysis.items():
            f.write(f"### {range_name}\n\n")
            f.write(f"**Wallet Count:** {metrics['count']}\n\n")

            if metrics['count'] > 0:
                f.write("**Key Characteristics:**\n")
                f.write(f"- Average Trading Volume: ${metrics['avg_volume']:,.2f}\n")
                f.write(f"- Average Transactions: {metrics['avg_transactions']:.1f}\n")
                f.write(f"- Average Days Active: {metrics['avg_days_active']:.1f}\n")
                f.write(f"- Liquidation Ratio: {metrics['liquidation_ratio']:.4f}\n")
                f.write(f"- Deposit Activity: {metrics['deposit_activity_ratio']:.2f}\n")
                f.write(f"- Borrow Activity: {metrics['borrow_activity_ratio']:.2f}\n")
                f.write(f"- Asset Diversity: {metrics['asset_diversity']:.1f}\n")
                f.write(f"- Burst Activity Ratio: {metrics['burst_activity_ratio']:.4f}\n")
                f.write(f"- Lenders: {metrics['lenders_percentage']:.1f}%\n")
                f.write(f"- Borrowers: {metrics['borrowers_percentage']:.1f}%\n\n")
            else:
                f.write("No wallets in this range.\n\n")

        f.write("## Detailed Statistical Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n\n")

        f.write("## High-Risk Patterns Identified\n\n")
        high_risk = results[results['credit_score'] < 350]
        if len(high_risk) > 0:
            f.write(f"**{len(high_risk)} wallets** identified as high-risk (score < 350):\n\n")
            f.write("**Common Risk Factors (Averages for this group):**\n")
            high_risk_features = feature_df[feature_df['wallet'].isin(high_risk['wallet'])]

            f.write(f"- Average Liquidation Ratio: {high_risk_features['liquidation_ratio'].mean():.4f}\n")
            f.write(f"- Average Time Regularity: {high_risk_features['time_regularity'].mean():.4f} (Higher values indicate less regularity)\n")
            f.write(f"- Average Burst Activity Ratio: {high_risk_features['burst_activity_ratio'].mean():.4f}\n")
            f.write(f"- Average Top 10% Volume Ratio: {high_risk_features['top_10_percent_volume_ratio'].mean():.4f} (Potentially indicating concentrated, less diversified large transactions)\n")
            f.write(f"- Average Round Amount Ratio: {high_risk_features['round_amount_ratio'].mean():.4f}\n")

        else:
            f.write("No high-risk wallets identified in this dataset.\n")

        f.write("\n![Credit Score Analysis Visuals](credit_score_analysis.png)\n")
        f.write("---\n")


if __name__ == '__main__':
    # Ensure a 'user-wallet-transactions.json' file exists in the same directory
    path = 'user-wallet-transactions.json'
    transaction_df = load_transactions(path)

    if not transaction_df.empty:
        features_df = feature_engineering(transaction_df)
        ml_scores_df, pca, feature_importance = create_ml_scores(features_df)
        results = generate_results(ml_scores_df)
        report = generate_report(results, pca, feature_importance)
        range_counts = create_visualizations(results)
        behavior = analyze_wallet_behavior(results, features_df)
        create_analysis_markdown(results, features_df, report, behavior, range_counts)
        print('\nAnalysis complete and report generated: analysis.md')
    else:
        print("No transactions loaded or file not found. Analysis halted.")