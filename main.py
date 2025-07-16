import json
from datetime import datetime,timedelta
from typing import Dict,List,Tuple,Any
import pandas as pd
import numpy as np
def _normalize_amount(row):
    """Normalize token amounts based on common decimal patterns."""
    amount=row['amount']
    symbol=row['assetSymbol']
    decimal_map={
        'USDC':6,
        'USDT':6,
        'DAI':18,
        'WETH':18,
        'WMATIC':18,
        'WBTC':8
    }
    decimals=decimal_map.get(symbol,18)
    return amount/(10**decimals)
def load_transactions(file_path:str)->pd.DataFrame:
    """Load data from JSON file"""
    print(f"Loading transactions from {file_path}")
    try:
        with open(file_path,'r') as f:
            data=json.load(f)
        transactions=[]
        for record in data:
            tx={
                'wallet':record.get('userWallet',''),
                'network':record.get('network',''),
                'protocol':record.get('protocol',''),
                'txHash':record.get('txHash',''),
                'timestamp':record.get('timestamp',''),
                'blockNumber':record.get('blockNumber',''),
                'action':record.get('action',''),
            }
            action_data=record.get('actionData',{})
            tx.update({
                'amount':action_data.get('amount','0'),
                'assetSymbol':action_data.get('assetSymbol','0'),
                'assetPriceUSD':action_data.get('assetPriceUSD','0'),
                'poolId':action_data.get('poolId','0'),
                'userId':action_data.get('userId','0'),
                'type':action_data.get('type',''),
            })
            if 'createdAt' in record and isinstance(record['createdAt'],dict):
                if '$date' in record['createdAt']:
                    tx['createdAt']=record['createdAt']['$date']
            transactions.append(tx)
        df=pd.DataFrame(transactions)
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='s',errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')
        df['blockNumber'] = pd.to_numeric(df['blockNumber'], errors='coerce')
        df['amountUSD'] = df['amount'] * df['assetPriceUSD']
        df['normalizedAmount']=df.apply(_normalize_amount,axis=1)
        df['normalizedAmountUSD']=df['normalizedAmount']*df['assetPriceUSD']
        df=df[df['wallet'].notna() & (df['wallet']!='')]
        df=df[df['timestamp'].notna()]
        df=df[df['amount']>0]
        #print(len(df))
        #print(df.columns)
        #print(df.describe)
        return df
    except Exception as e:
        print(f'error loading transactions:{e}')
        return pd.DataFrame()


# path= 'user-wallet-transactions.json'
# transaction_df=load_transactions(path)


def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    '''Feature engineering'''
    print('Performing feature engineering')
    feature=[]
    for wallet in df['wallet'].unique():
        wallet_df=df[df['wallet']==wallet].copy()
        wallet_df=wallet_df.sort_values('timestamp')
        _features={'wallet':wallet}
        amount_usd=wallet_df['normalizedAmountUSD'].dropna()
        if len(amount_usd)>0:
            _features['total_volume_usd']=amount_usd.sum()
            _features['avg_tx_size_usd']=amount_usd.mean()
            _features['max_tx_size_usd']=amount_usd.max()
            _features['volume_std']=amount_usd.std()
            _features['volume_coefficient_variation']=_features['volume_std']/max(_features['avg_tx_size_usd'],1)
            _features['volume_skewness']=amount_usd.skew()
            _features['volume_kurtosis']=amount_usd.kurtosis()
        else:
            _features.update({
                'total_volume_usd':0,'avg_tx_size_usd':0,'max_tx_size_usd':0,'volume_std':0,'volume_coefficient_variation':0,'volume_skewness':0,'volume_kurtosis':0
            })
        _features['total_transactions']=len(wallet_df)
        _features['days_active']=(wallet_df['timestamp'].max()-wallet_df['timestamp'].min()).days+1
        _features['tx_frequency']=_features['total_transactions']/max(_features['days_active'],1)
        if len(wallet_df)>1:
            time_diffs=wallet_df['timestamp'].diff().dt.total_seconds().dropna()
            _features['avg_time_between_tx']=time_diffs.mean()
            _features['time_regularity']=time_diffs.std()/max(time_diffs.mean(),1)
            _features['min_time_gap']=time_diffs.min()
            _features['max_time_gap']=time_diffs.max()
            short_gaps=(time_diffs<3600).sum()
            _features['burst_activity_ratio']=short_gaps/len(time_diffs)
        else:
            _features.update({
                'avg_time_between_tx': 0, 'time_regularity': 0, 'min_time_gap': 0,'max_time_gap': 0, 'burst_activity_ratio': 0
            })
        #Action composition
        action_counts=wallet_df['action'].value_counts()
        total_actions=len(wallet_df)
        _features['deposit_ratio']=action_counts.get('deposit',0)/total_actions
        _features['borrow_ratio']=action_counts.get('borrow',0)/total_actions
        _features['repay_ratio']=action_counts.get('repay',0)/total_actions
        _features['liquidation_ratio']=action_counts.get('liquidationcall',0)/total_actions
        _features['withdraw_ratio']=action_counts.get('redeemunderlying',0)/total_actions
        # Behavioral
        _features['action_diversity']=wallet_df['action'].nunique()
        _features['is_borrower']=1 if action_counts.get('borrow',0)>0 else 0
        _features['is_lender']=1 if action_counts.get('deposit',0)>0 else 0
        _features['borrow_repay_balance']=(action_counts.get('repay',0)-action_counts.get('borrow',0))/max(action_counts.get('borrow',1),1)
        #Asset behaviour
        _features['unique_assets']=wallet_df['assetSymbol'].nunique()
        if _features['unique_assets']>0:
            asset_counts=wallet_df['assetSymbol'].value_counts()
            _features['primary_asset_dominance']=asset_counts.iloc[0]/len(wallet_df)
            # Asset diversity
            asset_probs=asset_counts/len(wallet_df)
            _features['asset_entropy']=-sum(p*np.log2(p) for p in asset_probs if p>0)
        else:
            _features.update({'primary_asset_dominance':1,'asset_entropy':0})
        
        coins=['USDC','USDT','WETH','DAI','USDC.e','WMATIC','WBTC']
        coin_txs=wallet_df[wallet_df['assetSymbol'].isin(coins)]
        _features['coin_preference']=len(coin_txs)/len(wallet_df)
        #Pattern detection
        if len(amount_usd)>0:
            round_amt=amount_usd%1000==0
            _features['round_amount_ratio']=round_amt.mean()
            unique_amounts=amount_usd.nunique()
            _features['amount_repetition']=1-(unique_amounts/len(amount_usd))
        else:
            _features.update({'round_amount_ratio':0,'amount_repetition':0})
        
        # Weekly activity pattern
        if len(wallet_df)>=7:
            weekday_counts=wallet_df['timestamp'].dt.dayofweek.value_counts()
            weekday_entropy=-sum((c/len(wallet_df))*np.log2(c/len(wallet_df)) for c in weekday_counts.values)
            _features['weekly_pattern_entropy']=weekday_entropy/np.log2(7)
        else:
            _features['weekly_pattern_entropy']=0
        
        # Hourly activity patterns
        if len(wallet_df)>=24:
            hour_counts=wallet_df['timestamp'].dt.hour.value_counts()
            hour_entropy=-sum((c/len(wallet_df))*np.log2(c/len(wallet_df)) for c in hour_counts.values)
            _features['hourly_pattern_entropy']=hour_entropy/np.log2(24)
        else:
            _features['hourly_pattern_entropy']=0
        
        # Risk indicators
        if len(wallet_df)>1:
            rapid_txs=(wallet_df['timestamp'].diff()<timedelta(minutes=5)).sum()
            _features['rapid_transaction_ratio']=rapid_txs/len(wallet_df)
        else:
            _features['rapid_transaction_ratio']=0
        
        if len(amount_usd)>0:
            sorted_amounts=amount_usd.sort_values(ascending=False)
            top_10_percent=int(len(sorted_amounts)*0.1) or 1
            _features['top_10_percent_volume_ratio']=sorted_amounts.head(top_10_percent).sum()/sorted_amounts.sum()
        else:
            _features['top_10_percent_volume_ratio']=0
        recent_cutoff=wallet_df['timestamp'].max()-timedelta(days=30)
        recent_activity=wallet_df[wallet_df['timestamp']>=recent_cutoff]
        _features['recent_activity_ratio']=len(recent_activity)/len(wallet_df)
        feature.append(_features)
    feature_df=pd.DataFrame(feature)
    numeric_column=feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_column]=feature_df[numeric_column].fillna(0)
    #print(len(feature_df))
    # print(f"Engineered {len(feature_df.columns)-1} features for {len(feature_df)} wallets")
    # print('features:',feature_df)
    # print('head:',feature_df.head())
    # print('describe:',feature_df.describe)
    return feature_df

#feature_df=feature_engineering(transaction_df)


from sklearn.ensemble import RandomForestRegressor,IsolationForest,RandomForestClassifier,GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import silhouette_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_ml_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Create credit scores"""
    print("Creating credit scores...")
    
    # Prepare features
    feature_cols = [col for col in feature_df.columns if col != 'wallet']
    X = feature_df[feature_cols].values
    scaler=StandardScaler()
    # Normalize features
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    pca=PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # Cluster analysis to find behavioral groups
    clustering_model=KMeans(n_clusters=5,random_state=42)
    cluster_labels = clustering_model.fit_predict(X_pca)
    
    # Anomaly detection for risk assessment
    anomaly_detector=IsolationForest(contamination=0.1,random_state=42)
    
    anomaly_scores = anomaly_detector.fit_predict(X_scaled)
    anomaly_scores_continuous = anomaly_detector.decision_function(X_scaled)
    
    # composite behavioral score
    behavioral_scores = []
    
    for i in range(len(feature_df)):
        # Base score from cluster (0-600 range)
        cluster = cluster_labels[i]
        cluster_means = []
        for c in range(clustering_model.n_clusters):
            cluster_mask = cluster_labels == c
            if cluster_mask.sum() > 0:
                # Calculate cluster quality metrics
                cluster_data = X_pca[cluster_mask]
                cluster_center = clustering_model.cluster_centers_[c]
                
                # Distance from cluster center (lower = more typical)
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                avg_distance = distances.mean()
                
                # Volume and consistency indicators
                volume_score = feature_df.iloc[cluster_mask]['total_volume_usd'].mean()
                consistency_score = 1 / (1 + feature_df.iloc[cluster_mask]['time_regularity'].mean())
                
                cluster_score = (volume_score * 0.4 + consistency_score * 0.6) / avg_distance
                cluster_means.append(cluster_score)
            else:
                cluster_means.append(0)
        
        # Normalize cluster scores and assign
        if len(cluster_means) > 0:
            cluster_scores_norm = MinMaxScaler(feature_range=(200, 600)).fit_transform(
                np.array(cluster_means).reshape(-1, 1)
            ).flatten()
            base_score = cluster_scores_norm[cluster]
        else:
            base_score = 400
        
        # Risk adjustment from anomaly detection (-200 to +200)
        anomaly_adjustment = (anomaly_scores_continuous[i] + 0.5) * 200  # Normalize roughly to -200,+200
        risk_penalty = 0 if anomaly_scores[i] == 1 else -100  # Penalty for anomalies
        
        # Behavioral bonuses/penalties
        row = feature_df.iloc[i]
        
        behavioral_bonus = 0
        # Positive behaviors
        if row['is_lender'] and row['borrow_repay_balance'] > 0:
            behavioral_bonus += 100
        if row['coin_preference'] > 0.5:
            behavioral_bonus += 50
        if row['asset_entropy'] > 1.0:  
            behavioral_bonus += 30
        
        # Negative behaviors
        if row['liquidation_ratio'] > 0:
            behavioral_bonus -= 200
        if row['round_amount_ratio'] > 0.5:
            behavioral_bonus -= 100
        if row['burst_activity_ratio'] > 0.8:
            behavioral_bonus -= 150
        
        # Final score
        final_score = base_score + anomaly_adjustment + risk_penalty + behavioral_bonus
        final_score = max(0, min(1000, final_score))  
        
        behavioral_scores.append(final_score)
    
    feature_df['credit_score'] = behavioral_scores
    
    # Store feature importance from PCA
    feature_importance = dict(zip(feature_cols, np.abs(pca.components_[0])))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    # for k, v in list(feature_importance.items())[:10]:
    #     print(f"{k}: {v:.4f}")
    return feature_df


#create_ml_scores(feature_df)


def categorize_score(self, score: float) -> str:
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

def generate_results(feature_df:pd.DataFrame)->pd.DataFrame:
    '''Final results'''
    results=feature_df[['wallet','credit_score']].copy()
    results['credit_score']=results['credit_score'].round().astype(int)
    results['score_category']=results['credit_score'].apply(categorize_score)
    results['is_borrower']=feature_df['is_borrower']
    results['is_lender']=feature_df['is_lender']
    results['total_volume_usd']=feature_df['total_volume_usd'].round(2)
    results['days_active']=feature_df['days_active']
    results['liquidation_ratio']=feature_df['liquidation_ratio'].round(4)
    return results.sort_values('credit_score',ascending=False)