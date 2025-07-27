import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class WalletScoringSystem:
    def __init__(self, json_file_path, output_csv='wallet_scores.csv'):
        self.json_file_path = json_file_path
        self.output_csv = output_csv
        self.df = None
        self.features_df = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.isolation_forest = None
        
    def load_and_prepare_data(self):
        """Load JSON data and prepare features for ML"""
        print("Loading and preparing data...")
        
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract features for each wallet
        wallet_data = []
        for wallet_id, wallet_info in data.items():
            features = wallet_info.get('features', {})
            features['wallet_address'] = wallet_id
            features['total_transactions'] = wallet_info.get('total_compound_transactions', 0)
            wallet_data.append(features)
        
        self.df = pd.DataFrame(wallet_data)
        print(f"Loaded {len(self.df)} wallets")
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
        # Create additional derived features
        self.create_derived_features()
        
        # Select features for ML (exclude wallet_address)
        feature_columns = [col for col in self.df.columns if col != 'wallet_address']
        self.features_df = self.df[feature_columns].copy()
        
        print(f"Prepared {len(feature_columns)} features for analysis")
        return self.df
    
    def create_derived_features(self):
        """Create additional features for better scoring"""
        print("Creating derived features...")
        
        # Activity intensity features
        self.df['activity_intensity'] = (
            self.df['total_transactions'] * self.df['transaction_frequency']
        )
        
        # Financial sophistication score
        self.df['financial_sophistication'] = (
            self.df['unique_functions'] * 0.3 +
            self.df['unique_contracts'] * 0.3 +
            self.df['function_diversity_ratio'] * 0.4
        )
        
        # Risk profile
        self.df['risk_profile'] = (
            self.df['failed_transaction_rate'] * 0.4 +
            self.df['avg_gas_price_gwei'] / 100 * 0.3 +  # Normalized
            self.df['liquidate_transactions'] * 0.3
        )
        
        # Lending behavior score
        self.df['lending_behavior'] = (
            self.df['supply_to_borrow_ratio'] * 0.4 +
            self.df['repay_to_borrow_ratio'] * 0.4 +
            (self.df['mint_transactions'] + self.df['supply_transactions']) * 0.2
        )
        
        # Protocol adoption score
        self.df['protocol_adoption'] = (
            (self.df['compound_v2_usage'] + self.df['compound_v3_usage']) * 0.5 +
            self.df['contract_diversity_ratio'] * 0.5
        )
        
        # Volume efficiency (value per transaction)
        self.df['volume_efficiency'] = np.where(
            self.df['total_transactions'] > 0,
            self.df['total_value_eth'] / self.df['total_transactions'],
            0
        )
        
        # Gas efficiency
        self.df['gas_efficiency'] = np.where(
            self.df['total_gas_fees_eth'] > 0,
            self.df['total_value_eth'] / self.df['total_gas_fees_eth'],
            0
        )
        
        # Time-based consistency
        self.df['consistency_score'] = np.where(
            self.df['avg_transaction_interval_hours'] > 0,
            1 / (1 + self.df['avg_transaction_interval_hours'] / 24),  # Normalize to daily
            0
        )
        
        print("Derived features created successfully")
    
    def detect_outliers_and_clean(self):
        """Detect and handle outliers"""
        print("Detecting and handling outliers...")
        
        # Use IQR method for outlier detection
        Q1 = self.features_df.quantile(0.25)
        Q3 = self.features_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap extreme outliers instead of removing them
        for column in self.features_df.columns:
            if self.features_df[column].dtype in ['int64', 'float64']:
                self.features_df[column] = np.clip(
                    self.features_df[column], 
                    lower_bound[column], 
                    upper_bound[column]
                )
        
        print("Outlier handling completed")
    
    def scale_features(self):
        """Scale features for ML algorithms"""
        print("Scaling features...")
        
        # Use RobustScaler as it's less sensitive to outliers
        self.scaler = RobustScaler()
        scaled_features = self.scaler.fit_transform(self.features_df)
        
        self.scaled_df = pd.DataFrame(
            scaled_features, 
            columns=self.features_df.columns,
            index=self.features_df.index
        )
        
        print("Feature scaling completed")
        return self.scaled_df
    
    def perform_pca_analysis(self, n_components=10):
        """Perform PCA for dimensionality reduction and feature importance"""
        print(f"Performing PCA analysis with {n_components} components...")
        
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(self.scaled_df)
        
        # Create PCA DataFrame
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        self.pca_df = pd.DataFrame(pca_features, columns=pca_columns)
        
        # Calculate explained variance
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"PCA explained variance ratio: {explained_variance_ratio[:5]}")
        print(f"Cumulative variance (first 5 components): {cumulative_variance[:5]}")
        
        # Feature importance from PCA
        feature_importance = np.abs(self.pca.components_).mean(axis=0)
        self.feature_importance = pd.DataFrame({
            'feature': self.features_df.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("PCA analysis completed")
        return self.pca_df
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(self.scaled_df) // 2))
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(self.scaled_df)
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_df, kmeans_temp.labels_))
        
        # Choose optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        # Perform final clustering
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.scaled_df)
        
        self.df['cluster'] = cluster_labels
        
        print("Clustering completed")
        return cluster_labels
    
    def detect_anomalies(self):
        """Detect anomalous wallets using Isolation Forest"""
        print("Detecting anomalous wallets...")
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = self.isolation_forest.fit_predict(self.scaled_df)
        anomaly_scores = self.isolation_forest.decision_function(self.scaled_df)
        
        self.df['is_anomaly'] = anomaly_labels == -1
        self.df['anomaly_score'] = anomaly_scores
        
        anomaly_count = sum(anomaly_labels == -1)
        print(f"Detected {anomaly_count} anomalous wallets ({anomaly_count/len(self.df)*100:.1f}%)")
        
        return anomaly_scores
    
    def calculate_composite_scores(self):
        """Calculate comprehensive wallet scores"""
        print("Calculating composite wallet scores...")
        
        # Normalize key metrics to 0-1 scale
        scaler_score = MinMaxScaler()
        
        # Key metrics for scoring
        scoring_features = [
            'total_transactions', 'total_value_eth', 'transaction_frequency',
            'financial_sophistication', 'lending_behavior', 'protocol_adoption',
            'volume_efficiency', 'gas_efficiency', 'consistency_score'
        ]
        
        # Ensure all scoring features exist
        available_features = [f for f in scoring_features if f in self.df.columns]
        
        if not available_features:
            print("Warning: No scoring features available. Using basic features.")
            available_features = ['total_transactions', 'total_value_eth']
        
        # Create scoring matrix
        scoring_matrix = self.df[available_features].fillna(0)
        normalized_scores = scaler_score.fit_transform(scoring_matrix)
        
        # Weight different aspects
        weights = {
            'total_transactions': 0.15,
            'total_value_eth': 0.20,
            'transaction_frequency': 0.10,
            'financial_sophistication': 0.15,
            'lending_behavior': 0.15,
            'protocol_adoption': 0.10,
            'volume_efficiency': 0.05,
            'gas_efficiency': 0.05,
            'consistency_score': 0.05
        }
        
        # Calculate weighted score
        composite_score = np.zeros(len(self.df))
        for i, feature in enumerate(available_features):
            weight = weights.get(feature, 1.0 / len(available_features))
            composite_score += normalized_scores[:, i] * weight
        
        # Cluster-based adjustment
        if 'cluster' in self.df.columns:
            cluster_stats = self.df.groupby('cluster')[available_features].mean()
            cluster_ranks = cluster_stats.mean(axis=1).rank(ascending=False)
            cluster_multipliers = (cluster_ranks / cluster_ranks.max() * 0.2) + 0.9  
            
            for cluster_id in cluster_multipliers.index:
                mask = self.df['cluster'] == cluster_id
                composite_score[mask] *= cluster_multipliers[cluster_id]
        
        
        if 'is_anomaly' in self.df.columns:
            anomaly_boost = self.df['is_anomaly'].astype(int) * 0.1
            composite_score += anomaly_boost
        
        # Scale final scores to 1-1000 range
        self.df['raw_score'] = composite_score
        self.df['score'] = ((composite_score - composite_score.min()) / 
                          (composite_score.max() - composite_score.min()) * 999 + 1).round().astype(int)
        
        print("Composite scores calculated successfully")
        print(f"Score distribution: Min={self.df['score'].min()}, Max={self.df['score'].max()}, Mean={self.df['score'].mean():.1f}")
        
    def generate_insights(self):
        """Generate insights about the scoring"""
        print("\n=== WALLET SCORING INSIGHTS ===")
        
        # Score distribution
        print(f"Score Statistics:")
        print(f"  Mean Score: {self.df['score'].mean():.1f}")
        print(f"  Median Score: {self.df['score'].median():.1f}")
        print(f"  Standard Deviation: {self.df['score'].std():.1f}")
        
        # Top wallets
        top_wallets = self.df.nlargest(5, 'score')[['wallet_address', 'score', 'total_transactions', 'total_value_eth']]
        print(f"\nTop 5 Wallets:")
        for _, wallet in top_wallets.iterrows():
            print(f"  {wallet['wallet_address']}: Score {wallet['score']} ({wallet['total_transactions']} txs, {wallet['total_value_eth']:.2f} ETH)")
        
        # Cluster analysis
        if 'cluster' in self.df.columns:
            cluster_stats = self.df.groupby('cluster').agg({
                'score': ['mean', 'count'],
                'total_transactions': 'mean',
                'total_value_eth': 'mean'
            }).round(2)
            print(f"\nCluster Analysis:")
            print(cluster_stats)
        if hasattr(self, 'feature_importance'):
            print(f"\nTop 10 Most Important Features:")
            for _, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
    
    def save_results(self):
        """Save results to CSV"""
        print(f"Saving results to {self.output_csv}...")
        output_df = self.df[['wallet_address', 'score']].copy()
        output_df.columns = ['wallet_id', 'score']
        output_df = output_df.sort_values('score', ascending=False)
        output_df.to_csv(self.output_csv, index=False)
        detailed_output = self.output_csv.replace('.csv', '_detailed.csv')
        detailed_columns = [
            'wallet_address', 'score', 'total_transactions', 'total_value_eth',
            'transaction_frequency', 'unique_functions', 'unique_contracts',
        ]
        
        # Add cluster and anomaly info if available
        if 'cluster' in self.df.columns:
            detailed_columns.append('cluster')
        if 'is_anomaly' in self.df.columns:
            detailed_columns.append('is_anomaly')
        
        available_detailed_columns = [col for col in detailed_columns if col in self.df.columns]
        self.df[available_detailed_columns].to_csv(detailed_output, index=False)
        
        print(f"Results saved:")
        print(f"  Main scores: {self.output_csv}")
        print(f"  Detailed analysis: {detailed_output}")
        
        return output_df
    
    def run_complete_analysis(self):
        """Run the complete ML scoring pipeline"""
        print("=== STARTING WALLET SCORING ANALYSIS ===\n")
        
        try:
            self.load_and_prepare_data()
            
            self.detect_outliers_and_clean()
            
            self.scale_features()
            
            self.perform_pca_analysis()
            
            self.perform_clustering()
            
            self.detect_anomalies()
            
            self.calculate_composite_scores()
            
            self.generate_insights()
            results = self.save_results()
            
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Processed {len(self.df)} wallets successfully")
            
            return results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    
    scorer = WalletScoringSystem(
        json_file_path='extracted_data.json',  
        output_csv='wallet_scores.csv'
    )
    
    results = scorer.run_complete_analysis()
    
    if results is not None:
        print("\nSample of final results:")
        print(results.head(10))
    else:
        print("Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()