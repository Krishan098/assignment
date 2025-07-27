import asyncio
import aiohttp
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import defaultdict
import re
import os
from dotenv import load_dotenv

class EtherscanAPICompoundAnalyzer:
    def __init__(self, api_key, csv_file_path, output_file='compound_api_data.json', rate_limit=5):
        self.api_key = api_key
        self.csv_file_path = csv_file_path
        self.output_file = output_file
        self.rate_limit = rate_limit  
        self.base_url = "https://api.etherscan.io/api"
        self.all_wallet_data = {}
        
        self.compound_contracts = {
            # Compound V2 cTokens
            '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643': 'cDAI',
            '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5': 'cETH',
            '0x39aa39c021dfbadb6ec6e45c19bf3eb1d31b17a0': 'cUSDC',
            '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9': 'cUSDT',
            '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4': 'cWBTC',
            '0x70e36f6bf80a52b3b46b3af8e106cc0ed743e8e4': 'cLEND',
            '0xb3319f5d18bc0d84dd1b4825dcde5d5f7266d407': 'cZRX',
            '0x158079ee67fce2f58472a96584a73c7ab9ac95c1': 'cREP',
            '0xf5dce57282a584d2746faf1593d3121fcac444dc': 'cSAI',
            '0x35a18000230da775cac24873d00ff85bccded550': 'cUNI',
            '0x4b0181102a0112a2ef11abee5563bb4a3176c9d7': 'cSUSHI',
            
            # Compound V3 contracts
            '0xc3d688b66703497daa19211eedff47f25384cdc3': 'cUSDCv3',
            '0xa17581a9e3356d9a858b789d68b4d866e593ae94': 'cETHv3',
            
            # Comptroller addresses
            '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b': 'Comptroller_V2',
            '0x1b0e765f6224c21223aea2af16c1c46e38885a40': 'Comptroller_V3_ETH',
            '0x3895f3833d355f1d33ca1742e5a82b97b8c6c9d7': 'Comptroller_V3_USDC'
        }
        
        # Compound function signatures
        self.compound_functions = {
            '0xa0712d68': 'mint',
            '0xdb006a75': 'redeem',
            '0x852a12e3': 'redeemUnderlying',
            '0xc5ebeaec': 'borrow',
            '0x0e752702': 'repayBorrow',
            '0x4e4d9fea': 'repayBorrowBehalf',
            '0xf5e3c462': 'liquidateBorrow',
            '0x1249c58b': 'mint()',  # V3
            '0x2e1a7d4d': 'withdraw',  # V3
            '0xf2fde38b': 'supply',   # V3
        }

    def load_wallet_ids(self):
        """Load wallet IDs from CSV file"""
        try:
            df = pd.read_csv(self.csv_file_path)
            wallet_ids = df.iloc[:, 0].dropna().tolist()
            print(f"Loaded {len(wallet_ids)} wallet IDs from CSV")
            return [str(w).strip().lower() for w in wallet_ids if str(w).strip() != 'nan']
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return []

    async def make_api_request(self, session, params):
        """Make rate-limited API request"""
        params['apikey'] = self.api_key
        
        try:
            await asyncio.sleep(1 / self.rate_limit)  
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == '1':
                        return data.get('result', [])
                    else:
                        print(f"API Error: {data.get('message', 'Unknown error')}")
                        return []
                else:
                    print(f"HTTP Error: {response.status}")
                    return []
        except Exception as e:
            print(f"Request error: {e}")
            return []

    async def get_normal_transactions(self, session, address):
        """Get all normal transactions for an address"""
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': 10000,  
            'sort': 'desc'
        }
        
        all_transactions = []
        page = 1
        
        while True:
            params['page'] = page
            transactions = await self.make_api_request(session, params)
            
            if not transactions:
                break
                
            compound_txs = [tx for tx in transactions if self.is_compound_transaction(tx)]
            all_transactions.extend(compound_txs)
            
            print(f"  Page {page}: {len(compound_txs)} Compound transactions found")
            if len(transactions) < 10000:
                break
                
            page += 1
            if page > 100:
                print("  Reached page limit (100)")
                break
        
        return all_transactions

    async def get_internal_transactions(self, session, address):
        """Get internal transactions (for complete analysis)"""
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': 10000,
            'sort': 'desc'
        }
        
        transactions = await self.make_api_request(session, params)
        # Filter for Compound-related internal transactions
        return [tx for tx in transactions if self.is_compound_transaction(tx)]

    async def get_erc20_transfers(self, session, address):
        """Get ERC20 token transfers"""
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': 10000,
            'sort': 'desc'
        }
        
        all_transfers = []
        page = 1
        
        while True:
            params['page'] = page
            transfers = await self.make_api_request(session, params)
            
            if not transfers:
                break
            
            # Filter for Compound token transfers
            compound_transfers = [tx for tx in transfers if self.is_compound_transaction(tx)]
            all_transfers.extend(compound_transfers)
            
            print(f"  Token transfers page {page}: {len(compound_transfers)} Compound transfers found")
            
            if len(transfers) < 10000:
                break
                
            page += 1
            
            if page > 50:  
                break
        
        return all_transfers

    def is_compound_transaction(self, tx):
        """Check if transaction is Compound-related"""
        if not tx:
            return False
        
        to_address = tx.get('to', '').lower()
        from_address = tx.get('from', '').lower()
        contract_address = tx.get('contractAddress', '').lower()
        
        compound_addresses = [addr.lower() for addr in self.compound_contracts.keys()]
        
        if (to_address in compound_addresses or 
            from_address in compound_addresses or 
            contract_address in compound_addresses):
            return True
        
        
        input_data = tx.get('input', '')
        if input_data and len(input_data) >= 10:
            func_sig = input_data[:10].lower()
            if func_sig in self.compound_functions:
                return True
        
        return False

    def parse_transaction(self, tx, tx_type='normal'):
        """Parse transaction data for analysis"""
        try:
            parsed = {
                'hash': tx.get('hash', ''),
                'block_number': int(tx.get('blockNumber', 0)),
                'timestamp': int(tx.get('timeStamp', 0)),
                'from': tx.get('from', '').lower(),
                'to': tx.get('to', '').lower(),
                'value': int(tx.get('value', 0)),
                'value_eth': int(tx.get('value', 0)) / 1e18,
                'gas': int(tx.get('gas', 0)),
                'gas_price': int(tx.get('gasPrice', 0)),
                'gas_used': int(tx.get('gasUsed', 0)),
                'gas_fee_eth': (int(tx.get('gasUsed', 0)) * int(tx.get('gasPrice', 0))) / 1e18,
                'transaction_type': tx_type,
                'is_error': tx.get('isError', '0') == '1',
                'contract_address': tx.get('contractAddress', ''),
            }
            
            # Add compound contract info
            to_address = parsed['to']
            if to_address in [addr.lower() for addr in self.compound_contracts.keys()]:
                for addr, name in self.compound_contracts.items():
                    if addr.lower() == to_address:
                        parsed['compound_contract'] = name
                        break
            
            # Add function info
            input_data = tx.get('input', '')
            if input_data and len(input_data) >= 10:
                func_sig = input_data[:10].lower()
                parsed['function_signature'] = func_sig
                parsed['function_name'] = self.compound_functions.get(func_sig, 'unknown')
            
            # Add token info for ERC20 transfers
            if tx_type == 'erc20':
                parsed['token_name'] = tx.get('tokenName', '')
                parsed['token_symbol'] = tx.get('tokenSymbol', '')
                parsed['token_decimal'] = int(tx.get('tokenDecimal', 18))
                parsed['token_value'] = int(tx.get('value', 0)) / (10 ** parsed['token_decimal'])
            
            return parsed
            
        except Exception as e:
            print(f"Error parsing transaction: {e}")
            return None

    def get_empty_features(self):
        """Return empty feature set for wallets with no Compound activity"""
        return {key: 0 for key in [
            'total_transactions', 'total_value_eth', 'total_gas_fees_eth',
            'avg_transaction_value', 'avg_gas_fee', 'days_active',
            'transaction_frequency', 'avg_transaction_interval_hours',
            'unique_functions', 'function_diversity_ratio', 'mint_transactions',
            'redeem_transactions', 'borrow_transactions', 'repay_transactions',
            'liquidate_transactions', 'supply_transactions', 'withdraw_transactions',
            'unique_contracts', 'contract_diversity_ratio', 'failed_transaction_rate',
            'avg_gas_price_gwei', 'supply_to_borrow_ratio', 'repay_to_borrow_ratio',
            'compound_v2_usage', 'compound_v3_usage'
        ]}

    async def analyze_wallet(self, session, wallet_address):
        """Analyze a single wallet for Compound activity"""
        print(f"\nAnalyzing wallet: {wallet_address}")
        
        
        normal_txs = await self.get_normal_transactions(session, wallet_address)
        internal_txs = await self.get_internal_transactions(session, wallet_address)
        erc20_txs = await self.get_erc20_transfers(session, wallet_address)
        
        all_transactions = []
        
        for tx in normal_txs:
            parsed = self.parse_transaction(tx, 'normal')
            if parsed:
                all_transactions.append(parsed)
        
        for tx in internal_txs:
            parsed = self.parse_transaction(tx, 'internal')
            if parsed:
                all_transactions.append(parsed)
                
        for tx in erc20_txs:
            parsed = self.parse_transaction(tx, 'erc20')
            if parsed:
                all_transactions.append(parsed)
        
        print(f"  Found {len(all_transactions)} Compound transactions")
        
        if not all_transactions:
            features = self.get_empty_features()
        else:
            features = self.calculate_features(all_transactions, wallet_address)
        self.all_wallet_data[wallet_address] = {
            'features': features,
            'transactions': all_transactions,
            'summary': {
                'total_compound_transactions': len(all_transactions),
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        return features

    def calculate_features(self, transactions, wallet_address):
        """Calculate comprehensive features from transactions"""
        if not transactions:
            return self.get_empty_features()
        
        features = {}
        
        # Basic transaction metrics
        features['total_transactions'] = len(transactions)
        features['total_value_eth'] = sum(tx['value_eth'] for tx in transactions)
        features['total_gas_fees_eth'] = sum(tx['gas_fee_eth'] for tx in transactions)
        
        # Average metrics
        features['avg_transaction_value'] = features['total_value_eth'] / len(transactions)
        features['avg_gas_fee'] = features['total_gas_fees_eth'] / len(transactions)
        features['avg_gas_price_gwei'] = sum(tx['gas_price'] for tx in transactions) / len(transactions) / 1e9
        
        # Time-based metrics
        timestamps = [tx['timestamp'] for tx in transactions]
        if len(timestamps) > 1:
            min_time, max_time = min(timestamps), max(timestamps)
            features['days_active'] = (max_time - min_time) / 86400  # seconds to days
            features['transaction_frequency'] = len(transactions) / max(features['days_active'], 1)
            sorted_times = sorted(timestamps)
            intervals = [sorted_times[i] - sorted_times[i-1] for i in range(1, len(sorted_times))]
            features['avg_transaction_interval_hours'] = sum(intervals) / len(intervals) / 3600 if intervals else 0
        else:
            features['days_active'] = 0
            features['transaction_frequency'] = 0
            features['avg_transaction_interval_hours'] = 0
        
        # Function analysis
        functions = [tx.get('function_name', 'unknown') for tx in transactions if tx.get('function_name')]
        unique_functions = set(functions)
        features['unique_functions'] = len(unique_functions)
        features['function_diversity_ratio'] = len(unique_functions) / len(transactions) if transactions else 0
        
        # Count specific function types
        features['mint_transactions'] = sum(1 for f in functions if f and 'mint' in f.lower())
        features['redeem_transactions'] = sum(1 for f in functions if f and 'redeem' in f.lower())
        features['borrow_transactions'] = sum(1 for f in functions if f and 'borrow' in f.lower())
        features['repay_transactions'] = sum(1 for f in functions if f and 'repay' in f.lower())
        features['liquidate_transactions'] = sum(1 for f in functions if f and 'liquidate' in f.lower())
        features['supply_transactions'] = sum(1 for f in functions if f and 'supply' in f.lower())
        features['withdraw_transactions'] = sum(1 for f in functions if f and 'withdraw' in f.lower())
        
        # Contract diversity
        contracts = [tx.get('compound_contract', 'unknown') for tx in transactions if tx.get('compound_contract')]
        unique_contracts = set(contracts)
        features['unique_contracts'] = len(unique_contracts)
        features['contract_diversity_ratio'] = len(unique_contracts) / len(transactions) if transactions else 0
        
        # Error rate
        failed_txs = sum(1 for tx in transactions if tx.get('is_error', False))
        features['failed_transaction_rate'] = failed_txs / len(transactions) if transactions else 0
        
        # Protocol version usage
        v2_contracts = ['cDAI', 'cETH', 'cUSDC', 'cUSDT', 'cWBTC', 'cLEND', 'cZRX', 'cREP', 'cSAI', 'cUNI', 'cSUSHI']
        v3_contracts = ['cUSDCv3', 'cETHv3']
        
        v2_txs = sum(1 for tx in transactions if tx.get('compound_contract') in v2_contracts)
        v3_txs = sum(1 for tx in transactions if tx.get('compound_contract') in v3_contracts)
        
        features['compound_v2_usage'] = v2_txs / len(transactions) if transactions else 0
        features['compound_v3_usage'] = v3_txs / len(transactions) if transactions else 0
        supply_txs = features['mint_transactions'] + features['supply_transactions']
        borrow_txs = features['borrow_transactions']
        repay_txs = features['repay_transactions']
        
        features['supply_to_borrow_ratio'] = supply_txs / max(borrow_txs, 1)
        features['repay_to_borrow_ratio'] = repay_txs / max(borrow_txs, 1)
        
        return features

    async def run_analysis(self):
        """Main analysis workflow"""
        print("=== COMPOUND PROTOCOL ANALYZER ===")
        wallet_addresses = self.load_wallet_ids()
        if not wallet_addresses:
            print("No wallet addresses found!")
            return
        
        print(f"Starting analysis of {len(wallet_addresses)} wallets...")
        
        async with aiohttp.ClientSession() as session:
            for i, wallet_address in enumerate(wallet_addresses, 1):
                try:
                    print(f"\n[{i}/{len(wallet_addresses)}] Processing: {wallet_address}")
                    features = await self.analyze_wallet(session, wallet_address)
                    if features['total_transactions'] > 0:
                        print(f"  ✓ {features['total_transactions']} transactions, "
                              f"{features['total_value_eth']:.4f} ETH total value")
                    else:
                        print(f"  - No Compound activity found")
                        
                except Exception as e:
                    print(f"  ✗ Error analyzing wallet {wallet_address}: {e}")
                    continue
        await self.save_results()
        self.print_summary()

    def print_summary(self):
        """Print analysis summary"""
        total_wallets = len(self.all_wallet_data)
        active_wallets = sum(1 for data in self.all_wallet_data.values() 
                            if data['features']['total_transactions'] > 0)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Total wallets analyzed: {total_wallets}")
        print(f"Wallets with Compound activity: {active_wallets}")
        print(f"Wallets with no activity: {total_wallets - active_wallets}")
        
        if active_wallets > 0:
            all_features = [data['features'] for data in self.all_wallet_data.values() 
                           if data['features']['total_transactions'] > 0]
            
            avg_transactions = sum(f['total_transactions'] for f in all_features) / len(all_features)
            avg_value = sum(f['total_value_eth'] for f in all_features) / len(all_features)
            
            print(f"Average transactions per active wallet: {avg_transactions:.2f}")
            print(f"Average ETH value per active wallet: {avg_value:.4f}")

    async def save_results(self):
        """Save comprehensive results"""
        try:
            # Save detailed JSON data
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_wallet_data, f, indent=2, ensure_ascii=False)
            
            # Save summary CSV
            summary_file = self.output_file.replace('.json', '_summary.csv')
            summary_data = []
            
            for wallet_address, data in self.all_wallet_data.items():
                features = data['features']
                features['wallet_address'] = wallet_address
                summary_data.append(features)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                print(f"\n=== RESULTS SAVED ===")
                print(f"Detailed data: {self.output_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
load_dotenv()

async def main():
    """Main execution function"""
    API_KEY = os.getenv("ETHERSCAN_API_KEY") 
    
    if not API_KEY:
        print("Error: ETHERSCAN_API_KEY not found in environment variables")
        return
      
    analyzer = EtherscanAPICompoundAnalyzer(
        api_key=API_KEY,
        csv_file_path='Wallet id - Sheet1.csv', 
        output_file='extracted_data.json',
        rate_limit=5  
    )
    await analyzer.run_analysis()

if __name__ == "__main__":
    asyncio.run(main())