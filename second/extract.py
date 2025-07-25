import os
import requests
chains=[42161, 8453, 10, 534352, 81457]
api_key=os.environ.get('ETHERSCAN_API_KEY')
data=[]
for chain in chains:
    print(chain)
    response=requests.get(f"https://api.etherscan.io/v2/api?chainid=1924&module=account&action=balance&address=0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511&tag=latest&apikey={api_key}")
    data.append(response.json())
if len(data)==0:
    raise ValueError('nothing')
print(data)

# response=requests.get("https://api.etherscan.io/v2/api?chainid=1924&module=account&action=balance&address=0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511&tag=latest&apikey={api_key}")
# data.append(response.json())
# print(data)
# print(api_key)