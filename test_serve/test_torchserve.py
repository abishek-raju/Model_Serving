import argparse
import requests
from requests import Response
import os

# parser = argparse.ArgumentParser(description='Process input.')
# parser.add_argument('--url', metavar='URL', type=str,
#                     help='URL to test')
# # parser.add_argument('--sum', dest='accumulate', action='store_const',
# #                     const=sum, default=max,
# #                     help='sum the integers (default: find the max)')

# args = parser.parse_args()

# BASE_URL = args.url
# print(os.environ.get('BASE_URL'))
BASE_URL = os.environ.get('BASE_URL')
print(f"\nBASE_URL : {BASE_URL}")
def test_server_available():
    headers = {
            'Content-Type': 'application/json'
            }
    response: Response = requests.request("GET", BASE_URL, headers=headers, timeout=15)
    # print(f"response: {response}")
    assert response.status_code == 200
