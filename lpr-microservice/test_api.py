import requests
import sys
import json

# Test the LPR microservice
url = "http://localhost:8000/detect"

# Test with Back.png
print("Testing with Back.png...")
files = {'file': open(r'c:\Users\jamie.barker\Desktop\GithubProjects\IPPR-MLPRS\sampleData\Back.png', 'rb')}
data = {'return_phases': 'false'}

response = requests.post(url, files=files, data=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*80 + "\n")

# Test with Front-Upclose.png
print("Testing with Front-Upclose.png...")
files = {'file': open(r'c:\Users\jamie.barker\Desktop\GithubProjects\IPPR-MLPRS\sampleData\Front-Upclose.png', 'rb')}

response = requests.post(url, files=files, data=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
