import requests

data_to_be_sent = {
    'sepal_length' : '2.1',
    'sepal_width' : '1.0',
    'petal_length' : '3.2',
    'petal_width' : '0.2'
}

url = 'http://127.0.0.1:5000/Predictions'

response = requests.post(url = url, data = data_to_be_sent)

if response.status_code == 200:
    print(response.text)
else: 
    print(f'Exiting with error code : {response.status_code}')
