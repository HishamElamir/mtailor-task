import requests

model_inputs = {'prompt': 'n01667114_mud_turtle.JPEG'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())