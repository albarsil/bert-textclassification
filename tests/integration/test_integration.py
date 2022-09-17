import os
import time

import pytest
import requests

d = '.'
print([os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))])

def test_health_endpoint():
    response = requests.get('http://localhost:80/health')
    assert response.status_code == 200

def test_ping_endpoint():
    response = requests.get('http://localhost:80/ping')
    assert response.status_code == 200

@pytest.mark.parametrize(
    "payload", [
        {'text': 'oi'}
        ])
def test_model_endpoint_with_correct_payload(payload):
    response = requests.post('http://localhost:80/invocations/predict', json=payload)
    assert response.status_code == 200
    
    expected_output_keys = ['score', 'text']
    output_keys = response.json().keys()

    for key in expected_output_keys:
        assert key in output_keys

@pytest.mark.parametrize(
    "payload", [
        {'text': 'oi'}
        ])

def test_model_endpoint_with_incorrect_payload(payload):
    response = requests.post('http://localhost:80/invocations/predict', json=payload)
    assert response.status_code == 400
