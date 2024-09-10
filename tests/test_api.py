from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Ad Spend Optimization API"}

def test_predict():
    response = client.post(
        "/predict",
        json={"impressions": 1000, "clicks": 100, "spend": 500}
    )
    assert response.status_code == 200
    assert "roas" in response.json()