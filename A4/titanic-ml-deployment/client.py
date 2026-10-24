"""Simple client for exercising the Titanic FastAPI prediction service."""

import os

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
VALID_PASSENGER = {
    "Pclass": 2,
    "Sex": "female",
    "Age": 29.0,
    "SibSp": 0,
    "Parch": 1,
    "Fare": 26.0,
    "Embarked": "C",
}
INVALID_PASSENGER = {
    **VALID_PASSENGER,
    "Pclass": 4,
}


def print_response(response: requests.Response) -> None:
    """Print an HTTP response consistently for the demonstration."""
    print(f"Status code: {response.status_code}")
    try:
        print(response.json())
    except ValueError:
        print(response.text)


def main() -> None:
    """Check health and demonstrate valid and invalid API requests."""
    base_url = os.environ.get("TITANIC_API_URL", DEFAULT_BASE_URL).rstrip("/")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        print("GET /health")
        print_response(health_response)

        print("\nPOST /predict with a valid passenger")
        valid_response = requests.post(
            f"{base_url}/predict",
            json=VALID_PASSENGER,
            timeout=10,
        )
        print_response(valid_response)

        print("\nPOST /predict with an invalid passenger (Pclass=4)")
        invalid_response = requests.post(
            f"{base_url}/predict",
            json=INVALID_PASSENGER,
            timeout=10,
        )
        print_response(invalid_response)
        if invalid_response.status_code != 422:
            print("Warning: the intentionally invalid request did not return HTTP 422.")
    except requests.exceptions.ConnectionError:
        print(
            f"Could not connect to {base_url}. "
            "Start the FastAPI server with Uvicorn before running client.py."
        )
    except requests.exceptions.RequestException as error:
        print(f"Request to the FastAPI server failed: {error}")


if __name__ == "__main__":
    main()
