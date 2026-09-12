import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN was not found in .env")
    raise SystemExit(1)

url = "https://router.huggingface.co/v1/models"

response = requests.get(
    url,
    headers={
        "Authorization": f"Bearer {token}"
    },
    timeout=30,
)

print("HTTP STATUS:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

data = response.json().get("data", [])

print("\nLIVE MODELS")
print("=" * 70)

count = 0

for model in data:
    providers = model.get("providers", [])

    live_providers = [
        provider
        for provider in providers
        if provider.get("status") == "live"
    ]

    if live_providers:
        print(f"\nMODEL: {model.get('id')}")

        for provider in live_providers:
            print(
                f"  Provider: {provider.get('provider')}"
            )

        count += 1

    if count >= 30:
        break

print("\n" + "=" * 70)
print(f"Displayed {count} models.")