"""Presentation names shared with the native Core resource; no identity migration."""
import json
from pathlib import Path

RESOURCE = Path(__file__).resolve().parents[1] / 'Sources/KwipMLCore/Resources/ProductBrand.json'
BRAND = json.loads(RESOURCE.read_text())
if set(BRAND) != {'name', 'formerName', 'publisher', 'tagline', 'businessTagline'} or any(
    not isinstance(value, str) or not value.strip() or len(value) > 120 for value in BRAND.values()
):
    raise ValueError('Invalid product presentation resource')
NAME = BRAND['name']
FORMER_NAME = BRAND['formerName']

if __name__ == '__main__':
    print(NAME)
