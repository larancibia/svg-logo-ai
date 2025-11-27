#!/usr/bin/env python3
"""
Purge Cloudflare cache for logos.guanacolabs.com
"""

import os
import requests

# Cloudflare config
API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
EMAIL = os.getenv('CLOUDFLARE_EMAIL')
ZONE_ID = "b76e19b10b96af97e3bf81b2eb8edd65"

if not API_TOKEN or not EMAIL:
    print("❌ Error: CLOUDFLARE_API_TOKEN or CLOUDFLARE_EMAIL not set")
    exit(1)

# API request
url = f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/purge_cache"
headers = {
    "X-Auth-Email": EMAIL,
    "X-Auth-Key": API_TOKEN,
    "Content-Type": "application/json"
}

# Purge everything
data = {"purge_everything": True}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("✅ Cloudflare cache purged successfully")
    print(f"   Zone: logos.guanacolabs.com")
    print(f"   URL: https://logos.guanacolabs.com/gallery.html")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
