#!/usr/bin/env python3
"""
Agrega DNS record en Cloudflare para logos.guanacolabs.com
"""

import os
import requests
import json

API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
EMAIL = os.getenv("CLOUDFLARE_EMAIL")

headers = {
    "X-Auth-Email": EMAIL,
    "X-Auth-Key": API_TOKEN,
    "Content-Type": "application/json"
}

base_url = "https://api.cloudflare.com/client/v4"

print("🌐 Configurando DNS para logos.guanacolabs.com")
print(f"Email: {EMAIL}\n")

# Step 1: Get Zone ID para guanacolabs.com
print("🔍 Obteniendo Zone ID...")
response = requests.get(
    f"{base_url}/zones?name=guanacolabs.com",
    headers=headers
)

data = response.json()
if not data.get("success") or not data.get("result"):
    print(f"❌ Error: {data.get('errors')}")
    exit(1)

zone_id = data["result"][0]["id"]
zone_name = data["result"][0]["name"]
print(f"✓ Zone: {zone_name}")
print(f"✓ Zone ID: {zone_id}\n")

# Step 2: Get server IP
import socket
server_ip = socket.gethostbyname(socket.gethostname())
print(f"🖥️  Server IP: {server_ip}")

# Better: get public IP
try:
    public_ip = requests.get('https://api.ipify.org').text
    print(f"🌍 Public IP: {public_ip}\n")
    server_ip = public_ip
except:
    print(f"⚠️  Usando IP local: {server_ip}\n")

# Step 3: Check if DNS record exists
print("🔍 Verificando DNS existente...")
response = requests.get(
    f"{base_url}/zones/{zone_id}/dns_records?name=logos.guanacolabs.com",
    headers=headers
)

existing = response.json().get("result", [])

if existing:
    record_id = existing[0]["id"]
    print(f"✓ Record encontrado: {existing[0]['content']}")
    print(f"   Actualizando a: {server_ip}...")

    # Update
    response = requests.put(
        f"{base_url}/zones/{zone_id}/dns_records/{record_id}",
        headers=headers,
        json={
            "type": "A",
            "name": "logos",
            "content": server_ip,
            "ttl": 1,
            "proxied": True
        }
    )
else:
    print(f"📝 Creando nuevo record A...")

    # Create
    response = requests.post(
        f"{base_url}/zones/{zone_id}/dns_records",
        headers=headers,
        json={
            "type": "A",
            "name": "logos",
            "content": server_ip,
            "ttl": 1,
            "proxied": True  # Orange cloud = CDN + SSL gratis
        }
    )

result = response.json()

if result.get("success"):
    print("✅ DNS configurado exitosamente!")
    print(f"\n   Tipo: A")
    print(f"   Nombre: logos.guanacolabs.com")
    print(f"   Apunta a: {server_ip}")
    print(f"   Proxied: ✅ (Cloudflare CDN + SSL)")
    print(f"\n🔒 SSL: Cloudflare Flexible/Full")
    print(f"⏱️  Propagación: ~1-2 minutos")
else:
    print(f"❌ Error: {result.get('errors')}")
