#!/usr/bin/env python3
"""
Deploy directo a Cloudflare Pages usando las credenciales del entorno
"""

import os
import sys
import requests
import json
from pathlib import Path

# Credenciales del entorno
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
EMAIL = os.getenv("CLOUDFLARE_EMAIL")

if not API_TOKEN:
    print("❌ Error: CLOUDFLARE_API_TOKEN no encontrado en variables de entorno")
    sys.exit(1)

print("🔑 Credenciales encontradas")
print(f"   Email: {EMAIL}")
print(f"   Token: {API_TOKEN[:10]}...{API_TOKEN[-4:]}")

# Headers - probar ambos métodos de auth
headers_token = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

headers_legacy = {
    "X-Auth-Email": EMAIL,
    "X-Auth-Key": API_TOKEN,
    "Content-Type": "application/json"
}

base_url = "https://api.cloudflare.com/client/v4"

print("\n🔍 Obteniendo Account ID...")

# Intentar con Bearer token primero
response = requests.get(f"{base_url}/accounts", headers=headers_token)

if not response.json().get("success"):
    print("   Probando con auth legacy...")
    response = requests.get(f"{base_url}/accounts", headers=headers_legacy)
    headers = headers_legacy
else:
    headers = headers_token

data = response.json()

if not data.get("success"):
    print(f"❌ Error: {data.get('errors')}")
    sys.exit(1)

accounts = data.get("result", [])
if not accounts:
    print("❌ No se encontraron cuentas")
    sys.exit(1)

account = accounts[0]
account_id = account["id"]
account_name = account["name"]

print(f"✓ Account encontrado: {account_name}")
print(f"✓ Account ID: {account_id}")

# Usar wrangler con las credenciales
deploy_dir = Path("output/deploy")

if not deploy_dir.exists():
    print(f"❌ Directorio {deploy_dir} no encontrado")
    sys.exit(1)

print(f"\n📦 Archivos en {deploy_dir}:")
for f in deploy_dir.iterdir():
    if f.is_file():
        print(f"   ✓ {f.name}")

# Setear variables para wrangler
os.environ["CLOUDFLARE_ACCOUNT_ID"] = account_id

print(f"\n🚀 Deployando a Cloudflare Pages...")
print(f"   Proyecto: logo-gallery-ai")
print(f"   Directorio: {deploy_dir}")

# Ejecutar wrangler
import subprocess

try:
    result = subprocess.run(
        ["npx", "wrangler", "pages", "deploy", str(deploy_dir),
         "--project-name=logo-gallery-ai",
         "--branch=main"],
        capture_output=True,
        text=True,
        env={**os.environ, "CLOUDFLARE_ACCOUNT_ID": account_id}
    )

    print(result.stdout)

    if result.returncode == 0:
        print("\n" + "="*70)
        print("✅ DEPLOYMENT EXITOSO")
        print("="*70)
        print(f"\n🌍 URL: https://logo-gallery-ai.pages.dev")
        print(f"📁 Proyecto: logo-gallery-ai")
        print("\n" + "="*70)
    else:
        print(f"\n❌ Error en deployment:")
        print(result.stderr)

except Exception as e:
    print(f"❌ Error ejecutando wrangler: {e}")
    print("\n💡 Alternativa: Usa el drag & drop manual")
    print("   1. Ve a: https://dash.cloudflare.com")
    print("   2. Workers & Pages → Create → Upload assets")
    print(f"   3. Sube: {deploy_dir.absolute()}")
