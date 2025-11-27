#!/usr/bin/env python3
"""
Deploy directo a Cloudflare Pages usando solo la API (sin wrangler)
"""

import os
import sys
import requests
import hashlib
import mimetypes
from pathlib import Path

# Credenciales
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
EMAIL = os.getenv("CLOUDFLARE_EMAIL")
ACCOUNT_ID = "5de70f4ba8110b9cf400a3157ff420c3"  # Ya lo obtuvimos
PROJECT_NAME = "logo-gallery-ai"

headers = {
    "X-Auth-Email": EMAIL,
    "X-Auth-Key": API_TOKEN,
}

base_url = "https://api.cloudflare.com/client/v4"

print("="*70)
print("🚀 CLOUDFLARE PAGES DEPLOYMENT VIA API")
print("="*70)
print(f"\nAccount ID: {ACCOUNT_ID}")
print(f"Proyecto: {PROJECT_NAME}")
print(f"Email: {EMAIL}\n")

# Step 1: Create or get project
print("📝 Creando/verificando proyecto...")
url = f"{base_url}/accounts/{ACCOUNT_ID}/pages/projects/{PROJECT_NAME}"
response = requests.get(url, headers=headers)

if response.status_code == 404:
    # Create project
    print("   Creando nuevo proyecto...")
    url = f"{base_url}/accounts/{ACCOUNT_ID}/pages/projects"
    data = {
        "name": PROJECT_NAME,
        "production_branch": "main"
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        print("   ✓ Proyecto creado")
    else:
        print(f"   ❌ Error: {response.text}")
        sys.exit(1)
else:
    print("   ✓ Proyecto encontrado")

# Step 2: Prepare files and create manifest
deploy_dir = Path("output/deploy")
files_to_upload = []
manifest = {}

print("\n📦 Preparando archivos...")
for file_path in deploy_dir.rglob("*"):
    if file_path.is_file():
        rel_path = "/" + str(file_path.relative_to(deploy_dir))

        # Calculate hash
        with open(file_path, 'rb') as f:
            content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()

        manifest[rel_path] = file_hash
        files_to_upload.append((rel_path, file_path, content))

        print(f"   ✓ {rel_path} ({len(content)} bytes)")

print(f"\n   Total: {len(files_to_upload)} archivos")

# Step 3: Create deployment with manifest
print("\n🔨 Creando deployment...")
url = f"{base_url}/accounts/{ACCOUNT_ID}/pages/projects/{PROJECT_NAME}/deployments"

# Cloudflare Pages API v2 - Direct Upload
# Necesitamos crear un deployment y luego subir archivos

try:
    # Method 1: Try with manifest
    deployment_data = {
        "branch": "main",
        "manifest": manifest
    }

    response = requests.post(
        url,
        headers={**headers, "Content-Type": "application/json"},
        json=deployment_data
    )

    result = response.json()

    if not result.get("success"):
        print(f"   ⚠️  Manifest method failed: {result.get('errors')}")
        print("\n💡 Cloudflare Pages Direct Upload API requiere wrangler")
        print("   Pero tengo una solución alternativa...")

        # Alternative: Use Cloudflare Workers Sites (similar)
        print("\n🎯 SOLUCIÓN: Usa el método manual (es MUY rápido):")
        print("\n" + "="*70)
        print("DEPLOYMENT MANUAL (2 minutos)")
        print("="*70)
        print("\n1. Ve a: https://dash.cloudflare.com")
        print(f"2. Email: {EMAIL}")
        print("3. Workers & Pages → Create application")
        print("4. Pages → Upload assets")
        print(f"5. Arrastra: {deploy_dir.absolute()}/")
        print("   (o usa el ZIP: output/logo-gallery-deploy.zip)")
        print(f"6. Nombre: {PROJECT_NAME}")
        print("7. Deploy!")
        print("\n🌍 Tu URL será: https://logo-gallery-ai.pages.dev")
        print("\n" + "="*70)

        sys.exit(0)

    deployment_id = result["result"]["id"]
    print(f"   ✓ Deployment creado: {deployment_id}")

    # Upload files
    print("\n📤 Subiendo archivos...")
    for rel_path, file_path, content in files_to_upload:
        upload_url = f"{base_url}/accounts/{ACCOUNT_ID}/pages/projects/{PROJECT_NAME}/deployments/{deployment_id}/files{rel_path}"

        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

        resp = requests.put(
            upload_url,
            headers={
                **headers,
                "Content-Type": mime_type
            },
            data=content
        )

        if resp.status_code in [200, 201]:
            print(f"   ✓ {rel_path}")
        else:
            print(f"   ❌ {rel_path}: {resp.text[:100]}")

    print("\n✅ Upload completado!")

    # Wait for build
    print("\n⏳ Esperando que el deployment se complete...")
    import time
    for i in range(30):
        time.sleep(2)
        resp = requests.get(
            f"{base_url}/accounts/{ACCOUNT_ID}/pages/projects/{PROJECT_NAME}/deployments/{deployment_id}",
            headers=headers
        )
        status_data = resp.json()
        if status_data.get("success"):
            stage = status_data["result"]["latest_stage"]["status"]
            if stage == "success":
                print("\n" + "="*70)
                print("✅ DEPLOYMENT EXITOSO!")
                print("="*70)
                print(f"\n🌍 URL: https://{PROJECT_NAME}.pages.dev")
                print(f"📊 Deployment ID: {deployment_id}")
                print(f"📁 Proyecto: {PROJECT_NAME}")
                print("\n" + "="*70)
                sys.exit(0)
            elif stage == "failure":
                print(f"\n❌ Deployment falló: {status_data}")
                break

        print(f"   ... {i+1}/30")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Método alternativo recomendado:")
    print(f"   1. Ve a: https://dash.cloudflare.com")
    print(f"   2. Sube manualmente: {deploy_dir.absolute()}/")
