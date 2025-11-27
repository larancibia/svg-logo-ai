"""
Sistema de deployment automático a Cloudflare Pages
Sube la galería de logos con subdominio personalizado
"""

import os
import json
import requests
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv


class CloudflareDeployer:
    """Deploy galería a Cloudflare Pages"""

    def __init__(self, env_file: str = "../.env.cloudflare"):
        load_dotenv(env_file)

        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.zone_id = os.getenv("CLOUDFLARE_ZONE_ID")
        self.project_name = os.getenv("CLOUDFLARE_PROJECT_NAME", "logo-gallery-ai")
        self.custom_domain = os.getenv("CUSTOM_DOMAIN")

        if not self.api_token or not self.account_id:
            raise ValueError("CLOUDFLARE_API_TOKEN y CLOUDFLARE_ACCOUNT_ID son requeridos")

        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def prepare_site(self, output_dir: str = "../output/deploy") -> Dict:
        """
        Prepara el sitio estático para deployment

        Returns:
            Dict con manifest de archivos
        """
        print("📦 Preparando sitio para deployment...")

        # Crear directorio de deployment
        deploy_path = Path(output_dir)
        deploy_path.mkdir(parents=True, exist_ok=True)

        # Copiar galería HTML
        gallery_src = Path("../output/gallery.html")
        gallery_dst = deploy_path / "index.html"

        if not gallery_src.exists():
            print("⚠️  Generando galería...")
            from gallery_generator import GalleryGenerator
            generator = GalleryGenerator()
            generator.generate_gallery()

        # Copiar HTML
        import shutil
        shutil.copy(gallery_src, gallery_dst)
        print(f"✓ Copiado: gallery.html → index.html")

        # Copiar SVGs
        svg_dir = Path("../output")
        svg_files = list(svg_dir.glob("*.svg"))

        if svg_files:
            for svg in svg_files:
                shutil.copy(svg, deploy_path / svg.name)
            print(f"✓ Copiados: {len(svg_files)} archivos SVG")

        # Copiar metadata JSON
        metadata_src = svg_dir / "logos_metadata.json"
        if metadata_src.exists():
            shutil.copy(metadata_src, deploy_path / "logos_metadata.json")
            print(f"✓ Copiado: logos_metadata.json")

        # Crear manifest
        manifest = self._create_manifest(deploy_path)
        print(f"✓ Manifest creado: {len(manifest)} archivos")

        return manifest

    def _create_manifest(self, deploy_path: Path) -> Dict:
        """Crea manifest de archivos con hashes"""
        manifest = {}

        for file_path in deploy_path.rglob("*"):
            if file_path.is_file():
                # Path relativo
                rel_path = str(file_path.relative_to(deploy_path))

                # Calcular hash SHA-256
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                manifest[f"/{rel_path}"] = file_hash

        return manifest

    def create_or_get_project(self) -> Dict:
        """Crea o obtiene el proyecto en Cloudflare Pages"""
        print(f"\n🔍 Verificando proyecto '{self.project_name}'...")

        # Check if project exists
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            print(f"✓ Proyecto encontrado")
            return response.json()["result"]

        # Create new project
        print(f"📝 Creando nuevo proyecto...")
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects"

        data = {
            "name": self.project_name,
            "production_branch": "main",
            "build_config": {
                "build_command": "",
                "destination_dir": ".",
                "root_dir": ""
            }
        }

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code in [200, 201]:
            print(f"✓ Proyecto creado")
            return response.json()["result"]
        else:
            raise Exception(f"Error creando proyecto: {response.text}")

    def upload_files(self, deploy_path: Path, manifest: Dict) -> str:
        """
        Sube archivos a Cloudflare Pages usando Direct Upload

        Returns:
            deployment_id
        """
        print(f"\n📤 Iniciando deployment...")

        # Step 1: Create deployment
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}/deployments"

        # Preparar form data
        files_data = []

        for file_path in deploy_path.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(deploy_path))

                # Leer contenido
                with open(file_path, 'rb') as f:
                    content = f.read()

                # Determinar content-type
                content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

                files_data.append((
                    rel_path,
                    (rel_path, content, content_type)
                ))

        # Upload usando multipart/form-data
        print(f"   Subiendo {len(files_data)} archivos...")

        # Cloudflare Pages requiere un approach diferente
        # Usar wrangler o Direct Upload API v2
        return self._direct_upload_v2(deploy_path, manifest)

    def _direct_upload_v2(self, deploy_path: Path, manifest: Dict) -> str:
        """Upload usando Direct Upload API v2"""

        # Step 1: Crear deployment
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}/deployments"

        response = requests.post(
            url,
            headers=self.headers,
            json={"manifest": manifest}
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Error creando deployment: {response.text}")

        result = response.json()["result"]
        upload_url = result.get("upload_form", {}).get("url")
        deployment_id = result["id"]

        print(f"✓ Deployment creado: {deployment_id}")

        if not upload_url:
            # Direct upload
            for file_path in deploy_path.rglob("*"):
                if file_path.is_file():
                    rel_path = "/" + str(file_path.relative_to(deploy_path))
                    self._upload_single_file(deployment_id, rel_path, file_path)

        return deployment_id

    def _upload_single_file(self, deployment_id: str, path: str, file_path: Path):
        """Sube un archivo individual"""
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}/deployments/{deployment_id}/files/{path}"

        with open(file_path, 'rb') as f:
            response = requests.put(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                },
                data=f.read()
            )

        if response.status_code not in [200, 201]:
            print(f"⚠️  Error subiendo {path}: {response.text}")

    def finalize_deployment(self, deployment_id: str):
        """Finaliza el deployment"""
        print(f"\n✅ Finalizando deployment...")

        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}/deployments/{deployment_id}"

        # Wait for deployment to be ready
        import time
        max_attempts = 30

        for attempt in range(max_attempts):
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                result = response.json()["result"]
                status = result["latest_stage"]["status"]

                if status == "success":
                    print(f"✓ Deployment completado!")
                    return result
                elif status == "failure":
                    print(f"❌ Deployment falló: {result.get('latest_stage', {}).get('message')}")
                    return result

            time.sleep(2)
            print(f"   Esperando... ({attempt + 1}/{max_attempts})")

        raise Exception("Timeout esperando deployment")

    def setup_custom_domain(self):
        """Configura custom domain (requiere CLOUDFLARE_ZONE_ID)"""
        if not self.custom_domain or not self.zone_id:
            print("\n⚠️  Custom domain no configurado (opcional)")
            return

        print(f"\n🌐 Configurando custom domain: {self.custom_domain}")

        # Add custom domain to Pages project
        url = f"{self.base_url}/accounts/{self.account_id}/pages/projects/{self.project_name}/domains"

        data = {"name": self.custom_domain}

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code in [200, 201]:
            print(f"✓ Domain agregado")

            # Get DNS instructions
            result = response.json()["result"]
            if "cname_target" in result:
                print(f"\n📝 Configuración DNS:")
                print(f"   Tipo: CNAME")
                print(f"   Nombre: {self.custom_domain.split('.')[0]}")
                print(f"   Target: {result['cname_target']}")
                print(f"\n   O usa Cloudflare DNS automático si el dominio está en CF")
        else:
            print(f"⚠️  Error configurando domain: {response.text}")

    def deploy(self) -> str:
        """
        Deployment completo

        Returns:
            URL del sitio deployado
        """
        print("="*70)
        print("🚀 CLOUDFLARE PAGES DEPLOYMENT")
        print("="*70)

        try:
            # 1. Preparar sitio
            manifest = self.prepare_site()

            # 2. Crear/obtener proyecto
            project = self.create_or_get_project()

            # 3. Upload files
            deploy_path = Path("../output/deploy")
            deployment_id = self.upload_files(deploy_path, manifest)

            # 4. Finalizar
            result = self.finalize_deployment(deployment_id)

            # 5. Custom domain
            self.setup_custom_domain()

            # URL del sitio
            site_url = f"https://{self.project_name}.pages.dev"
            if self.custom_domain:
                site_url = f"https://{self.custom_domain}"

            print("\n" + "="*70)
            print("✅ DEPLOYMENT COMPLETADO")
            print("="*70)
            print(f"\n🌍 URL: {site_url}")
            print(f"📊 Deployment ID: {deployment_id}")
            print(f"📁 Proyecto: {self.project_name}")
            print("\n" + "="*70)

            return site_url

        except Exception as e:
            print(f"\n❌ Error en deployment: {e}")
            raise


def main():
    """Deploy a Cloudflare Pages"""

    # Check if .env.cloudflare exists
    if not os.path.exists("../.env.cloudflare"):
        print("❌ Error: No se encontró .env.cloudflare")
        print("\n1. Copia .env.cloudflare.example a .env.cloudflare")
        print("2. Completa con tus credenciales de Cloudflare")
        print("\nVer: https://dash.cloudflare.com/profile/api-tokens")
        return

    deployer = CloudflareDeployer()
    url = deployer.deploy()

    print(f"\n🎉 ¡Galería deployada exitosamente!")
    print(f"🔗 Abre: {url}")


if __name__ == "__main__":
    main()
