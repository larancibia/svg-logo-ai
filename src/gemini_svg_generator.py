"""
Generador de SVG usando Gemini (Vertex AI)
Implementación inicial para generar logos vectoriales con chain-of-thought reasoning
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LogoRequest:
    """Especificación de un logo a generar"""
    company_name: str
    industry: str
    style: str  # "minimalist", "geometric", "modern", "organic"
    colors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    inspiration: Optional[str] = None


class GeminiSVGGenerator:
    """
    Genera SVG usando Gemini con chain-of-thought reasoning

    Requiere:
        pip install google-cloud-aiplatform
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
    """

    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Inicializa el generador con credenciales de GCP

        Args:
            project_id: ID del proyecto GCP
            location: Región de Vertex AI
        """
        self.project_id = project_id
        self.location = location
        self._init_vertex_ai()

    def _init_vertex_ai(self):
        """Inicializa Vertex AI"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel("gemini-pro")
            print("✓ Vertex AI inicializado")
        except ImportError:
            print("❌ Error: Instala google-cloud-aiplatform")
            print("   pip install google-cloud-aiplatform")
            raise
        except Exception as e:
            print(f"❌ Error inicializando Vertex AI: {e}")
            print("   Verifica GOOGLE_APPLICATION_CREDENTIALS")
            raise

    def _build_prompt(self, request: LogoRequest) -> str:
        """Construye el prompt con chain-of-thought reasoning"""

        colors_str = ", ".join(request.colors) if request.colors else "apropiados para la industria"
        keywords_str = ", ".join(request.keywords) if request.keywords else "ninguna"
        inspiration_str = request.inspiration or "diseño original"

        prompt = f"""Eres un diseñador de logos experto especializado en SVG vectorial.

Tu tarea: Diseñar un logo profesional para "{request.company_name}"

ESPECIFICACIONES:
- Industria: {request.industry}
- Estilo: {request.style}
- Colores: {colors_str}
- Keywords: {keywords_str}
- Inspiración: {inspiration_str}

PROCESO (Chain-of-Thought):

1. ANÁLISIS Y CONCEPTO:
   - Analiza la industria y el nombre de la empresa
   - Identifica 2-3 conceptos visuales clave
   - Explica el razonamiento detrás del concepto elegido

2. ESTRUCTURA GEOMÉTRICA:
   - Define las formas geométricas principales (círculos, rectángulos, paths)
   - Explica las proporciones y simetría
   - Describe la composición (centrado, asimétrico, etc.)

3. GENERACIÓN DE CÓDIGO SVG:
   - Escribe código SVG limpio y bien estructurado
   - Usa viewBox para escalabilidad
   - Agrupa elementos relacionados con <g>
   - Comenta secciones importantes

FORMATO DE RESPUESTA:

## 1. Concepto
[Tu análisis conceptual]

## 2. Estructura Geométrica
[Descripción de formas y proporciones]

## 3. Código SVG
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Tu código SVG aquí -->
</svg>
```

RESTRICCIONES:
- Logo debe ser simple (máximo 5-7 elementos geométricos principales)
- Colores: máximo 3 colores
- Debe funcionar bien en tamaños pequeños (favicon) y grandes
- NO usar texto/fuentes (solo formas)
- Código SVG debe ser válido y optimizado

Genera el logo ahora:"""

        return prompt

    def generate_logo(self, request: LogoRequest) -> Dict:
        """
        Genera un logo basado en la especificación

        Args:
            request: Especificación del logo

        Returns:
            Dict con: reasoning, structure, svg_code, full_response
        """
        print(f"\n🎨 Generando logo para: {request.company_name}")
        print(f"   Estilo: {request.style} | Industria: {request.industry}")

        prompt = self._build_prompt(request)

        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            result['full_response'] = response.text

            print("✓ Logo generado exitosamente")
            return result

        except Exception as e:
            print(f"❌ Error generando logo: {e}")
            raise

    def _parse_response(self, response_text: str) -> Dict:
        """Parsea la respuesta de Gemini para extraer componentes"""

        # Extraer código SVG
        svg_code = None
        if "```xml" in response_text:
            start = response_text.find("```xml") + 6
            end = response_text.find("```", start)
            svg_code = response_text[start:end].strip()
        elif "```svg" in response_text:
            start = response_text.find("```svg") + 6
            end = response_text.find("```", start)
            svg_code = response_text[start:end].strip()
        elif "<svg" in response_text:
            start = response_text.find("<svg")
            end = response_text.find("</svg>") + 6
            svg_code = response_text[start:end].strip()

        # Extraer secciones
        concept = self._extract_section(response_text, "Concepto")
        structure = self._extract_section(response_text, "Estructura")

        return {
            'reasoning': concept,
            'structure': structure,
            'svg_code': svg_code,
            'has_valid_svg': svg_code is not None and '<svg' in svg_code
        }

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extrae una sección específica del texto"""
        markers = [
            f"## {section_name}",
            f"### {section_name}",
            f"**{section_name}**"
        ]

        for marker in markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # Buscar el siguiente marker
                next_marker_pos = len(text)
                for next_marker in ["##", "```"]:
                    pos = text.find(next_marker, start + 10)
                    if pos != -1 and pos < next_marker_pos:
                        next_marker_pos = pos

                return text[start:next_marker_pos].strip()

        return None

    def save_svg(self, svg_code: str, filename: str):
        """Guarda el código SVG en un archivo"""
        output_dir = "../output"
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_code)

        print(f"✓ SVG guardado en: {filepath}")
        return filepath


def demo():
    """Demo de uso del generador"""

    # Configurar
    project_id = os.getenv("GCP_PROJECT_ID")

    if not project_id:
        print("❌ Error: Define GCP_PROJECT_ID en .env o export GCP_PROJECT_ID=tu-project-id")
        return

    # Crear generador
    generator = GeminiSVGGenerator(project_id=project_id)

    # Ejemplo 1: Logo tech startup
    request1 = LogoRequest(
        company_name="TechFlow",
        industry="Software/Technology",
        style="minimalist",
        colors=["#2563eb", "#1e40af"],
        keywords=["flow", "connection", "innovation"]
    )

    result = generator.generate_logo(request1)

    print("\n" + "="*60)
    print("CONCEPTO:")
    print("="*60)
    print(result.get('reasoning', 'N/A'))

    print("\n" + "="*60)
    print("ESTRUCTURA:")
    print("="*60)
    print(result.get('structure', 'N/A'))

    if result['has_valid_svg']:
        print("\n✓ SVG generado correctamente")
        generator.save_svg(result['svg_code'], "techflow_logo.svg")
    else:
        print("\n❌ No se pudo extraer SVG válido")

    # Ejemplo 2: Logo health startup
    request2 = LogoRequest(
        company_name="HealthHub",
        industry="Healthcare",
        style="modern",
        colors=["#10b981", "#059669"],
        keywords=["health", "care", "connection"]
    )

    result2 = generator.generate_logo(request2)

    if result2['has_valid_svg']:
        generator.save_svg(result2['svg_code'], "healthhub_logo.svg")


if __name__ == "__main__":
    demo()
