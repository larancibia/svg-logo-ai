"""
Generador de SVG v2 con principios profesionales de diseño
Incorpora hallazgos de investigación: Chain-of-Thought, Golden Ratio, Gestalt, etc.
"""

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from logo_examples import get_examples_by_industry, format_examples_for_prompt
from logo_metadata import LogoMetadata
from logo_validator import LogoValidator


@dataclass
class LogoRequest:
    """Especificación de un logo a generar"""
    company_name: str
    industry: str
    style: str  # "minimalist", "geometric", "modern", "organic"
    colors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    inspiration: Optional[str] = None
    target_complexity: int = 30  # Óptimo: 20-40


# Color psychology por industria
INDUSTRY_COLOR_PSYCHOLOGY = {
    "technology": {
        "primary": ["#2563eb", "#1e40af", "#3b82f6"],
        "meaning": "Confianza, innovación, profesionalismo"
    },
    "healthcare": {
        "primary": ["#10b981", "#059669", "#34d399"],
        "meaning": "Salud, crecimiento, vida"
    },
    "finance": {
        "primary": ["#1e3a8a", "#1e40af", "#1d4ed8"],
        "meaning": "Confianza, estabilidad, seguridad"
    },
    "food": {
        "primary": ["#ef4444", "#dc2626", "#f97316"],
        "meaning": "Apetito, energía, calidez"
    },
    "retail": {
        "primary": ["#7c3aed", "#6d28d9", "#8b5cf6"],
        "meaning": "Premium, creatividad, modernidad"
    }
}


class ProfessionalLogoGenerator:
    """
    Generador profesional de logos con principios de diseño avanzados
    """

    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self._init_vertex_ai()

    def _init_vertex_ai(self):
        """Inicializa Vertex AI"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project_id, location=self.location)
            # Usar Gemini 2.0 Flash si está disponible, sino Pro
            try:
                self.model = GenerativeModel("gemini-2.0-flash")
                print("✓ Usando Gemini 2.0 Flash")
            except:
                self.model = GenerativeModel("gemini-pro")
                print("✓ Usando Gemini Pro")

        except ImportError:
            print("❌ Error: Instala google-cloud-aiplatform")
            raise
        except Exception as e:
            print(f"❌ Error inicializando Vertex AI: {e}")
            raise

    def _get_color_recommendations(self, industry: str) -> Dict:
        """Obtiene recomendaciones de color basadas en psicología"""
        industry_lower = industry.lower()

        for key, colors in INDUSTRY_COLOR_PSYCHOLOGY.items():
            if key in industry_lower:
                return colors

        # Default: tech colors
        return INDUSTRY_COLOR_PSYCHOLOGY["technology"]

    def _build_advanced_prompt(self, request: LogoRequest) -> str:
        """Construye prompt avanzado con Chain-of-Thought y principios de diseño"""

        # Obtener ejemplos relevantes
        examples = get_examples_by_industry(request.industry, n=2)
        examples_text = format_examples_for_prompt(examples)

        # Obtener recomendaciones de color
        color_rec = self._get_color_recommendations(request.industry)

        colors_str = ", ".join(request.colors) if request.colors else ", ".join(color_rec["primary"])
        color_meaning = color_rec.get("meaning", "apropiados")

        keywords_str = ", ".join(request.keywords) if request.keywords else "ninguna"
        inspiration_str = request.inspiration or "diseño original"

        prompt = f"""Eres un diseñador de logos experto especializado en SVG vectorial profesional.

# TASK: Diseñar un logo profesional para "{request.company_name}"

## ESPECIFICACIONES DEL CLIENTE:
- **Nombre:** {request.company_name}
- **Industria:** {request.industry}
- **Estilo:** {request.style}
- **Colores sugeridos:** {colors_str} ({color_meaning})
- **Keywords:** {keywords_str}
- **Inspiración:** {inspiration_str}
- **Complejidad target:** {request.target_complexity} (óptimo: 20-40 elementos)

## PRINCIPIOS PROFESIONALES A APLICAR:

### 1. Golden Ratio (φ = 1.618)
- Usa proporciones basadas en φ para armonía visual
- Radio grande : radio pequeño = 1.618 : 1
- Ejemplo: si círculo exterior = 60, interior = 37 (60/1.618)

### 2. Teoría de Gestalt
- **Closure:** Formas que el cerebro completa mentalmente
- **Proximity:** Elementos cercanos vistos como grupo
- **Similarity:** Elementos similares percibidos juntos
- **Figure-Ground:** Espacio negativo creativo (como flecha en FedEx)
- **Continuation:** El ojo sigue direcciones lógicas

### 3. Psicología del Color
- Máximo 1-3 colores
- Color aumenta reconocimiento de marca en 80%
- {color_meaning}

### 4. Simplicidad
- Target: {request.target_complexity} elementos (óptimo 20-40)
- Logos top 100 promedian 32 elementos
- Debe funcionar en 16px (favicon) y 1024px (grande)

### 5. Balance
- Tipos: Symmetrical (formal), Asymmetrical (dinámico), Radial (unificado)
- Elige según personalidad de marca

## EJEMPLOS DE REFERENCIA (Few-Shot):

{examples_text}

## PROCESO (Chain-of-Thought Reasoning):

### ETAPA 1: ANÁLISIS CONCEPTUAL
Analiza:
- Significado del nombre "{request.company_name}"
- Valores de la industria {request.industry}
- Keywords: {keywords_str}
- Identifica 2-3 conceptos visuales clave
- Selecciona el concepto más fuerte y explica por qué

### ETAPA 2: DISEÑO ESTRUCTURAL
Define:
- Formas geométricas principales (círculos, rectángulos, paths)
- Aplicación de Golden Ratio (proporciones específicas)
- Principio de Gestalt dominante a usar
- Tipo de balance (symmetrical/asymmetrical/radial)
- Paleta de colores (1-3 colores) y su significado

### ETAPA 3: CONSTRUCCIÓN GEOMÉTRICA
Describe:
- Grid system o círculos guía
- Cálculos de proporción específicos
- Puntos clave de paths
- Relaciones matemáticas entre elementos

### ETAPA 4: GENERACIÓN DE CÓDIGO SVG
Requisitos técnicos:
- viewBox="0 0 200 200" (normalizado)
- Máximo {request.target_complexity} elementos principales
- Comentarios en secciones importantes
- IDs semánticos para accesibilidad
- Optimizado (2-3 decimales de precisión)
- Código limpio y estructurado

### ETAPA 5: VALIDACIÓN
Verifica:
- Simplicidad: ¿Está en rango 20-40?
- Escalabilidad: ¿Funciona en tamaños pequeños?
- Originalidad: ¿Es distintivo?
- Memorabilidad: ¿Es fácil de recordar?

## FORMATO DE RESPUESTA:

Usa EXACTAMENTE este formato:

---
## ETAPA 1: ANÁLISIS CONCEPTUAL
[Tu análisis detallado aquí]

**Concepto seleccionado:** [nombre del concepto]
**Rationale:** [por qué este concepto]

---
## ETAPA 2: DISEÑO ESTRUCTURAL
**Formas principales:** [lista]
**Golden Ratio aplicado:** [cómo y dónde]
**Gestalt principle:** [cuál y cómo]
**Balance:** [tipo]
**Colores:** [lista con significado]

---
## ETAPA 3: CONSTRUCCIÓN GEOMÉTRICA
[Descripción técnica de la construcción]

---
## ETAPA 4: CÓDIGO SVG
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Tu código SVG aquí -->
</svg>
```

---
## ETAPA 5: VALIDACIÓN
**Complejidad:** [número de elementos]
**Escalabilidad:** [evaluación]
**Puntuación estimada:** [0-100]

---

## RESTRICCIONES CRÍTICAS:
- Logo debe ser SIMPLE (target {request.target_complexity}, máximo 50 elementos)
- Máximo 3 colores
- NO usar texto/tipografía (solo formas)
- Código SVG DEBE ser válido
- viewBox DEBE ser "0 0 200 200"

¡Genera el logo profesional ahora siguiendo TODAS las etapas!"""

        return prompt

    def generate_logo(self, request: LogoRequest, verbose: bool = True) -> Dict:
        """
        Genera un logo con principios profesionales

        Returns:
            Dict con todas las etapas del reasoning + código SVG
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎨 Generando logo profesional para: {request.company_name}")
            print(f"{'='*70}")
            print(f"   Industria: {request.industry}")
            print(f"   Estilo: {request.style}")
            print(f"   Complejidad target: {request.target_complexity}")

        prompt = self._build_advanced_prompt(request)

        try:
            if verbose:
                print(f"\n🤖 Ejecutando Chain-of-Thought reasoning...")

            response = self.model.generate_content(prompt)
            result = self._parse_advanced_response(response.text)
            result['full_response'] = response.text
            result['request'] = request

            if verbose:
                print("✓ Reasoning completado")
                if result['has_valid_svg']:
                    print(f"✓ SVG válido generado")
                    print(f"   Complejidad: {result.get('complexity', 'N/A')}")
                    print(f"   Score estimado: {result.get('score', 'N/A')}/100")
                else:
                    print("⚠ No se pudo extraer SVG válido")

            return result

        except Exception as e:
            print(f"❌ Error generando logo: {e}")
            raise

    def _parse_advanced_response(self, response_text: str) -> Dict:
        """Parsea respuesta con todas las etapas"""

        result = {
            'stage1_analysis': self._extract_stage(response_text, "ETAPA 1"),
            'stage2_structure': self._extract_stage(response_text, "ETAPA 2"),
            'stage3_geometry': self._extract_stage(response_text, "ETAPA 3"),
            'stage4_svg': self._extract_svg_code(response_text),
            'stage5_validation': self._extract_stage(response_text, "ETAPA 5"),
            'complexity': self._extract_complexity(response_text),
            'score': self._extract_score(response_text)
        }

        result['svg_code'] = result['stage4_svg']
        result['has_valid_svg'] = result['svg_code'] is not None and '<svg' in result['svg_code']

        return result

    def _extract_stage(self, text: str, stage_name: str) -> Optional[str]:
        """Extrae contenido de una etapa específica"""
        pattern = rf"##\s*{stage_name}[:\s]+(.+?)(?=##|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_svg_code(self, text: str) -> Optional[str]:
        """Extrae código SVG"""
        patterns = [
            r"```xml\s*(<svg[^>]*>.*?</svg>)\s*```",
            r"```svg\s*(<svg[^>]*>.*?</svg>)\s*```",
            r"```\s*(<svg[^>]*>.*?</svg>)\s*```",
            r"(<svg[^>]*>.*?</svg>)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return None

    def _extract_complexity(self, text: str) -> Optional[int]:
        """Extrae complejidad del análisis"""
        match = re.search(r"Complejidad[:\s]+(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_score(self, text: str) -> Optional[int]:
        """Extrae score estimado"""
        match = re.search(r"Puntuación[^:]*:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def save_logo(self, result: Dict, base_filename: str, validate: bool = True):
        """Guarda logo, análisis y metadata"""
        output_dir = "../output"
        os.makedirs(output_dir, exist_ok=True)

        svg_path = None
        validation_results = None

        # Guardar SVG
        if result['has_valid_svg']:
            svg_path = os.path.join(output_dir, f"{base_filename}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(result['svg_code'])
            print(f"✓ SVG guardado: {svg_path}")

            # Validar si se solicita
            if validate:
                validator = LogoValidator()
                validation_results = validator.validate_all(result['svg_code'])
                print(f"   Validation Score: {validation_results['final_score']}/100")

        # Guardar análisis completo
        analysis_path = os.path.join(output_dir, f"{base_filename}_analysis.md")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"# Análisis de Logo: {result['request'].company_name}\n\n")
            f.write(f"**Industria:** {result['request'].industry}\n")
            f.write(f"**Estilo:** {result['request'].style}\n")
            f.write(f"**Complejidad:** {result.get('complexity', 'N/A')}\n")
            f.write(f"**Score:** {result.get('score', 'N/A')}/100\n\n")
            if validation_results:
                f.write(f"**Validation Score:** {validation_results['final_score']}/100\n\n")
            f.write("---\n\n")
            f.write(result['full_response'])

        print(f"✓ Análisis guardado: {analysis_path}")

        # Guardar metadata
        metadata = LogoMetadata()

        # Determinar número de iteración
        existing = metadata.get_by_company(result['request'].company_name)
        iteration = len(existing) + 1

        logo_id = metadata.add_logo(
            filename=f"{base_filename}.svg",
            company_name=result['request'].company_name,
            industry=result['request'].industry,
            style=result['request'].style,
            score=validation_results['final_score'] if validation_results else result.get('score', 0),
            complexity=result.get('complexity', 0),
            version="v2",
            iteration=iteration,
            colors=result['request'].colors or [],
            notes=f"Generated with Chain-of-Thought reasoning",
            validation_results=validation_results
        )

        print(f"✓ Metadata guardada (ID: {logo_id}, Iteración: {iteration})")

        return svg_path, analysis_path, logo_id


def demo():
    """Demo del generador mejorado"""

    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("❌ Error: Define GCP_PROJECT_ID")
        print("   export GCP_PROJECT_ID=tu-project-id")
        return

    generator = ProfessionalLogoGenerator(project_id=project_id)

    # Ejemplo 1: Tech startup
    print("\n" + "="*70)
    print("DEMO 1: Tech Startup")
    print("="*70)

    request1 = LogoRequest(
        company_name="QuantumFlow",
        industry="AI/Technology",
        style="minimalist",
        colors=["#2563eb", "#3b82f6"],
        keywords=["flow", "quantum", "innovation", "future"],
        target_complexity=28
    )

    result1 = generator.generate_logo(request1)

    if result1['has_valid_svg']:
        generator.save_logo(result1, "quantumflow_logo")

        print("\n" + "-"*70)
        print("ANÁLISIS CONCEPTUAL:")
        print("-"*70)
        print(result1['stage1_analysis'][:500] + "...")

        print("\n" + "-"*70)
        print("DISEÑO ESTRUCTURAL:")
        print("-"*70)
        print(result1['stage2_structure'][:500] + "...")

    # Ejemplo 2: Healthcare
    print("\n\n" + "="*70)
    print("DEMO 2: Healthcare Company")
    print("="*70)

    request2 = LogoRequest(
        company_name="VitalCare",
        industry="Healthcare",
        style="modern",
        colors=["#10b981", "#059669"],
        keywords=["care", "health", "life", "growth"],
        target_complexity=32
    )

    result2 = generator.generate_logo(request2)

    if result2['has_valid_svg']:
        generator.save_logo(result2, "vitalcare_logo")

    print("\n" + "="*70)
    print("✅ Demo completado. Revisa la carpeta output/")
    print("="*70)


if __name__ == "__main__":
    demo()
