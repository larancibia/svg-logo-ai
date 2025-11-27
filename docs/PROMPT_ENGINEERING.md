# Prompt Engineering para Generación de SVG con LLMs

**Guía técnica de técnicas avanzadas de prompt engineering para generación de logos y gráficos vectoriales usando GPT-4, Claude y Gemini.**

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Estado del Arte en SVG + LLMs](#estado-del-arte-en-svg--llms)
3. [Prompt Patterns Efectivos](#prompt-patterns-efectivos)
4. [Técnicas Específicas para SVG](#técnicas-específicas-para-svg)
5. [Evaluación y Feedback Loops](#evaluación-y-feedback-loops)
6. [Casos de Éxito y Proyectos Open Source](#casos-de-éxito-y-proyectos-open-source)
7. [Prompt Templates Listos para Usar](#prompt-templates-listos-para-usar)
8. [Comparación de Técnicas](#comparación-de-técnicas)
9. [Recomendaciones para Nuestro Proyecto](#recomendaciones-para-nuestro-proyecto)

---

## Introducción

Los LLMs modernos (GPT-4, Claude Sonnet 3.5, Gemini 2.0 Flash) muestran capacidad para generar código SVG, pero con limitaciones importantes:

**Desafíos principales:**
- Generación inconsistente de SVG válido
- Dificultad con geometrías complejas
- Problemas con coordenadas y proporciones
- Falta de optimización de paths

**Hallazgos clave de la investigación:**
> "LLMs are really, really bad at generating SVG code" - TextGrad Study (2024)

Sin embargo, con técnicas apropiadas de prompt engineering, podemos mejorar significativamente los resultados.

---

## Estado del Arte en SVG + LLMs

### Modelos Destacados (2024-2025)

Según el benchmark **SVGenius**, los modelos de mejor rendimiento son:

**Propietarios:**
- **Claude 3.7 Sonnet** - Mejor para artifacts y visualización interactiva
- **Gemini 2.0 Flash** - Excelente balance velocidad/calidad
- **GPT-4o** - Capacidades robustas de generación

**Open Source:**
- **DeepSeek-R1** - Razonamiento avanzado
- **QwQ-32B** - Comprensión visual
- **Qwen3-32B** - Multimodal

### Investigación Reciente

#### **LLM4SVG** (CVPR 2025)
- **Dataset:** 250k SVGs + 580k instrucciones SVG-Text-Image
- **Innovación:** 55 tokens semánticos especializados (15 tags, 30 atributos, 10 comandos path)
- **GitHub:** https://github.com/ximinng/LLM4SVG

#### **Chat2SVG** (CVPR 2025)
- **Enfoque:** Hybrid LLM + Diffusion Models
- **Prompting:** Expansión en 3 capas (scene/object/layout)
- **GitHub:** https://chat2svg.github.io/

#### **OmniSVG** (NeurIPS 2025)
- **Dataset:** MMSVG-2M (2 millones de SVGs anotados)
- **Capacidad:** Desde íconos simples hasta personajes anime complejos
- **GitHub:** https://github.com/OmniSVG/OmniSVG

#### **StarVector**
- **Base:** StarCoder (código) + adaptador multimodal
- **Modelos:** StarVector-8B, StarVector-1B
- **Hugging Face:** starvector/starvector-8b-im2svg

#### **Reason-SVG** (2025)
- **Paradigma:** Drawing-with-Thought (DwT)
- **Entrenamiento:** Supervised FT + Reinforcement Learning con hybrid reward

---

## Prompt Patterns Efectivos

### 1. Chain-of-Thought (CoT) para Diseño

**Concepto:** Descomponer el diseño en pasos de razonamiento intermedio.

#### Ejemplo básico:

```
Quiero un logo de una empresa de tecnología.

Antes de generar el SVG, analiza:
1. ¿Qué elementos visuales representan tecnología? (circuitos, redes, datos)
2. ¿Qué formas geométricas transmiten profesionalismo? (geometría limpia, simetría)
3. ¿Qué paleta de colores es apropiada? (azules, grises, tonos tech)
4. ¿Cómo organizar el layout? (centrado, proporción áurea)

Luego genera el SVG siguiendo tu análisis.
```

#### Variantes avanzadas de CoT:

**Contrastive Denoising CoT (CD-CoT):**
Mejora 17.8% accuracy contrastando razonamientos defectuosos vs correctos.

```
Genera un logo circular.

CORRECTO: "Usaré <circle> con cx, cy y r apropiados"
INCORRECTO: "Dibujaré un círculo con <path> usando curvas Bézier complejas"

Explica tu razonamiento y elige el enfoque correcto.
```

**Logic-of-Thought (LoT):**
Incorpora lógica proposicional formal.

```
Reglas lógicas para diseño de logo:
- SI contiene texto ENTONCES usar <text> O <path> (texto a curvas)
- SI requiere simetría ENTONCES usar transformaciones (rotate, scale)
- SI múltiples colores ENTONCES definir <defs> con gradientes

Aplica estas reglas para generar un logo de [descripción].
```

### 2. Few-Shot Learning con SVG

**Concepto:** Proporcionar ejemplos de SVG de calidad como referencia.

#### Template Few-Shot:

```
Soy un generador SVG experto. Aquí hay ejemplos de mi trabajo:

EJEMPLO 1 - Logo minimalista circular:
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="#2563eb"/>
  <path d="M 50 30 L 50 70" stroke="white" stroke-width="4"/>
  <path d="M 30 50 L 70 50" stroke="white" stroke-width="4"/>
</svg>

EJEMPLO 2 - Logo con path geométrico:
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <polygon points="50,10 90,35 75,80 25,80 10,35" fill="#10b981"/>
</svg>

EJEMPLO 3 - Logo con gradiente:
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100" height="100" rx="20" fill="url(#grad1)"/>
</svg>

Ahora genera un logo de [TU DESCRIPCIÓN] siguiendo el mismo nivel de calidad y estructura.
```

#### Hallazgos de Investigación:

Según estudios con GPT-4:
- **Zero-shot:** Rendimiento base
- **One-shot (1 ejemplo):** +15% precisión
- **Few-shot (3 ejemplos):** +28% precisión

### 3. Drawing-with-Thought (DwT) - Reason-SVG

**Paradigma:** Planning-then-drawing con razonamiento visual explícito.

#### Las 6 Etapas:

```
Genera un logo de [DESCRIPCIÓN] siguiendo este proceso:

1. CONCEPT SKETCHING
   - Identifica componentes visuales clave
   - Lista: [componente1, componente2, ...]

2. CANVAS PLANNING
   - viewBox: [determina dimensiones]
   - Layout: [estructura espacial]
   - Proporciones: [ratios y espaciado]

3. SHAPE DECOMPOSITION
   - Descompón en primitivas geométricas
   - circle: [usos]
   - rect: [usos]
   - polygon: [usos]
   - path: [solo si necesario para formas complejas]

4. COORDINATE CALCULATION
   - Calcula puntos de control
   - Posicionamiento preciso
   - Alineación y simetría

5. STYLING & COLORING
   - Paleta de colores
   - Gradientes (si aplica)
   - Strokes y fills

6. FINAL ASSEMBLY
   - Integra todos los elementos
   - Agrupa con <g> y IDs semánticos
   - Genera SVG final
```

#### Ejemplo Concreto:

```
Prompt: "A minimalist icon of a steaming coffee cup, flat design"

1. CONCEPT SKETCHING:
   - Cup body (main container)
   - Handle (curved)
   - Steam lines (3 wavy paths)
   - Saucer (optional base)

2. CANVAS PLANNING:
   - viewBox: 0 0 100 100 (square canvas)
   - Cup centered: x=30-70, y=40-85
   - Steam above: y=15-40
   - Layout: vertical center alignment

3. SHAPE DECOMPOSITION:
   - Cup body: <path> (trapezoid for perspective)
   - Handle: <path> (smooth Bézier curve)
   - Steam: 3x <path> (gentle S-curves)
   - Saucer: <ellipse>

4. COORDINATE CALCULATION:
   - Cup path: M 35,85 L 30,50 L 70,50 L 65,85 Z
   - Handle: M 70,60 Q 80,60 80,70 Q 80,80 70,80
   - Steam paths calculated with gentle curves

5. STYLING:
   - Cup: fill="#3b3b3b" (dark gray)
   - Steam: stroke="#9ca3af", fill="none", stroke-width="2"
   - Style: minimalist, flat design

6. FINAL ASSEMBLY:
[Genera el SVG completo aquí]
```

### 4. Multi-Stage Prompting - Chat2SVG

**Concepto:** Expansión de prompts en 3 capas para enriquecer especificaciones.

#### Capa 1: Scene-Level Analysis

```
USER: "Create a logo for a bakery"

LLM EXPANSION:
"A bakery logo should include:
- Main element: bread/pastry/rolling pin/chef hat
- Complementary elements: wheat stalks, oven, warm colors
- Style: friendly, artisanal, appetizing
- Additional context: circular badge or vintage emblem"
```

#### Capa 2: Object-Level Decomposition

```
OBJECT: "Croissant logo"

DECOMPOSITION:
- Body: curved crescent shape
- Layers: 3-4 visible folded layers
- Texture: subtle lines indicating flaky pastry
- Shadow/depth: gradient or overlapping shapes
- Size ratio: fills 60% of canvas
```

#### Capa 3: Layout-Level Planning

```
LAYOUT PLAN:
- Canvas: 200x200 viewBox
- Croissant: positioned center-left (x=60-140, y=60-140)
- Text placement: right or bottom
- Color scheme:
  * Croissant: #d4a574 (golden brown)
  * Background: transparent or #f8f5f0 (cream)
  * Accent: #8b4513 (dark brown for outlines)
- Spacing: 10px padding around elements
```

#### Template Completo Multi-Stage:

```
# STAGE 1: Scene Analysis
Analiza holísticamente el prompt: "[USER_INPUT]"

Identifica objetos esenciales y complementarios.
Sugiere elementos que mejoren la completitud visual.

# STAGE 2: Object Breakdown
Para cada objeto identificado, descompón en componentes:
- Formas base
- Detalles visuales
- Relaciones entre componentes

# STAGE 3: Layout Planning
Desarrolla un plan de layout comprehensivo:
- Dimensiones del canvas (viewBox)
- Posicionamiento de elementos (coordenadas)
- Tamaños relativos
- Paleta de colores
- Relaciones espaciales

# STAGE 4: SVG Generation
Genera el código SVG usando SOLO primitivas básicas:
- <rect>, <circle>, <ellipse>
- <line>, <polyline>, <polygon>
- <path> (solo paths CORTOS para detalles)

# STAGE 5: Visual Verification
[Si es posible, renderizar y verificar]
Revisa inconsistencias:
- Proporciones desalineadas
- Elementos mal escalados
- Orden incorrecto de capas (z-index)
```

---

## Técnicas Específicas para SVG

### 1. Generación de SVG Válido Consistentemente

#### Constrained Generation

**Concepto:** Restringir la selección de tokens para garantizar output válido.

**Implementación:**
- Manipular logits en la capa de salida
- Permitir solo tokens que cumplan reglas XML/SVG
- 100% compliance con schema JSON/XML

**Ejemplo con schema constraint:**

```
Genera SVG que DEBE cumplir estas reglas:
1. Root element: <svg> con xmlns="http://www.w3.org/2000/svg"
2. viewBox obligatorio: formato "x y width height"
3. Solo tags permitidos: svg, g, rect, circle, ellipse, line, polyline, polygon, path, text, defs, linearGradient, radialGradient, stop
4. Todos los tags deben cerrarse correctamente
5. Atributos numéricos: solo números válidos (no texto)
6. Colores: hex (#RRGGBB) o nombres CSS válidos

PROHIBIDO:
- Tags HTML dentro de SVG
- Atributos inventados
- Valores de atributos inválidos
```

#### Semantic Tokens (LLM4SVG approach)

**Problema:** LLMs tratan SVG como texto plano, causando:
- Tokenización ineficiente de coordenadas
- Ambigüedad en atributos
- Pérdida de estructura semántica

**Solución:** 55 tokens especializados

```
TAG TOKENS (15):
<svg>, <g>, <rect>, <circle>, <ellipse>, <line>, <polyline>, <polygon>,
<path>, <text>, <defs>, <linearGradient>, <radialGradient>, <stop>, <use>

ATTRIBUTE TOKENS (30):
viewBox, width, height, x, y, cx, cy, r, rx, ry, x1, y1, x2, y2,
points, d, fill, stroke, stroke-width, opacity, transform, id, class,
offset, stop-color, gradientUnits, href, font-family, font-size, text-anchor

PATH COMMAND TOKENS (10):
M (moveto), L (lineto), H (horizontal), V (vertical), C (cubic Bézier),
S (smooth cubic), Q (quadratic Bézier), T (smooth quadratic), A (arc), Z (close)
```

**Benefit:** Embedding initialization con averaging semántico mejora representación.

### 2. Manejo de viewBox y Proporciones

#### Reglas Fundamentales:

```
viewBox="min-x min-y width height"

STANDARD SIZES:
- Iconos: viewBox="0 0 24 24" (Material Design)
- Iconos: viewBox="0 0 100 100" (facilidad cálculo)
- Logos: viewBox="0 0 200 100" (landscape)
- Logos: viewBox="0 0 100 200" (portrait)
- Square: viewBox="0 0 100 100"
```

#### Template para Proporciones Correctas:

```
Genera un logo con estas especificaciones:

CANVAS:
- viewBox="0 0 100 100"
- Coordenadas normalizadas: 0-100 en ambos ejes
- Permite escalado sin distorsión

PROPORTIONS:
- Elemento principal: 60-80% del canvas (20-80 en coordenadas)
- Padding: 10% mínimo en cada borde
- Aspect ratio: 1:1 (cuadrado), 2:1 (horizontal), 1:2 (vertical)

POSITIONING:
- Centro del canvas: cx="50" cy="50"
- Alineación: usar coordenadas relativas al centro
- Simetría: si requiere, calcular offsets desde el centro

EJEMPLO - Círculo centrado con padding correcto:
<circle cx="50" cy="50" r="40" /> <!-- r=40 deja 10px padding -->
```

#### PreserveAspectRatio:

```
<svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
  <!-- Contenido centrado, escalado proporcional, sin crop -->
</svg>

OPCIONES:
- xMidYMid meet: centrado, fit completo (default recomendado)
- xMinYMin slice: esquina superior-izq, crop si necesario
- none: estiramiento sin preservar ratio (evitar)
```

### 3. Optimización de Paths

#### Problema con LLMs:

Los LLMs generan paths subóptimos:
- Demasiados puntos de control
- Comandos redundantes
- Coordenadas con decimales innecesarios

#### Estrategia 1: Limitar Complejidad

```
Al generar paths:

1. USA FORMAS BÁSICAS cuando sea posible:
   - Círculo → <circle>, NO <path d="M...A...">
   - Rectángulo → <rect>, NO <path d="M...L...L...L...Z">
   - Línea → <line>, NO <path d="M...L...">

2. PATHS SOLO PARA:
   - Formas orgánicas/irregulares
   - Curvas suaves (Bézier)
   - Combinaciones de líneas/curvas complejas

3. SIMPLIFICA COMMANDS:
   - Preferir: M, L, C, Z (comandos absolutos)
   - Evitar: comandos relativos (m, l, c) si no es necesario
   - Combinar: M x,y L x,y L x,y vs múltiples paths
```

#### Estrategia 2: SVGO-inspired Rules

```
OPTIMIZACIÓN DE PATHS:

1. MERGE CONSECUTIVE COMMANDS:
   ❌ <path d="M 10,10 L 20,20 M 20,20 L 30,30"/>
   ✅ <path d="M 10,10 L 20,20 L 30,30"/>

2. REMOVE REDUNDANT COMMANDS:
   ❌ <path d="M 10,10 L 10,10 L 20,20"/>
   ✅ <path d="M 10,10 L 20,20"/>

3. ROUND COORDINATES (1 decimal max):
   ❌ <path d="M 10.4567,20.8934 L 30.1234,40.5678"/>
   ✅ <path d="M 10.5,20.9 L 30.1,40.6"/>

4. CONVERT TO RELATIVE cuando sea más corto:
   ❌ <path d="M 100,100 L 101,100 L 101,101"/>
   ✅ <path d="M 100,100 l 1,0 l 0,1"/>

5. USE SHORTHAND:
   ❌ <path d="M 10,10 L 10,20"/>  (vertical line)
   ✅ <path d="M 10,10 V 20"/>
```

#### Ejemplo Práctico - Path Optimization:

```
ANTES (LLM típico):
<path d="M 10.456,10.789 L 10.456,50.234 L 50.123,50.234 L 50.123,10.789 Z"
      fill="#ff0000" stroke="#000000" stroke-width="2.000"/>

DESPUÉS (optimizado):
<rect x="10.5" y="10.8" width="39.6" height="39.4"
      fill="#f00" stroke="#000" stroke-width="2"/>

SAVINGS:
- Caracteres: 119 → 84 (29% reducción)
- Claridad: forma geométrica explícita
- Parsing: más eficiente
```

### 4. Naming Conventions para Elementos

#### Estructura Semántica con IDs:

```
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg-gradient">
      <stop offset="0%" stop-color="#667eea"/>
      <stop offset="100%" stop-color="#764ba2"/>
    </linearGradient>
  </defs>

  <g id="logo-container">
    <g id="logo-background">
      <rect id="bg-rect" width="100" height="100" fill="url(#bg-gradient)"/>
    </g>

    <g id="logo-icon">
      <circle id="icon-circle1" cx="50" cy="40" r="15" fill="#fff"/>
      <circle id="icon-circle2" cx="50" cy="60" r="10" fill="#fff" opacity="0.8"/>
    </g>

    <g id="logo-text">
      <text id="text-main" x="50" y="90" text-anchor="middle"
            font-family="Arial" font-size="12" fill="#fff">
        LOGO
      </text>
    </g>
  </g>
</svg>
```

#### Convenciones de Nomenclatura:

```
REGLAS:
1. IDs: kebab-case (logo-icon, bg-gradient)
2. Prefijos por tipo:
   - bg-*: backgrounds
   - icon-*: íconos y elementos gráficos
   - text-*: elementos de texto
   - grad-*: gradientes
   - mask-*: máscaras
   - clip-*: clipping paths

3. Numeración: sufijos sin separador
   - icon-circle1, icon-circle2 (NO icon-circle-1)

4. Descriptivos y semánticos:
   - ✅ <g id="company-logo-icon">
   - ❌ <g id="group1">

5. Sin caracteres especiales:
   - Solo: a-z, 0-9, guiones (-)
   - NO: espacios, underscore, símbolos
```

#### Accesibilidad (ARIA):

```
<!-- Logo informativo -->
<svg role="img" aria-labelledby="logo-title logo-desc" viewBox="0 0 100 100">
  <title id="logo-title">Acme Corp Logo</title>
  <desc id="logo-desc">
    A circular blue logo with a white lightning bolt in the center
  </desc>
  <!-- contenido SVG -->
</svg>

<!-- Logo decorativo -->
<svg aria-hidden="true" viewBox="0 0 100 100">
  <!-- contenido SVG -->
</svg>

<!-- Elemento interactivo -->
<g id="clickable-icon" tabindex="0" role="button" aria-label="Download file">
  <rect width="50" height="50" fill="#007bff"/>
  <path d="M 25,20 L 25,35 L 15,35 L 30,50 L 45,35 L 35,35 L 35,20 Z" fill="#fff"/>
</g>
```

---

## Evaluación y Feedback Loops

### 1. Validación de SVG Generado

#### Niveles de Validación:

**Nivel 1: Sintaxis XML**
```python
import xml.etree.ElementTree as ET

def validate_xml_syntax(svg_string):
    try:
        ET.fromstring(svg_string)
        return True, "Valid XML"
    except ET.ParseError as e:
        return False, f"XML Error: {e}"
```

**Nivel 2: Conformidad SVG**
```python
def validate_svg_structure(svg_string):
    try:
        root = ET.fromstring(svg_string)

        # Check root tag
        if not root.tag.endswith('svg'):
            return False, "Root element must be <svg>"

        # Check required attributes
        if 'viewBox' not in root.attrib:
            return False, "Missing viewBox attribute"

        # Validate viewBox format
        viewbox = root.attrib['viewBox'].split()
        if len(viewbox) != 4:
            return False, "Invalid viewBox format"

        return True, "Valid SVG structure"
    except Exception as e:
        return False, str(e)
```

**Nivel 3: Validación Visual (Rendering)**
```python
from cairosvg import svg2png
from PIL import Image
import io

def validate_rendering(svg_string):
    try:
        # Render to PNG
        png_bytes = svg2png(bytestring=svg_string.encode('utf-8'))

        # Open with PIL
        img = Image.open(io.BytesIO(png_bytes))

        # Check if image is not blank
        extrema = img.convert('L').getextrema()
        if extrema == (255, 255) or extrema == (0, 0):
            return False, "Rendered image is blank"

        return True, f"Valid rendering: {img.size}"
    except Exception as e:
        return False, f"Rendering error: {e}"
```

### 2. Sistemas de Scoring Automático

#### SVGEditBench Metrics:

**Mean Squared Error (MSE):**
```python
import numpy as np
from cairosvg import svg2png
from PIL import Image

def calculate_mse(svg_generated, svg_reference, size=(72, 72)):
    # Render both SVGs to PNG
    img1 = svg_to_image(svg_generated, size)
    img2 = svg_to_image(svg_reference, size)

    # Convert to numpy arrays (normalize to 0-1)
    arr1 = np.array(img1).astype(float) / 255.0
    arr2 = np.array(img2).astype(float) / 255.0

    # Calculate MSE across all channels
    mse = np.mean((arr1 - arr2) ** 2)

    return mse

def svg_to_image(svg_string, size):
    png_bytes = svg2png(
        bytestring=svg_string.encode('utf-8'),
        output_width=size[0],
        output_height=size[1],
        background_color='white'
    )
    return Image.open(io.BytesIO(png_bytes))
```

**Compression Ratio:**
```python
def calculate_compression_ratio(original_svg, optimized_svg):
    original_length = len(original_svg)
    optimized_length = len(optimized_svg)

    ratio = optimized_length / original_length

    return {
        'ratio': ratio,
        'original_bytes': original_length,
        'optimized_bytes': optimized_length,
        'savings_percent': (1 - ratio) * 100
    }
```

#### SVGenius Multi-Dimensional Metrics:

```python
class SVGEvaluator:
    def evaluate(self, svg_generated, svg_reference=None, text_prompt=None):
        scores = {}

        # 1. Code similarity (si hay referencia)
        if svg_reference:
            scores['bleu'] = self.calculate_bleu(svg_generated, svg_reference)

        # 2. Visual quality metrics
        scores['fid'] = self.calculate_fid(svg_generated, svg_reference)
        scores['ssim'] = self.calculate_ssim(svg_generated, svg_reference)
        scores['lpips'] = self.calculate_lpips(svg_generated, svg_reference)

        # 3. Semantic alignment (si hay prompt)
        if text_prompt:
            scores['clip_score'] = self.calculate_clip_score(svg_generated, text_prompt)
            scores['hps_v2'] = self.calculate_hps(svg_generated, text_prompt)

        # 4. Validity
        scores['valid'] = self.check_validity(svg_generated)

        # 5. Token efficiency
        scores['token_count'] = len(svg_generated.split())

        return scores
```

#### TextGrad Weighted Evaluation:

```python
def evaluate_svg_quality(svg_code, description):
    """
    Weighted evaluation system:
    - Technical Correctness: 40%
    - Style Requirements: 30%
    - Adherence to Description: 30%
    """

    technical_score = evaluate_technical(svg_code)  # /40
    style_score = evaluate_style(svg_code)          # /30
    description_score = evaluate_description(svg_code, description)  # /30

    total = technical_score + style_score + description_score

    return f"Final Score: {total} / 100"

def evaluate_technical(svg_code):
    """Technical Correctness (40 points)"""
    score = 0

    # viewBox usage (10 pts)
    if 'viewBox=' in svg_code:
        score += 10

    # Valid path commands (10 pts)
    valid_commands = all(cmd in 'MLHVCSQTAZmlhvcsqtaz'
                         for cmd in extract_path_commands(svg_code))
    if valid_commands:
        score += 10

    # Efficient elements (10 pts)
    # Prefer basic shapes over complex paths
    basic_shapes = svg_code.count('<circle') + svg_code.count('<rect') + \
                   svg_code.count('<ellipse') + svg_code.count('<polygon')
    complex_paths = svg_code.count('<path')
    if basic_shapes >= complex_paths:
        score += 10

    # Clean structure (10 pts)
    if '<g id=' in svg_code and '</g>' in svg_code:
        score += 10

    return score

def evaluate_style(svg_code):
    """Style Requirements (30 points)"""
    score = 0

    # Color constraints (10 pts)
    # Check for specific color scheme

    # Transparent background (10 pts)
    if 'fill="none"' in svg_code or 'transparent' in svg_code.lower():
        score += 10

    # Stroke consistency (10 pts)
    stroke_widths = extract_stroke_widths(svg_code)
    if len(set(stroke_widths)) <= 3:  # Max 3 different widths
        score += 10

    return score
```

### 3. Re-Prompting Basado en Errores

#### Verification and Correction Prompting (VCP):

```
SYSTEM: Eres un generador y verificador de SVG experto.

PASO 1 - GENERACIÓN INICIAL:
[SVG generado por el modelo]

PASO 2 - VERIFICACIÓN:
Analiza el SVG generado y detecta errores:
- [ ] Sintaxis XML válida
- [ ] viewBox presente y correcto
- [ ] Todos los tags cerrados
- [ ] Coordenadas dentro de viewBox
- [ ] Colores en formato válido
- [ ] Proporciones correctas

PASO 3 - CORRECCIÓN:
Si encontraste errores, genera una versión corregida.
Lista los cambios realizados:

ERRORES ENCONTRADOS:
1. [descripción del error]
2. [descripción del error]

CORRECCIONES APLICADAS:
1. [cambio realizado]
2. [cambio realizado]

SVG CORREGIDO:
[nueva versión del SVG]
```

#### Iterative Refinement Loop:

```python
def iterative_svg_generation(prompt, max_iterations=3):
    svg_current = None
    errors = []

    for i in range(max_iterations):
        # Generate or refine
        if i == 0:
            # Initial generation
            full_prompt = f"Generate SVG: {prompt}"
        else:
            # Refinement based on errors
            error_feedback = "\n".join([f"- {e}" for e in errors])
            full_prompt = f"""
            Previous SVG had these issues:
            {error_feedback}

            Fix these issues and regenerate the SVG for: {prompt}
            """

        svg_current = llm.generate(full_prompt)

        # Validate
        is_valid, validation_errors = validate_svg_comprehensive(svg_current)

        if is_valid:
            return svg_current, i + 1  # Success

        errors = validation_errors

    return svg_current, max_iterations  # Return best attempt
```

#### Chat2SVG Visual Rectification:

```
STAGE 1: Generación inicial
[SVG code]

STAGE 2: Renderizado y verificación visual
[Renderiza el SVG a imagen]

STAGE 3: Detección de inconsistencias
Analiza la imagen renderizada y detecta:
- Elementos mal alineados
- Proporciones incorrectas
- Elementos faltantes vs. descripción
- Orden incorrecto de capas (z-index)
- Colores desalineados

STAGE 4: Re-prompting con feedback visual
El SVG renderizado muestra estos problemas:
1. [problema visual detectado]
2. [problema visual detectado]

Corrige el código SVG para resolver estos problemas visuales.

[Iteración 2 - máximo 2 iteraciones según paper]
```

### 4. Human-in-the-Loop Workflows

#### SVGauge - Human-Aligned Evaluation:

```
EVALUACIÓN HUMANA - Escala 1-5:

1. PROMPT ALIGNMENT (Alineación con descripción):
   ¿El SVG refleja con precisión lo solicitado en el prompt?
   1 - Totalmente diferente
   3 - Parcialmente correcto
   5 - Perfectamente alineado

2. VISUAL QUALITY (Calidad visual):
   ¿El SVG tiene buena calidad estética?
   1 - Muy pobre
   3 - Aceptable
   5 - Excelente calidad profesional

3. TECHNICAL CORRECTNESS (Corrección técnica):
   ¿El código SVG es válido y eficiente?
   1 - Errores críticos
   3 - Funcional con issues menores
   5 - Código óptimo

4. USABILITY (Usabilidad):
   ¿El SVG es fácil de editar y mantener?
   1 - Código confuso/inmantenible
   3 - Moderadamente editable
   5 - Muy bien estructurado
```

#### Interactive Feedback System:

```python
class HumanInTheLoopSVGGenerator:
    def generate_with_feedback(self, initial_prompt):
        # Initial generation
        svg = self.llm.generate(initial_prompt)
        iteration = 1

        while iteration <= 5:  # Max 5 iterations
            # Display to user
            print(f"\n=== Iteration {iteration} ===")
            self.display_svg(svg)

            # Get human feedback
            print("\nFeedback options:")
            print("1. Approve (done)")
            print("2. Request changes")
            print("3. Start over")

            choice = input("Your choice: ")

            if choice == "1":
                return svg, iteration

            elif choice == "2":
                feedback = input("Describe the changes needed: ")

                # Re-prompt with feedback
                refinement_prompt = f"""
                Original prompt: {initial_prompt}
                Current SVG: {svg}

                User feedback: {feedback}

                Modify the SVG to address the user's feedback.
                """

                svg = self.llm.generate(refinement_prompt)
                iteration += 1

            elif choice == "3":
                svg = self.llm.generate(initial_prompt)
                iteration = 1

        return svg, iteration
```

#### Counterfactual Editing:

```
SISTEMA: Has generado un SVG pero el usuario quiere cambios.

SVG ORIGINAL:
[código SVG actual]

FEEDBACK DEL USUARIO:
"El logo está muy a la izquierda, centralo. Y hazlo más grande."

ANÁLISIS CONTRAFACTUAL:
¿Qué pasaría si...?
1. Aumento cx de 30 a 50 (centro horizontal)
2. Aumento cy de 30 a 50 (centro vertical)
3. Aumento el radio de 20 a 35 (más grande)

RESULTADO ESPERADO:
- Elemento centrado en (50, 50)
- Tamaño aumentado ~75%
- Proporciones mantenidas

GENERA SVG MODIFICADO:
[aplica los cambios calculados]
```

---

## Casos de Éxito y Proyectos Open Source

### 1. LLM4SVG (CVPR 2025)

**GitHub:** https://github.com/ximinng/LLM4SVG

**Características:**
- 55 semantic tokens especializados
- Dataset: 250k SVGs + 580k instrucciones
- Soporta modelos: Llama 3.2, Qwen2.5-VL, Gemma 3, DeepSeek, Falcon, Phi-2, GPT2-XL

**Resultados:**
- Mejora significativa en comprensión de SVG complejo
- Generación consistente con context length 2048+
- 2x más rápido con vLLM backend

**Caso de uso:**
```bash
# Training
python train.py --model qwen-vl --dataset SVGX-SFT-1M --max_seq_length 2048

# Inference
python infer.py --model qwen-vl --prompt "minimalist tech logo"
```

### 2. OmniSVG (NeurIPS 2025)

**GitHub:** https://github.com/OmniSVG/OmniSVG

**Características:**
- Dataset MMSVG-2M: 2 millones de SVGs anotados
- Basado en Qwen-VL (vision-language model)
- SVG tokenizer especializado
- Genera desde iconos simples hasta personajes anime complejos

**Benchmark Results:**
- Text-to-SVG: State-of-the-art en FID, CLIPScore
- Image-to-SVG: Mejor reconstructabilidad visual
- Multimodal: Soporta texto + imagen como input

### 3. StarVector

**GitHub:** https://github.com/joanrod/star-vector
**Hugging Face:** starvector/starvector-8b-im2svg

**Características:**
- Basado en StarCoder (coding LLM)
- Vision-language architecture
- Modelos: StarVector-8B, StarVector-1B

**Ventaja:**
Trata SVG generation como code generation task, aprovechando habilidades de coding del modelo base.

**Uso:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("starvector/starvector-8b-im2svg")
tokenizer = AutoTokenizer.from_pretrained("starvector/starvector-8b-im2svg")

# Image + text to SVG
inputs = tokenizer(image=img, text="simple logo", return_tensors="pt")
outputs = model.generate(**inputs)
svg_code = tokenizer.decode(outputs[0])
```

### 4. Chat2SVG (CVPR 2025)

**Sitio:** https://chat2svg.github.io/

**Innovación:**
- Hybrid LLM + Diffusion Model
- Multi-stage prompt expansion (scene/object/layout)
- Visual rectification con feedback loop

**Pipeline:**
1. LLM expande prompt en 3 capas
2. LLM genera SVG template (primitivas básicas)
3. Diffusion model refina detalles visuales
4. LLM verifica y corrige inconsistencias (2 iteraciones)

**Casos de uso exitosos:**
- Iconos web flat design
- Ilustraciones técnicas
- Logos corporativos simples

### 5. SVGEditBench (CVPR 2024)

**GitHub:** https://github.com/mti-lab/SVGEditBench

**Aporte:**
Benchmark standardizado para evaluar capacidades de edición de SVG en LLMs.

**6 Tasks:**
1. Change Color
2. Set Contour
3. Upside-Down
4. Transparency
5. Crop to Half
6. Compression

**Métricas:**
- MSE (Mean Squared Error) en rendering 72x72
- Compression ratio

**Dataset:** 100 Twemoji emojis = 600 tareas

### 6. Reason-SVG (2025)

**Innovación:** Drawing-with-Thought paradigm

**Training:**
- Stage 1: Supervised Fine-Tuning (SFT) con DwT sequences
- Stage 2: Reinforcement Learning con hybrid reward

**Hybrid Reward Function:**
- Structural validity (SVG parsing correcto)
- Semantic alignment (CLIP score con prompt)
- Visual coherence (perceptual quality)

**Resultados:**
"Aha-moments" donde el modelo demuestra razonamiento visual emergente.

### 7. TextGrad (Headstorm Case Study)

**URL:** https://headstorm.com/case-study/technical-insight/optimizing-local-llm-svg-code-generation-with-textgrad/

**Concepto:**
LLM-as-judge optimization para prompt refinement.

**Proceso:**
1. Prompt inicial → SVG generado
2. Evaluador LLM → feedback textual
3. Optimizer → modifica prompt
4. Iteración hasta convergencia

**Mejoras logradas:**
- +40% en technical correctness
- +25% en style adherence
- Prompts optimizados reutilizables

**Optimized System Prompt Example:**
```
You are an expert SVG generator. Follow these rules:

OUTPUT:
- ONLY valid SVG code, no additional text
- Start with <?xml version="1.0" encoding="UTF-8"?>

DIMENSIONS:
- viewBox="0 0 100 100" (always normalized)
- width and height can be omitted (scalable by default)

COLORS:
- Use hex format: #RRGGBB
- Limit palette to 3-5 colors max

STRUCTURE:
- Use <g> for logical grouping
- Add descriptive id attributes (kebab-case)
- Include comments for complex sections

SHAPES:
- Prefer basic shapes (<circle>, <rect>) over <path>
- Use <path> only for complex/organic shapes
- Optimize path commands (remove redundancy)

VALIDATION:
- All tags must close properly
- All attributes must have valid values
- Coordinates must be within viewBox bounds
```

---

## Prompt Templates Listos para Usar

### Template 1: Logo Simple - Zero-Shot

```
Generate a minimalist logo in SVG format with these specifications:

DESCRIPTION: [describe tu logo, ej: "a geometric mountain with a sun"]

TECHNICAL REQUIREMENTS:
- viewBox="0 0 100 100"
- Use only basic shapes: <circle>, <rect>, <polygon>, <path> (if necessary)
- Maximum 3 colors (hex format)
- Clean, professional design
- All elements properly grouped with descriptive IDs

STYLE:
- Flat design, no gradients
- Geometric and modern
- Balanced composition
- 10px minimum padding from edges

OUTPUT:
Return ONLY the SVG code, starting with <svg> tag.
```

### Template 2: Logo con Chain-of-Thought

```
I need a logo for [INDUSTRY/COMPANY]. Before generating the SVG, think through:

STEP 1 - CONCEPT:
What visual metaphors represent [INDUSTRY]?
What emotions should the logo convey?
List 3-5 key visual elements.

STEP 2 - DESIGN DECISIONS:
Choose color palette (2-3 colors)
Decide on primary shape (circle/square/hexagon/custom)
Determine level of detail (minimalist/moderate/detailed)

STEP 3 - LAYOUT PLANNING:
Canvas: viewBox="0 0 100 100"
Element placement: [sketch coordinate ranges]
Proportions: [ratios of elements]

STEP 4 - GENERATION:
Now generate the SVG code following your design plan.
Use semantic IDs for all groups and elements.
Ensure clean, optimized code.

OUTPUT FORMAT:
First show your thinking (steps 1-3), then provide the SVG code.
```

### Template 3: Few-Shot Icon Generation

```
You are an expert SVG icon designer. Here are examples of your work:

EXAMPLE 1 - Settings Icon:
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="3" fill="none" stroke="#000" stroke-width="2"/>
  <path d="M 12,2 L 12,6 M 12,18 L 12,22 M 2,12 L 6,12 M 18,12 L 22,12"
        stroke="#000" stroke-width="2"/>
</svg>

EXAMPLE 2 - Download Icon:
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="M 12,2 L 12,16 M 8,12 L 12,16 L 16,12"
        stroke="#000" stroke-width="2" fill="none"/>
  <rect x="4" y="18" width="16" height="2" fill="#000"/>
</svg>

EXAMPLE 3 - User Icon:
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="8" r="4" fill="#000"/>
  <path d="M 4,22 Q 4,16 12,16 Q 20,16 20,22" fill="#000"/>
</svg>

Now create an icon for: [YOUR ICON DESCRIPTION]

Follow the same style:
- viewBox="0 0 24 24" (Material Design standard)
- Stroke width: 2
- Simple, recognizable shapes
- Maximum 3 elements
- Clean, minimalist aesthetic
```

### Template 4: Drawing-with-Thought (Reason-SVG)

```
Generate a logo using the Drawing-with-Thought process:

PROMPT: [your logo description]

STAGE 1 - CONCEPT SKETCHING:
List key visual components relevant to the prompt.
Components: [list 3-7 elements]

STAGE 2 - CANVAS PLANNING:
viewBox: [decide dimensions]
Layout structure: [describe spatial arrangement]
Proportions: [element size ratios]

STAGE 3 - SHAPE DECOMPOSITION:
Break down each component into geometric primitives.
For each element:
- Shape type: [circle/rect/polygon/path]
- Approximate size and position
- Relationships to other elements

STAGE 4 - COORDINATE CALCULATION:
Calculate precise coordinates for each shape.
Show your math for positioning and sizing.

STAGE 5 - STYLING & COLORING:
Color palette: [list colors with hex codes]
Fill vs stroke decisions: [specify for each element]
Visual hierarchy: [which elements stand out]

STAGE 6 - FINAL ASSEMBLY:
Generate the complete SVG code with:
- Proper structure (<svg>, <g> groups)
- Semantic IDs for all elements
- Optimized paths
- Comments for complex sections

OUTPUT:
Show your work for stages 1-5, then provide the final SVG code.
```

### Template 5: Multi-Stage Expansion (Chat2SVG)

```
MULTI-STAGE SVG GENERATION:

USER INPUT: [brief description]

STAGE 1 - SCENE-LEVEL EXPANSION:
Analyze the input holistically.
Essential objects: [list]
Complementary elements to enhance completeness: [suggestions]
Style direction: [minimalist/detailed/abstract/realistic]

STAGE 2 - OBJECT-LEVEL BREAKDOWN:
For each object identified:

Object: [name]
Components:
  - [component 1]: [description]
  - [component 2]: [description]
  - [component 3]: [description]
Relationships: [how components connect]

STAGE 3 - LAYOUT PLANNING:
Canvas: viewBox="0 0 [width] [height]"
Element positioning:
  - [element 1]: x=[range], y=[range]
  - [element 2]: x=[range], y=[range]
Sizing: [relative sizes]
Color scheme: [palette]
Spacing: [padding, margins]

STAGE 4 - SVG GENERATION:
Constraint: Use ONLY basic geometric primitives:
- <rect>, <circle>, <ellipse>
- <line>, <polyline>, <polygon>
- <path> (only SHORT paths for details)

Generate the SVG code following the layout plan.

STAGE 5 - VERIFICATION:
Review the generated SVG for:
- Proportional accuracy
- Element alignment
- Correct layering (z-order)
- Color consistency

If issues found, regenerate with corrections.

OUTPUT: Final optimized SVG code.
```

### Template 6: Iterative Refinement con Feedback

```
INITIAL GENERATION:
Create an SVG logo for: [description]

[LLM generates SVG]

VALIDATION CHECKLIST:
Evaluate the generated SVG:

Technical Correctness (40 points):
[ ] viewBox present and correct format (10 pts)
[ ] Valid path commands (10 pts)
[ ] Efficient use of basic shapes (10 pts)
[ ] Clean structure with <g> groups and IDs (10 pts)
Score: ___/40

Style Requirements (30 points):
[ ] Color scheme adherence (10 pts)
[ ] Consistent stroke widths (10 pts)
[ ] Appropriate detail level (10 pts)
Score: ___/30

Description Adherence (30 points):
[ ] Matches prompt specifications (10 pts)
[ ] Accurate visual representation (10 pts)
[ ] Includes all required elements (10 pts)
Score: ___/30

TOTAL SCORE: ___/100

REFINEMENT (if score < 80):
Issues found:
1. [issue description]
2. [issue description]

Corrections needed:
1. [specific fix]
2. [specific fix]

REGENERATE:
Apply corrections and generate improved SVG.

[Iterate until score ≥ 80]
```

### Template 7: Constrained Generation (Schema Validation)

```
Generate SVG with STRICT constraints:

REQUIRED STRUCTURE:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="[x] [y] [w] [h]">
  <g id="[descriptive-name]">
    [content]
  </g>
</svg>

ALLOWED TAGS:
- Shapes: <rect>, <circle>, <ellipse>, <polygon>, <polyline>, <line>, <path>
- Containers: <g>, <defs>, <symbol>
- Gradients: <linearGradient>, <radialGradient>, <stop>
- Text: <text>, <tspan>

FORBIDDEN:
- HTML tags inside SVG
- Invalid attributes
- Unclosed tags
- Coordinate values outside viewBox
- Non-hex color values

ATTRIBUTE RULES:
- Colors: hex format (#RRGGBB or #RGB)
- Numbers: max 1 decimal place
- IDs: kebab-case, no special characters
- viewBox: exactly 4 space-separated numbers

VALIDATION:
Your SVG will be validated against XML schema.
Any violations will cause rejection.

GENERATE:
Logo description: [your description]

Return ONLY valid SVG code.
```

### Template 8: Accessibility-First

```
Generate an accessible SVG logo:

DESCRIPTION: [your logo description]

ACCESSIBILITY REQUIREMENTS:

1. SEMANTIC STRUCTURE:
   - Include role="img"
   - Add <title> as first child
   - Add <desc> for detailed description
   - Use aria-labelledby

2. NAMING CONVENTIONS:
   - Descriptive IDs (kebab-case)
   - Semantic grouping with <g>
   - Logical layer names

3. INTERACTIVE ELEMENTS (if any):
   - tabindex="0" for focusable elements
   - aria-label for controls
   - Keyboard navigation support

4. COLOR ACCESSIBILITY:
   - Sufficient contrast ratios
   - Don't rely solely on color to convey meaning
   - Include text alternatives

TEMPLATE:
<svg role="img" aria-labelledby="logo-title logo-desc" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
  <title id="logo-title">[Concise title]</title>
  <desc id="logo-desc">[Detailed description of visual elements]</desc>

  <g id="[component-name]">
    <!-- SVG content -->
  </g>
</svg>

GENERATE accessible SVG for: [description]
```

### Template 9: Optimización con TextGrad

```
SYSTEM PROMPT (Optimized):
You are an expert SVG code generator specializing in clean, efficient vector graphics.

RULES:

OUTPUT FORMAT:
- Return ONLY SVG code
- No markdown, no explanations
- Valid XML structure

TECHNICAL STANDARDS:
- viewBox: Always use normalized coordinates (0 0 100 100 recommended)
- Precision: Round coordinates to 1 decimal max
- Efficiency: Prefer basic shapes over complex paths
  * Circle → <circle>, NOT <path with arcs>
  * Rectangle → <rect>, NOT <path with lines>
  * Only use <path> for organic/irregular shapes

STRUCTURE:
- Group related elements: <g id="descriptive-name">
- Logical hierarchy: background → main elements → foreground
- Semantic IDs: logo-icon, bg-shape, text-label (kebab-case)
- Add <!-- comments --> for complex sections

OPTIMIZATION:
- Merge consecutive path commands
- Remove redundant attributes
- Use shorthand (V, H) for vertical/horizontal lines
- Simplify: combine shapes when possible

COLORS:
- Hex format: #RGB or #RRGGBB
- Named colors: only CSS standard names
- Gradients: define in <defs>, reference by id

VALIDATION BEFORE OUTPUT:
- All tags properly closed
- No coordinates outside viewBox
- No invalid attribute values
- Proper nesting structure

USER PROMPT:
[Your logo/icon description]

EVALUATION CRITERIA:
Technical Correctness: 40%
Style Requirements: 30%
Description Adherence: 30%
Target: 90+/100

GENERATE optimized SVG.
```

### Template 10: Logo Profesional Completo

```
Create a professional brand logo with complete specifications:

BRAND INFO:
Company: [name]
Industry: [industry]
Values: [list 3-5 values, e.g., innovation, trust, sustainability]
Target audience: [description]

DESIGN BRIEF:
Style: [minimalist/modern/classic/playful/corporate]
Complexity: [simple icon / icon+text / emblem / wordmark]
Color scheme: [specify or let AI suggest based on industry]
Preferred shapes: [geometric/organic/abstract/literal]

TECHNICAL SPECS:
- Primary use: [web/print/both]
- Sizes needed: [favicon 16x16, icon 48x48, logo 200x200, etc.]
- Monochrome version: [yes/no]
- Background: [transparent/white/colored]

CONSTRAINTS:
- Must work at small sizes (favicon-friendly)
- Should be recognizable in silhouette
- Maximum 4 colors
- Scalable without loss of detail

DELIVERABLE:
Generate SVG with:
1. Main logo (full color)
2. Proper viewBox for primary use case
3. Organized layers/groups
4. Accessible markup (title, desc, ARIA)
5. Comments explaining design decisions

DESIGN PROCESS:
1. Brainstorm visual concepts
2. Sketch coordinate layout
3. Choose color palette with rationale
4. Generate optimized SVG code
5. Verify scalability and clarity

OUTPUT:
Present your design thinking, then the final SVG code.
```

---

## Comparación de Técnicas

### Tabla Comparativa: Efectividad por Caso de Uso

| Técnica | Iconos Simples | Logos Complejos | Ilustraciones | Facilidad de Uso | Consistencia |
|---------|----------------|-----------------|---------------|------------------|--------------|
| **Zero-Shot** | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Chain-of-Thought** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Few-Shot (3 ejemplos)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Drawing-with-Thought** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-Stage (Chat2SVG)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Constrained Generation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Iterative Refinement** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

### Comparación: Modelos LLM

| Modelo | SVG Quality | Speed | Cost | Best For |
|--------|-------------|-------|------|----------|
| **GPT-4o** | ⭐⭐⭐⭐ | ⭐⭐⭐ | $$$ | Logos complejos, razonamiento |
| **Claude 3.7 Sonnet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ | Artifacts, visualización interactiva |
| **Gemini 2.0 Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | Balance velocidad/calidad |
| **DeepSeek-R1** | ⭐⭐⭐⭐ | ⭐⭐⭐ | $ | Razonamiento avanzado |
| **StarVector-8B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Image-to-SVG, open source |
| **LLM4SVG (fine-tuned)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free* | SVG especializado, fine-tuning |

*Free: requiere hosting propio

### Benchmarks de Rendimiento

Según **SVGenius Benchmark**:

**Understanding (comprensión SVG):**
- Claude 3.7 Sonnet: 87.3%
- GPT-4o: 85.1%
- Gemini 2.0 Flash: 83.7%

**Degradación con complejidad:**
- Simple (0-20 elementos): 92% accuracy promedio
- Medium (21-50 elementos): 78% accuracy
- Complex (51+ elementos): 54% accuracy

**Editing (edición de SVG):**
- Claude 3.7 Sonnet: 81.2%
- GPT-4o: 79.5%
- DeepSeek-R1: 77.8%

**Generation (generación text-to-SVG):**
- LLM4SVG (fine-tuned): 89.7% validity
- Claude 3.7 Sonnet: 76.4% validity
- GPT-4o: 74.1% validity

**Resiliency (resistencia a complejidad):**
Generation tasks > Editing > Understanding
(La generación es más resiliente a complejidad que comprensión)

### Comparación: Estrategias de Prompting

**Velocidad de iteración:**
1. Zero-Shot: 1 llamada (~5 seg)
2. Chain-of-Thought: 1 llamada (~8 seg, más tokens)
3. Few-Shot: 1 llamada (~10 seg, contexto grande)
4. Multi-Stage: 3-5 llamadas (~30-50 seg)
5. Iterative Refinement: 2-5 llamadas (~20-60 seg)

**Costo (tokens aproximados):**
1. Zero-Shot: ~200 tokens
2. Chain-of-Thought: ~500 tokens
3. Few-Shot: ~800-1200 tokens (ejemplos)
4. Multi-Stage: ~1500-2500 tokens total
5. Iterative Refinement: ~1000-3000 tokens total

**Calidad final:**
1. Zero-Shot: 60-70% válido
2. Chain-of-Thought: 75-85% válido
3. Few-Shot: 85-92% válido
4. Multi-Stage: 88-95% válido
5. Iterative Refinement: 92-98% válido

### Trade-offs Clave

**Para PRODUCCIÓN:**
- Use Few-Shot o Constrained Generation
- Priorize consistencia sobre creatividad
- Implemente validación automática

**Para EXPLORACIÓN/DISEÑO:**
- Use Multi-Stage o Drawing-with-Thought
- Priorize calidad sobre velocidad
- Incluya human-in-the-loop

**Para SCALE:**
- Fine-tune modelo (LLM4SVG approach)
- Use semantic tokens
- Batch processing con validación

---

## Recomendaciones para Nuestro Proyecto

### Fase 1: MVP - Logo Generator Básico

**Stack recomendado:**
- **Modelo:** Claude 3.7 Sonnet (via Anthropic API)
- **Técnica:** Few-Shot + Chain-of-Thought
- **Validación:** XML parsing + rendering test

**Implementación:**

```python
# svg_generator.py

SYSTEM_PROMPT = """
You are an expert SVG logo generator. You create clean, professional,
scalable vector graphics following best practices.

EXAMPLES:

[Include 3-5 high-quality logo examples]

RULES:
- viewBox="0 0 100 100" (normalized coordinates)
- Use basic shapes when possible
- Maximum 4 colors
- Semantic IDs (kebab-case)
- Clean structure with <g> groups

OUTPUT: Only SVG code, no explanations.
"""

def generate_logo(description, industry, style):
    cot_prompt = f"""
    Create a logo for: {description}
    Industry: {industry}
    Style: {style}

    First, think through:
    1. Key visual metaphors for {industry}
    2. Appropriate color palette
    3. Shape selection (geometric/organic)
    4. Layout structure

    Then generate the SVG code.
    """

    response = claude.complete(
        system=SYSTEM_PROMPT,
        prompt=cot_prompt,
        max_tokens=2000
    )

    svg_code = extract_svg(response)

    # Validate
    is_valid, errors = validate_svg(svg_code)

    if not is_valid:
        # One refinement attempt
        svg_code = refine_svg(svg_code, errors)

    return svg_code
```

**Por qué esta combinación:**
- Claude Sonnet 3.7: mejor rendimiento en benchmark SVGenius
- Few-Shot: +28% accuracy vs zero-shot
- CoT: mejora razonamiento sin complejidad de multi-stage
- 1 refinement: balance costo/calidad

### Fase 2: Fine-Tuning (Mediano Plazo)

**Objetivo:** Modelo especializado en logos

**Enfoque LLM4SVG:**

1. **Dataset creation:**
   - Curate 10k-50k high-quality logos (SVG + descripciones)
   - Fuentes: Noun Project, Flaticon (con licencias), custom designs
   - Anotaciones: GPT-4 para descripciones detalladas

2. **Semantic tokens:**
   - Implement 55 specialized tokens
   - Initialize with semantic averaging

3. **Two-stage training:**
   - Stage 1: Feature alignment (solo embeddings)
   - Stage 2: Full fine-tuning con LoRA

4. **Modelos candidatos:**
   - Qwen2.5-VL (mejor multimodal open source)
   - Llama 3.2 Vision (buena performance, menos restricciones)
   - Gemma 3 (eficiente, Google ecosystem)

**Timeline estimado:** 3-6 meses

### Fase 3: Advanced Features

#### 3.1 Multi-Modal Input (Image-to-SVG)

**Use case:** Usuario sube sketch/imagen → genera SVG vectorizado

**Stack:**
- StarVector-8B (specialized image-to-SVG)
- O fine-tune Qwen2.5-VL con nuestro dataset

**Implementación:**
```python
def image_to_svg(image_path, style_prompt="minimalist logo"):
    # Load StarVector o Qwen-VL
    model = load_model("starvector/starvector-8b-im2svg")

    # Process
    image = load_image(image_path)
    prompt = f"Convert to SVG: {style_prompt}"

    svg = model.generate(image=image, text=prompt)

    # Post-process: simplify, optimize
    svg = optimize_svg(svg)

    return svg
```

#### 3.2 Style Transfer

**Use case:** Aplicar estilo de un logo a otro

**Técnica:** OmniSVG approach con style conditioning

```python
def style_transfer(content_svg, style_svg):
    # Extract style features
    style_features = extract_svg_style(style_svg)
    # Colors, stroke-width, shapes proportion, etc.

    # Reconstruct content with style
    prompt = f"""
    Recreate this logo: {content_svg}

    Apply this style:
    - Colors: {style_features['colors']}
    - Stroke width: {style_features['stroke_width']}
    - Shape style: {style_features['shape_style']}
    - Complexity: {style_features['complexity']}

    Maintain original concept, adapt to new style.
    """

    return llm.generate(prompt)
```

#### 3.3 Interactive Editing

**Use case:** Chat-based refinement

**Workflow:**
1. Generate initial logo
2. User: "make it bigger and move to the left"
3. System: parse intent → modify SVG → re-render
4. Iterate

**Implementation:**
```python
class InteractiveSVGEditor:
    def __init__(self, initial_svg):
        self.svg = initial_svg
        self.history = [initial_svg]

    def edit(self, user_feedback):
        prompt = f"""
        Current SVG:
        {self.svg}

        User feedback: {user_feedback}

        Modify the SVG to address feedback.
        Preserve structure and IDs.
        Return ONLY the modified SVG code.
        """

        new_svg = llm.generate(prompt)

        self.svg = new_svg
        self.history.append(new_svg)

        return new_svg

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.svg = self.history[-1]
        return self.svg
```

### Pipeline de Producción Recomendado

```
USER INPUT
    ↓
[Prompt Engineering Layer]
├─ Industry-specific templates
├─ Style conditioning
└─ Few-shot examples selection
    ↓
[Generation Layer]
├─ Primary: Claude 3.7 Sonnet (CoT + Few-Shot)
├─ Fallback: Gemini 2.0 Flash
└─ Future: Fine-tuned LLM4SVG model
    ↓
[Validation Layer]
├─ XML syntax check
├─ SVG schema validation
├─ Rendering test (CairoSVG)
└─ Quality scoring
    ↓
[Refinement Layer] (if score < 80)
├─ Error detection
├─ Re-prompting with feedback
└─ Max 2 iterations
    ↓
[Post-Processing Layer]
├─ SVGO optimization
├─ Accessibility additions (title, desc, ARIA)
├─ Multiple format export (SVG, PNG, PDF)
└─ Metadata embedding
    ↓
OUTPUT
├─ SVG file (optimized)
├─ Preview images (multiple sizes)
├─ Usage guidelines
└─ License/attribution
```

### Métricas de Éxito

**KPIs Fase 1:**
- SVG validity rate: >95%
- User satisfaction (1-5 stars): >4.2
- First-try success (no editing needed): >70%
- Generation time: <10 seconds

**KPIs Fase 2 (Fine-tuned):**
- SVG validity rate: >98%
- First-try success: >85%
- Generation time: <5 seconds
- Cost reduction: -60% vs API calls

**KPIs Fase 3 (Advanced):**
- Multi-modal accuracy: >80%
- Style transfer coherence: >4.0/5 user rating
- Interactive editing success rate: >90%

### Configuración de Desarrollo

**requirements.txt adiciones:**
```
anthropic>=0.18.0
google-generativeai>=0.3.0
openai>=1.0.0
cairosvg>=2.7.0
lxml>=4.9.0
Pillow>=10.0.0
svgwrite>=1.4.3
```

**Estructura de prompts:**
```
svg-logo-ai/
├── prompts/
│   ├── system_prompts/
│   │   ├── claude_svg_expert.txt
│   │   ├── gemini_svg_generator.txt
│   │   └── gpt4_svg_designer.txt
│   ├── few_shot_examples/
│   │   ├── tech_logos.json
│   │   ├── minimalist_icons.json
│   │   └── corporate_brands.json
│   └── templates/
│       ├── chain_of_thought.txt
│       ├── drawing_with_thought.txt
│       └── multi_stage.txt
```

### Testing Strategy

```python
# tests/test_svg_generation.py

def test_svg_validity():
    """All generated SVGs must be valid XML"""
    for prompt in test_prompts:
        svg = generator.generate(prompt)
        assert validate_xml(svg), f"Invalid XML for: {prompt}"

def test_svg_rendering():
    """All SVGs must render without errors"""
    for prompt in test_prompts:
        svg = generator.generate(prompt)
        assert can_render(svg), f"Cannot render: {prompt}"

def test_prompt_adherence():
    """Generated SVGs should match prompt description"""
    for prompt, expected_elements in test_cases:
        svg = generator.generate(prompt)
        score = calculate_adherence(svg, expected_elements)
        assert score > 0.7, f"Low adherence for: {prompt}"

def test_consistency():
    """Same prompt should generate similar results"""
    prompt = "minimalist tech logo"
    svgs = [generator.generate(prompt) for _ in range(5)]
    similarity = calculate_similarity_matrix(svgs)
    assert similarity.mean() > 0.75

def test_generation_time():
    """Generation should complete within time limit"""
    import time
    start = time.time()
    svg = generator.generate("simple icon")
    duration = time.time() - start
    assert duration < 15.0, f"Too slow: {duration}s"
```

### Consideraciones de Costos

**API Pricing (estimado):**
- Claude Sonnet 3.7: ~$0.02/logo (1500 tokens avg)
- GPT-4o: ~$0.03/logo
- Gemini 2.0 Flash: ~$0.01/logo

**Para 10,000 logos/mes:**
- Claude: $200/mes
- Fine-tuned model (hosting): $50-100/mes
- **ROI break-even:** ~2,000 logos

**Recommendation:** Start con API, migrar a fine-tuned a los 3-6 meses.

### Roadmap Sugerido

**Q1 2025:**
- ✅ Implementar Few-Shot + CoT pipeline
- ✅ Validación automática
- ✅ Web UI básico
- ✅ Export multi-formato

**Q2 2025:**
- 🔄 Dataset curation (10k+ logos)
- 🔄 Fine-tuning experiments
- 🔄 A/B testing API vs fine-tuned
- 🔄 Style presets library

**Q3 2025:**
- 📋 Image-to-SVG pipeline
- 📋 Interactive editing
- 📋 Batch processing API
- 📋 Enterprise features

**Q4 2025:**
- 📋 Multi-language support
- 📋 Advanced style transfer
- 📋 Plugin ecosystem (Figma, Adobe)
- 📋 On-premise deployment option

---

## Conclusión

La generación de SVG con LLMs ha madurado significativamente en 2024-2025 gracias a:

1. **Modelos especializados** (LLM4SVG, OmniSVG, StarVector)
2. **Técnicas de prompting** avanzadas (DwT, Multi-Stage, CoT)
3. **Benchmarks rigurosos** (SVGenius, SVGEditBench)
4. **Datasets de calidad** (MMSVG-2M, SVGX-SFT-1M)

**Para nuestro proyecto de generación de logos:**

**Short-term (MVP):**
- Claude 3.7 Sonnet + Few-Shot + CoT
- Validación automática robusta
- 1 iteración de refinement

**Medium-term (Scale):**
- Fine-tuning con LLM4SVG approach
- Semantic tokens especializados
- Qwen2.5-VL o Llama 3.2 Vision

**Long-term (Advanced):**
- Image-to-SVG (StarVector)
- Interactive editing
- Style transfer (OmniSVG)

La clave del éxito está en:
1. **Prompt engineering riguroso** (templates bien diseñados)
2. **Validación multi-nivel** (syntax, structure, visual)
3. **Feedback loops** (iterative refinement, human-in-the-loop)
4. **Especialización del modelo** (fine-tuning cuando escale)

Con esta base técnica, podemos construir un generador de logos SVG competitivo que combine la creatividad de los LLMs con la precisión requerida para gráficos vectoriales profesionales.

---

## Referencias

### Papers
1. **LLM4SVG** - Xing et al. (CVPR 2025): "Empowering LLMs to Understand and Generate Complex Vector Graphics"
2. **OmniSVG** - Li et al. (NeurIPS 2025): "A Unified Scalable Vector Graphics Generation Model"
3. **Chat2SVG** - Wu et al. (CVPR 2025): "Vector Graphics Generation with LLMs and Image Diffusion Models"
4. **Reason-SVG** - Chen et al. (2025): "Hybrid Reward RL for Aha-Moments in Vector Graphics Generation"
5. **SVGenius** - Zhang et al. (2024): "Benchmarking LLMs in SVG Understanding, Editing and Generation"
6. **SVGEditBench** - Tanaka et al. (CVPR 2024): "A Benchmark Dataset for LLM's SVG Editing Capabilities"

### GitHub Repositories
- LLM4SVG: https://github.com/ximinng/LLM4SVG
- OmniSVG: https://github.com/OmniSVG/OmniSVG
- StarVector: https://github.com/joanrod/star-vector
- SVGEditBench: https://github.com/mti-lab/SVGEditBench

### Hugging Face Models
- StarVector-8B: starvector/starvector-8b-im2svg
- SVGX Datasets: xingxm/SVGX-SFT-1M, xingxm/SVGX-Core-250k

### Blogs & Case Studies
- TextGrad Optimization: https://headstorm.com/case-study/technical-insight/optimizing-local-llm-svg-code-generation-with-textgrad/
- Chat2SVG Project: https://chat2svg.github.io/
- SVG with ChatGPT: https://praeclarum.org/2023/04/03/chatsvg.html

### Documentation
- MDN SVG Reference: https://developer.mozilla.org/en-US/docs/Web/SVG
- W3C SVG Specification: https://www.w3.org/TR/SVG2/
- SVG Accessibility: https://www.w3.org/WAI/WCAG21/Techniques/general/G140

---

**Última actualización:** 2025-01-25
**Autor:** Investigación compilada para proyecto svg-logo-ai
**Licencia:** MIT
