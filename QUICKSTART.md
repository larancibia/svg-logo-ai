# Quick Start Guide

Guía rápida para empezar a trabajar con el proyecto SVG Logo AI.

## ✅ Setup Completo

El proyecto ya está configurado con:
- ✓ Estructura de carpetas
- ✓ ChromaDB instalado
- ✓ Base de conocimiento poblada (7 papers, 5 modelos, 6 técnicas)
- ✓ Scripts de ejemplo listos

## 🚀 Usar la Base de Conocimiento

### 1. Activar entorno virtual

```bash
cd ~/svg-logo-ai
source venv/bin/activate
```

### 2. Ejecutar búsquedas

```bash
cd src
python example_usage.py
```

### Demo interactivo:

```bash
python example_usage.py --interactive
```

### Usar desde Python:

```python
from knowledge_base import SVGKnowledgeBase

kb = SVGKnowledgeBase()

# Buscar papers
papers = kb.search_papers("logo generation transformers")

# Buscar modelos disponibles
models = kb.search_models("commercial production")

# Búsqueda completa
results = kb.search_all("geometric minimalist logos")

# Estadísticas
print(kb.get_stats())
# Output: {'papers': 7, 'models': 5, 'techniques': 6}
```

## 🎨 Generar Logos con Gemini

### 1. Configurar GCP

```bash
# Exportar credenciales
export GCP_PROJECT_ID=tu-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# O usar .env
cp .env.example .env
# Editar .env con tus credenciales
```

### 2. Instalar dependencias de GCP

```bash
source venv/bin/activate
pip install google-cloud-aiplatform
```

### 3. Generar tu primer logo

```bash
cd src
python gemini_svg_generator.py
```

O desde Python:

```python
from gemini_svg_generator import GeminiSVGGenerator, LogoRequest

generator = GeminiSVGGenerator(project_id="tu-project-id")

request = LogoRequest(
    company_name="MiStartup",
    industry="Technology",
    style="minimalist",
    colors=["#2563eb", "#1e40af"],
    keywords=["innovation", "speed", "connection"]
)

result = generator.generate_logo(request)

if result['has_valid_svg']:
    generator.save_svg(result['svg_code'], "mi_logo.svg")
    print("Logo guardado en: output/mi_logo.svg")
```

## 📊 Explorar con Jupyter

```bash
source venv/bin/activate
pip install jupyter ipywidgets pandas
jupyter notebook notebooks/01_explore_knowledge_base.ipynb
```

## 🔍 Queries de Ejemplo

### Buscar modelos listos para producción:
```python
kb.search_models("commercial available production ready GCP")
```

### Buscar técnicas simples:
```python
kb.search_techniques("simple geometric easy beginner")
```

### Buscar papers sobre datasets:
```python
kb.search_papers("large dataset millions training data")
```

### Buscar sobre reasoning:
```python
kb.search_all("chain of thought reasoning SVG generation")
```

## 📁 Estructura del Proyecto

```
svg-logo-ai/
├── data/
│   └── chroma_db/          # Base de conocimiento (18 documentos)
├── docs/
│   └── RESEARCH_FINDINGS.md # Análisis completo de viabilidad
├── src/
│   ├── knowledge_base.py        # Sistema ChromaDB
│   ├── populate_knowledge.py    # Población de datos
│   ├── example_usage.py         # Ejemplos de búsqueda
│   └── gemini_svg_generator.py  # Generador con Gemini
├── notebooks/
│   └── 01_explore_knowledge_base.ipynb
├── output/                 # SVGs generados (se crea al usar)
└── venv/                   # Entorno virtual Python
```

## 💡 Próximos Pasos

### Opción 1: Experimentar con la base de conocimiento
```bash
python src/example_usage.py --interactive
```
Haz preguntas como:
- "reinforcement learning logos"
- "multimodal generation"
- "geometric primitive composition"

### Opción 2: Generar logos con Gemini
1. Configura credenciales GCP
2. Ejecuta `python src/gemini_svg_generator.py`
3. Revisa los SVGs en `output/`

### Opción 3: Desarrollar MVP
Ver roadmap en `docs/RESEARCH_FINDINGS.md`

## 🆘 Troubleshooting

### Error: "No module named 'chromadb'"
```bash
source venv/bin/activate
pip install chromadb
```

### Error: "GCP_PROJECT_ID not set"
```bash
export GCP_PROJECT_ID=tu-project-id
```

### Error: "Could not authenticate"
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### ChromaDB vacía
```bash
cd src
python populate_knowledge.py
```

## 📚 Recursos Clave

- **Research findings:** `docs/RESEARCH_FINDINGS.md`
- **Código base de conocimiento:** `src/knowledge_base.py`
- **Generador Gemini:** `src/gemini_svg_generator.py`
- **README principal:** `README.md`

## 🎯 Estado Actual

- ✅ Base de conocimiento funcionando (18 documentos indexados)
- ✅ Scripts de búsqueda listos
- ✅ Generador con Gemini implementado
- ⏳ Pendiente: Configurar credenciales GCP
- ⏳ Pendiente: Generar primeros logos
- ⏳ Pendiente: Evaluar calidad

**¡Listo para empezar a generar logos!** 🚀
