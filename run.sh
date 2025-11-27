#!/bin/bash

# Script helper para SVG Logo AI Generator
# Uso: ./run.sh [comando]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  SVG Logo AI Generator${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
}

function print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

function print_error() {
    echo -e "${RED}✗${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

function check_venv() {
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado"
        echo "Creando entorno virtual..."
        python3 -m venv venv
        print_success "Entorno virtual creado"
    fi
}

function activate_venv() {
    check_venv
    source venv/bin/activate
}

function install_deps() {
    print_header
    echo "📦 Instalando dependencias..."
    activate_venv
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    print_success "Dependencias instaladas"
}

function populate_kb() {
    print_header
    echo "📚 Poblando base de conocimiento..."
    activate_venv
    cd src
    python populate_knowledge.py
    cd ..
    print_success "Base de conocimiento poblada"
}

function search_kb() {
    print_header
    activate_venv
    cd src
    python example_usage.py
    cd ..
}

function search_interactive() {
    print_header
    activate_venv
    cd src
    python example_usage.py --interactive
    cd ..
}

function generate_logo() {
    print_header

    if [ -z "$GCP_PROJECT_ID" ]; then
        print_error "GCP_PROJECT_ID no configurado"
        echo "Exporta tu project ID:"
        echo "  export GCP_PROJECT_ID=tu-project-id"
        exit 1
    fi

    if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        print_warning "GOOGLE_APPLICATION_CREDENTIALS no configurado"
        echo "Exporta la ruta a tus credenciales:"
        echo "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json"
    fi

    echo "🎨 Generando logos..."
    activate_venv

    # Verificar si google-cloud-aiplatform está instalado
    if ! python -c "import vertexai" 2>/dev/null; then
        print_warning "google-cloud-aiplatform no instalado"
        echo "Instalando..."
        pip install -q google-cloud-aiplatform
    fi

    cd src
    python gemini_svg_generator.py
    cd ..
}

function stats() {
    print_header
    echo "📊 Estadísticas de la base de conocimiento:"
    activate_venv
    python -c "
import sys
sys.path.insert(0, 'src')
from knowledge_base import SVGKnowledgeBase
kb = SVGKnowledgeBase(persist_directory='data/chroma_db')
stats = kb.get_stats()
print(f\"📚 Papers:    {stats['papers']}\")
print(f\"🤖 Modelos:   {stats['models']}\")
print(f\"🛠️  Técnicas:  {stats['techniques']}\")
print(f\"━━━━━━━━━━━━━━━━━━━━━━\")
print(f\"Total:       {sum(stats.values())} documentos\")
"
}

function jupyter() {
    print_header
    echo "📓 Iniciando Jupyter..."
    activate_venv

    if ! python -c "import jupyter" 2>/dev/null; then
        print_warning "Jupyter no instalado"
        echo "Instalando..."
        pip install -q jupyter ipywidgets pandas
    fi

    jupyter notebook notebooks/
}

function gallery() {
    print_header
    echo "🎨 Generando galería de logos..."
    activate_venv
    cd src
    python gallery_generator.py
    cd ..

    # Abrir en navegador
    if command -v xdg-open > /dev/null; then
        xdg-open output/gallery.html
    elif command -v open > /dev/null; then
        open output/gallery.html
    else
        echo "Abre manualmente: output/gallery.html"
    fi
}

function logo_stats() {
    print_header
    echo "📊 Estadísticas de logos generados..."
    activate_venv
    python -c "
import sys
sys.path.insert(0, 'src')
from logo_metadata import LogoMetadata
metadata = LogoMetadata()
stats = metadata.get_stats()

print(f'Total logos: {stats[\"total\"]}')
print(f'Score promedio: {stats[\"avg_score\"]:.1f}/100')
print(f'Mejor score: {stats[\"max_score\"]}')
print(f'Favoritos: {stats[\"favorites\"]}')
print()
print('Por industria:')
for ind, data in stats['by_industry'].items():
    print(f'  {ind}: {data[\"count\"]} logos (avg: {data[\"avg_score\"]:.1f})')
print()
print('Por versión:')
for ver, data in stats['by_version'].items():
    print(f'  {ver}: {data[\"count\"]} logos (avg: {data[\"avg_score\"]:.1f})')
"
}

function show_help() {
    print_header
    echo ""
    echo "Uso: ./run.sh [comando]"
    echo ""
    echo "Comandos disponibles:"
    echo ""
    echo "  ${GREEN}setup${NC}         Instalar dependencias"
    echo "  ${GREEN}populate${NC}      Poblar base de conocimiento"
    echo "  ${GREEN}search${NC}        Buscar en base de conocimiento (demo)"
    echo "  ${GREEN}interactive${NC}   Modo de búsqueda interactivo"
    echo "  ${GREEN}generate${NC}      Generar logos con Gemini"
    echo "  ${GREEN}gallery${NC}       Generar y abrir galería de logos 🎨"
    echo "  ${GREEN}logo-stats${NC}    Ver estadísticas de logos generados"
    echo "  ${GREEN}stats${NC}         Ver estadísticas de la base de datos"
    echo "  ${GREEN}jupyter${NC}       Abrir Jupyter notebooks"
    echo "  ${GREEN}help${NC}          Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo ""
    echo "  ${YELLOW}# Instalación inicial${NC}"
    echo "  ./run.sh setup"
    echo "  ./run.sh populate"
    echo ""
    echo "  ${YELLOW}# Explorar base de conocimiento${NC}"
    echo "  ./run.sh search"
    echo "  ./run.sh interactive"
    echo ""
    echo "  ${YELLOW}# Generar logos${NC}"
    echo "  export GCP_PROJECT_ID=tu-project-id"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json"
    echo "  ./run.sh generate"
    echo ""
    echo "Documentación: cat README.md | less"
    echo "Quick start:   cat QUICKSTART.md | less"
    echo ""
}

# Main
case "${1:-help}" in
    setup)
        install_deps
        ;;
    populate)
        populate_kb
        ;;
    search)
        search_kb
        ;;
    interactive|i)
        search_interactive
        ;;
    generate|gen)
        generate_logo
        ;;
    gallery)
        gallery
        ;;
    logo-stats|logos)
        logo_stats
        ;;
    stats)
        stats
        ;;
    jupyter|notebook)
        jupyter
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Comando desconocido: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
