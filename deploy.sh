#!/bin/bash

# Script de deployment a Cloudflare Pages
# Usa Wrangler CLI (método recomendado)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Cloudflare Pages Deployment${NC}"
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

function check_wrangler() {
    if ! command -v wrangler &> /dev/null; then
        print_warning "Wrangler no instalado"
        echo ""
        echo "Instalando Wrangler CLI..."
        npm install -g wrangler
        print_success "Wrangler instalado"
    else
        print_success "Wrangler encontrado"
    fi
}

function check_auth() {
    if ! wrangler whoami &> /dev/null; then
        print_warning "No estás autenticado en Cloudflare"
        echo ""
        echo "Opciones de autenticación:"
        echo ""
        echo "  ${YELLOW}1. Login interactivo (recomendado):${NC}"
        echo "     wrangler login"
        echo ""
        echo "  ${YELLOW}2. Con API Token:${NC}"
        echo "     export CLOUDFLARE_API_TOKEN=tu_token_aqui"
        echo ""
        echo "  ${YELLOW}3. Editar .env.cloudflare:${NC}"
        echo "     cp .env.cloudflare.example .env.cloudflare"
        echo "     # Completar con tus credenciales"
        echo ""
        read -p "¿Quieres hacer login ahora? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            wrangler login
        else
            print_error "Deployment cancelado"
            exit 1
        fi
    else
        WHOAMI=$(wrangler whoami 2>&1)
        print_success "Autenticado: $WHOAMI"
    fi
}

function prepare_site() {
    print_header
    echo ""
    echo "📦 Preparando sitio..."

    # Crear directorio de deployment
    mkdir -p output/deploy

    # Activar venv
    source venv/bin/activate

    # Generar galería si no existe
    if [ ! -f "output/gallery.html" ]; then
        print_warning "Galería no encontrada, generando..."
        cd src
        python gallery_generator.py
        cd ..
    fi

    # Copiar archivos
    cp output/gallery.html output/deploy/index.html
    print_success "Copiado: gallery.html → index.html"

    # Copiar SVGs
    SVG_COUNT=$(ls -1 output/*.svg 2>/dev/null | wc -l)
    if [ "$SVG_COUNT" -gt 0 ]; then
        cp output/*.svg output/deploy/ 2>/dev/null || true
        print_success "Copiados: $SVG_COUNT archivos SVG"
    fi

    # Copiar metadata
    if [ -f "output/logos_metadata.json" ]; then
        cp output/logos_metadata.json output/deploy/
        print_success "Copiado: logos_metadata.json"
    fi

    echo ""
    echo "✓ Sitio preparado en: output/deploy/"
}

function deploy_to_cloudflare() {
    print_header
    echo ""
    echo "🚀 Deployando a Cloudflare Pages..."
    echo ""

    # Check config
    if [ -f ".env.cloudflare" ]; then
        source .env.cloudflare
        PROJECT_NAME="${CLOUDFLARE_PROJECT_NAME:-logo-gallery-ai}"
    else
        PROJECT_NAME="logo-gallery-ai"
        print_warning "Usando nombre de proyecto por defecto: $PROJECT_NAME"
    fi

    # Deploy
    cd output/deploy

    echo "   Proyecto: $PROJECT_NAME"
    echo "   Archivos: $(ls -1 | wc -l)"
    echo ""

    wrangler pages deploy . \
        --project-name="$PROJECT_NAME" \
        --branch=main

    cd ../..

    # Get URL
    SITE_URL="https://$PROJECT_NAME.pages.dev"

    echo ""
    print_success "Deployment completado!"
    echo ""
    echo "═══════════════════════════════════════════"
    echo -e "${GREEN}✅ SITIO DEPLOYADO${NC}"
    echo "═══════════════════════════════════════════"
    echo ""
    echo -e "🌍 URL: ${BLUE}$SITE_URL${NC}"
    echo -e "📁 Proyecto: ${BLUE}$PROJECT_NAME${NC}"
    echo ""
    echo "═══════════════════════════════════════════"
    echo ""
}

function setup_custom_domain() {
    print_header
    echo ""
    echo "🌐 Configurando custom domain..."
    echo ""

    if [ -f ".env.cloudflare" ]; then
        source .env.cloudflare
        PROJECT_NAME="${CLOUDFLARE_PROJECT_NAME:-logo-gallery-ai}"

        if [ -n "$CUSTOM_DOMAIN" ]; then
            echo "   Domain: $CUSTOM_DOMAIN"
            echo ""
            wrangler pages domain add "$CUSTOM_DOMAIN" --project-name="$PROJECT_NAME"
            print_success "Domain configurado!"
            echo ""
            echo "═══════════════════════════════════════════"
            echo -e "${GREEN}Configuración DNS${NC}"
            echo "═══════════════════════════════════════════"
            echo ""
            echo "Si tu dominio está en Cloudflare:"
            echo "  → DNS se configurará automáticamente"
            echo ""
            echo "Si tu dominio está en otro proveedor:"
            echo "  → Crea un CNAME:"
            echo "     Nombre: $(echo $CUSTOM_DOMAIN | cut -d. -f1)"
            echo "     Target: $PROJECT_NAME.pages.dev"
            echo ""
        else
            print_warning "CUSTOM_DOMAIN no configurado en .env.cloudflare"
        fi
    else
        print_warning ".env.cloudflare no encontrado"
        echo ""
        echo "Para configurar custom domain:"
        echo "  1. Copia .env.cloudflare.example a .env.cloudflare"
        echo "  2. Agrega: CUSTOM_DOMAIN=logos.tudominio.com"
        echo "  3. Ejecuta: ./deploy.sh domain"
    fi
}

function show_help() {
    print_header
    echo ""
    echo "Uso: ./deploy.sh [comando]"
    echo ""
    echo "Comandos:"
    echo ""
    echo "  ${GREEN}deploy${NC}     Deployar galería a Cloudflare Pages"
    echo "  ${GREEN}prepare${NC}    Solo preparar archivos (sin deployar)"
    echo "  ${GREEN}domain${NC}     Configurar custom domain"
    echo "  ${GREEN}setup${NC}      Instalar dependencias (wrangler)"
    echo "  ${GREEN}login${NC}      Login a Cloudflare"
    echo "  ${GREEN}status${NC}     Ver info de deployment"
    echo "  ${GREEN}help${NC}       Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo ""
    echo "  ${YELLOW}# Deployment completo${NC}"
    echo "  ./deploy.sh deploy"
    echo ""
    echo "  ${YELLOW}# Con custom domain${NC}"
    echo "  cp .env.cloudflare.example .env.cloudflare"
    echo "  # Editar .env.cloudflare"
    echo "  ./deploy.sh deploy"
    echo "  ./deploy.sh domain"
    echo ""
}

# Main
case "${1:-deploy}" in
    deploy)
        check_wrangler
        check_auth
        prepare_site
        deploy_to_cloudflare
        ;;
    prepare)
        prepare_site
        ;;
    domain)
        check_wrangler
        check_auth
        setup_custom_domain
        ;;
    setup)
        check_wrangler
        ;;
    login)
        wrangler login
        ;;
    status)
        check_wrangler
        check_auth
        if [ -f ".env.cloudflare" ]; then
            source .env.cloudflare
            PROJECT_NAME="${CLOUDFLARE_PROJECT_NAME:-logo-gallery-ai}"
        else
            PROJECT_NAME="logo-gallery-ai"
        fi
        wrangler pages deployment list --project-name="$PROJECT_NAME"
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
