# Cloudflare Pages - Guía de Setup

## Opción 1: Login Interactivo (Más Fácil)

Ejecutá este comando desde tu servidor:

```bash
cd /home/luis/svg-logo-ai/web
npx wrangler login
```

Esto va a:
1. Abrir tu navegador
2. Pedirte que autorices Wrangler
3. Guardar las credenciales automáticamente

Luego podés deployar con:
```bash
npx wrangler pages deploy . --project-name=svg-logo-ai-results
```

## Opción 2: API Token Manual

Si el login interactivo no funciona (servidor sin GUI), seguí estos pasos:

### Paso 1: Crear API Token en Cloudflare

1. Ir a: https://dash.cloudflare.com/profile/api-tokens
2. Click en "Create Token"
3. Usar template "Edit Cloudflare Workers"
4. O crear custom con permisos:
   - Account → Cloudflare Pages → Edit
   - Zone → DNS → Edit (opcional)
5. Copiar el token (lo verás solo una vez)

### Paso 2: Configurar Token

Opción A - Variable de entorno (temporal):
```bash
export CLOUDFLARE_API_TOKEN="tu-token-aqui"
export CLOUDFLARE_ACCOUNT_ID="5de70f4ba8110b9cf400a3157ff420c3"
```

Opción B - Archivo config (permanente):
```bash
mkdir -p ~/.wrangler
cat > ~/.wrangler/config/default.toml <<EOF
api_token = "tu-token-aqui"
account_id = "5de70f4ba8110b9cf400a3157ff420c3"
EOF
```

### Paso 3: Deploy

```bash
cd /home/luis/svg-logo-ai/web
npx wrangler pages deploy . --project-name=svg-logo-ai-results
```

## Opción 3: Usar Cloudflare Dashboard (Sin CLI)

Si no querés usar Wrangler:

1. Ir a: https://dash.cloudflare.com/
2. Click en "Workers & Pages" → "Create application"
3. Click en "Pages" → "Upload assets"
4. Arrastrar toda la carpeta `/home/luis/svg-logo-ai/web/`
5. Project name: `svg-logo-ai-results`
6. Click "Deploy site"

Listo! Tu sitio estará en: `https://svg-logo-ai-results.pages.dev`

## Datos que Tengo

Ya tenés configurado:
- **Account ID**: `5de70f4ba8110b9cf400a3157ff420c3`

Solo falta:
- **API Token**: Necesitás crearlo (ver Paso 1 arriba)

## Quick Start

Para deployar rápido sin Cloudflare:

### Usar GitHub Pages (Gratis si hacés repo público)

```bash
# 1. Hacer repo público
gh repo edit larancibia/svg-logo-ai --visibility public

# 2. Habilitar Pages en:
# https://github.com/larancibia/svg-logo-ai/settings/pages
# Branch: master, Folder: /web

# URL final:
# https://larancibia.github.io/svg-logo-ai/
```

### O usar Netlify (Gratis, más fácil)

```bash
cd /home/luis/svg-logo-ai/web
npx netlify-cli deploy --dir=. --prod
```

Netlify te va a pedir login una sola vez y después es automático.

## Recomendación

**Más fácil**: GitHub Pages (hacer repo público)
**Mejor para privados**: Cloudflare Pages (necesita API token)
**Alternativa**: Netlify (también funciona con privados)

¿Qué preferís?
