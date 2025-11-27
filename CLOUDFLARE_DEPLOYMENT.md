# 🌐 Deployment a Cloudflare Pages

**Status:** ✅ Completamente Configurado
**Fecha:** 25 Noviembre 2025

---

## 🎯 Overview

Sistema completo para deployar la galería de logos a Cloudflare Pages con:
- ✅ Subdominio gratuito (*.pages.dev)
- ✅ Custom domain opcional
- ✅ SSL automático
- ✅ CDN global
- ✅ Deployment en 2 minutos

---

## 🚀 Quick Start (Método Fácil)

### Paso 1: Instalar Wrangler

```bash
npm install -g wrangler
```

### Paso 2: Login a Cloudflare

```bash
wrangler login
# Se abre navegador → Login → Autorizar
```

### Paso 3: Deploy

```bash
cd ~/svg-logo-ai
./deploy.sh deploy
```

**¡Listo!** Tu galería está en: `https://logo-gallery-ai.pages.dev`

---

## 📋 Setup Completo

### 1. Requisitos Previos

- Cuenta de Cloudflare (gratis): https://dash.cloudflare.com/sign-up
- Node.js instalado (para npm/wrangler)
- Galería generada (`./run.sh gallery`)

### 2. Instalación de Wrangler

```bash
# Instalar globalmente
npm install -g wrangler

# Verificar instalación
wrangler --version
```

### 3. Autenticación

**Opción A: Login Interactivo (Recomendado)**
```bash
wrangler login
```
- Se abre navegador
- Login a Cloudflare
- Autorizar Wrangler

**Opción B: Con API Token**
```bash
# 1. Crear token en: https://dash.cloudflare.com/profile/api-tokens
# 2. Permisos necesarios: "Cloudflare Pages - Edit"
# 3. Exportar:
export CLOUDFLARE_API_TOKEN=tu_token_aqui
```

**Opción C: Archivo .env**
```bash
cp .env.cloudflare.example .env.cloudflare
# Editar .env.cloudflare con tu token
```

---

## 🎨 Deployment de la Galería

### Método 1: Script Automático (Recomendado)

```bash
cd ~/svg-logo-ai
./deploy.sh deploy
```

**El script hace:**
1. ✅ Verifica que wrangler esté instalado
2. ✅ Verifica autenticación
3. ✅ Prepara archivos (genera galería si hace falta)
4. ✅ Copia SVGs y metadata
5. ✅ Deploya a Cloudflare Pages
6. ✅ Te da la URL final

**Output:**
```
═══════════════════════════════════════════
  Cloudflare Pages Deployment
═══════════════════════════════════════════

✓ Wrangler encontrado
✓ Autenticado: user@example.com

📦 Preparando sitio...
✓ Copiado: gallery.html → index.html
✓ Copiados: 5 archivos SVG
✓ Copiado: logos_metadata.json

✓ Sitio preparado en: output/deploy/

═══════════════════════════════════════════
  Cloudflare Pages Deployment
═══════════════════════════════════════════

🚀 Deployando a Cloudflare Pages...

   Proyecto: logo-gallery-ai
   Archivos: 7

✅ Uploading... (100%)
✅ Success! Deployed to https://logo-gallery-ai.pages.dev

✓ Deployment completado!

═══════════════════════════════════════════
✅ SITIO DEPLOYADO
═══════════════════════════════════════════

🌍 URL: https://logo-gallery-ai.pages.dev
📁 Proyecto: logo-gallery-ai

═══════════════════════════════════════════
```

### Método 2: Manual con Wrangler

```bash
# 1. Preparar archivos
./deploy.sh prepare

# 2. Deploy
cd output/deploy
wrangler pages deploy . --project-name=logo-gallery-ai
```

### Método 3: Python API (Avanzado)

```bash
# Configurar .env.cloudflare primero
cp .env.cloudflare.example .env.cloudflare
# Editar con tus credenciales

# Deploy usando Python
cd src
python cloudflare_deployer.py
```

---

## 🌐 Custom Domain (Subdominio Personalizado)

### Si tu dominio está en Cloudflare:

**1. Configurar en .env.cloudflare:**
```bash
CUSTOM_DOMAIN=logos.tudominio.com
```

**2. Deployar con custom domain:**
```bash
./deploy.sh deploy
./deploy.sh domain
```

**3. DNS se configura automáticamente** ✅

### Si tu dominio está en otro proveedor:

**1. Agregar custom domain:**
```bash
# En .env.cloudflare:
CUSTOM_DOMAIN=logos.tudominio.com

# Deploy
./deploy.sh deploy
./deploy.sh domain
```

**2. Configurar DNS en tu proveedor:**
```
Tipo:   CNAME
Nombre: logos
Target: logo-gallery-ai.pages.dev
```

**3. Esperar propagación DNS** (5-30 minutos)

### Verificar Custom Domain:

```bash
# Ver status
./deploy.sh status

# O manualmente
wrangler pages deployment list --project-name=logo-gallery-ai
```

---

## 📁 Estructura de Deployment

```
output/deploy/           ← Directorio deployado
├── index.html          ← Galería (gallery.html renombrado)
├── logos_metadata.json ← Metadata de logos
├── techflow_logo.svg   ← SVG logos
├── vitalcare_logo.svg
└── ...
```

**Cloudflare Pages sirve:**
- `/` → index.html (galería principal)
- `/techflow_logo.svg` → Logo SVG
- `/logos_metadata.json` → Data para JS

---

## 🔄 Re-deployments y Updates

### Actualizar galería después de generar nuevos logos:

```bash
# 1. Generar nuevos logos
cd ~/svg-logo-ai
source venv/bin/activate
cd src
python gemini_svg_generator_v2.py

# 2. Regenerar galería
cd ..
./run.sh gallery

# 3. Re-deploy
./deploy.sh deploy
```

**Cloudflare automáticamente:**
- ✅ Actualiza el sitio
- ✅ Mantiene la misma URL
- ✅ Invalida cache
- ✅ Deploy toma ~30 segundos

### Ver historial de deployments:

```bash
./deploy.sh status
```

O en dashboard:
https://dash.cloudflare.com → Pages → logo-gallery-ai

---

## ⚙️ Configuración Avanzada

### Cambiar nombre del proyecto:

**En .env.cloudflare:**
```bash
CLOUDFLARE_PROJECT_NAME=mi-galeria-logos
```

**Re-deploy:**
```bash
./deploy.sh deploy
```

**Nueva URL:**
```
https://mi-galeria-logos.pages.dev
```

### Variables de entorno en producción:

**En wrangler.toml:**
```toml
[env.production]
MY_VAR = "valor"
```

### Redirects y Headers:

**Crear `output/deploy/_redirects`:**
```
/old-url  /new-url  301
/api/*    https://api.backend.com/:splat  200
```

**Crear `output/deploy/_headers`:**
```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
```

---

## 🔐 Seguridad y Privacidad

### Proteger con Access (opcional):

```bash
# Requiere Cloudflare Access (plan pago)
wrangler pages deployment create \
  --project-name=logo-gallery-ai \
  --branch=main \
  --access-allowed-emails=tu@email.com
```

### Agregar autenticación básica:

**Crear `output/deploy/_worker.js`:**
```javascript
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const auth = request.headers.get('authorization')

  if (!auth || !verifyAuth(auth)) {
    return new Response('Unauthorized', {
      status: 401,
      headers: {
        'WWW-Authenticate': 'Basic realm="Logo Gallery"'
      }
    })
  }

  return fetch(request)
}

function verifyAuth(auth) {
  const encoded = auth.split(' ')[1]
  const decoded = atob(encoded)
  return decoded === 'admin:password123'  // ← Cambiar
}
```

---

## 📊 Monitoreo y Analytics

### Cloudflare Web Analytics (Gratis):

**1. Habilitar en Dashboard:**
https://dash.cloudflare.com → Web Analytics

**2. Agregar script en galería:**

Editar `src/gallery_generator.py`, agregar antes de `</body>`:
```html
<script defer src='https://static.cloudflareinsights.com/beacon.min.js'
        data-cf-beacon='{"token": "TU_TOKEN_AQUI"}'></script>
```

**3. Ver analytics:**
https://dash.cloudflare.com → Web Analytics

### Logs de deployment:

```bash
# Ver logs en tiempo real
wrangler pages deployment tail --project-name=logo-gallery-ai

# O en dashboard
https://dash.cloudflare.com → Pages → logo-gallery-ai → Deployments
```

---

## 🐛 Troubleshooting

### Error: "wrangler: command not found"

```bash
# Reinstalar
npm install -g wrangler

# Verificar PATH
echo $PATH

# O usar npx
npx wrangler login
npx wrangler pages deploy . --project-name=logo-gallery-ai
```

### Error: "Authentication error"

```bash
# Re-login
wrangler logout
wrangler login

# O usar token
export CLOUDFLARE_API_TOKEN=tu_token_aqui
```

### Error: "Project already exists"

```bash
# Usar nombre diferente
export CLOUDFLARE_PROJECT_NAME=logo-gallery-ai-v2
./deploy.sh deploy

# O eliminar proyecto existente
wrangler pages project delete logo-gallery-ai
```

### Deployment stuck / timeout

```bash
# Verificar tamaño de archivos
du -sh output/deploy

# Si es muy grande (>25MB), optimizar SVGs
# Cloudflare Pages límite: 25MB por deployment

# Ver status
wrangler pages deployment list --project-name=logo-gallery-ai
```

### Custom domain no funciona

```bash
# 1. Verificar DNS
dig logos.tudominio.com

# 2. Verificar en dashboard
https://dash.cloudflare.com → Pages → logo-gallery-ai → Custom domains

# 3. Re-intentar
./deploy.sh domain
```

---

## 💰 Costos

**Cloudflare Pages - Plan Free:**
- ✅ Deployments ilimitados
- ✅ 500 builds/mes
- ✅ Bandwidth ilimitado
- ✅ SSL gratis
- ✅ CDN global
- ✅ 1 concurrent build

**Perfectamente suficiente para la galería de logos** ✅

**Plan Pro ($20/mes):**
- 5,000 builds/mes
- 5 concurrent builds
- Más previews

---

## 🔄 CI/CD con GitHub (Opcional)

### Setup:

**1. Crear repo en GitHub**

**2. Push código:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/tu-usuario/logo-gallery-ai.git
git push -u origin main
```

**3. Conectar Cloudflare Pages:**
- Dashboard → Pages → Create project
- Connect to Git → GitHub → Autorizar
- Seleccionar repo: logo-gallery-ai
- Build settings:
  - Framework: None
  - Build command: `./deploy.sh prepare`
  - Build output: `output/deploy`
- Save and Deploy

**4. Auto-deployment:**
- Cada push a `main` → Deploy automático
- Pull requests → Preview deployments

---

## 📝 Comandos Útiles

```bash
# Deploy
./deploy.sh deploy

# Solo preparar (sin deployar)
./deploy.sh prepare

# Configurar custom domain
./deploy.sh domain

# Ver status de deployments
./deploy.sh status

# Login/re-login
./deploy.sh login

# Instalar/verificar wrangler
./deploy.sh setup

# Ver logs en vivo
wrangler pages deployment tail --project-name=logo-gallery-ai

# Rollback a deployment anterior
wrangler pages deployment list --project-name=logo-gallery-ai
wrangler pages deployment rollback <deployment-id>

# Eliminar proyecto
wrangler pages project delete logo-gallery-ai
```

---

## ✅ Checklist de Deployment

### Primera Vez:
- [ ] Node.js instalado
- [ ] `npm install -g wrangler`
- [ ] `wrangler login`
- [ ] Galería generada (`./run.sh gallery`)
- [ ] `./deploy.sh deploy`
- [ ] Verificar URL: https://logo-gallery-ai.pages.dev

### Con Custom Domain:
- [ ] Dominio registrado
- [ ] `.env.cloudflare` configurado con `CUSTOM_DOMAIN`
- [ ] `./deploy.sh domain`
- [ ] DNS configurado (CNAME)
- [ ] Esperar propagación (~30 min)
- [ ] Verificar: https://logos.tudominio.com

### Updates Regulares:
- [ ] Generar nuevos logos
- [ ] `./run.sh gallery` (regenerar)
- [ ] `./deploy.sh deploy`
- [ ] Verificar actualización

---

## 🎉 Resultado Final

Después de deployment exitoso tendrás:

- 🌍 **URL pública:** https://logo-gallery-ai.pages.dev
- 🔒 **SSL:** Automático (HTTPS)
- 🚀 **CDN:** Global (Cloudflare)
- 📊 **Performance:** Excelente
- 💰 **Costo:** $0 (plan free)
- ⚡ **Deploy time:** ~30 segundos
- 🔄 **Updates:** Instantáneos

**Tu galería de logos ahora es profesional y compartible con todo el mundo** ✨

---

## 📚 Recursos

- **Cloudflare Pages Docs:** https://developers.cloudflare.com/pages
- **Wrangler Docs:** https://developers.cloudflare.com/workers/wrangler
- **Dashboard:** https://dash.cloudflare.com
- **Status:** https://www.cloudflarestatus.com
- **Community:** https://community.cloudflare.com

---

**¿Listo para deployar?**

```bash
cd ~/svg-logo-ai
./deploy.sh deploy
```

🚀 **¡A deployar!**
