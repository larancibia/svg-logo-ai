# Logo Datasets para Entrenamiento/Evaluación de Modelos

Documento de investigación sobre datasets disponibles para entrenar y evaluar modelos de generación de logos.

**Última actualización:** 25 de Noviembre, 2025

---

## Tabla Comparativa de Datasets

| Dataset | Tamaño | Formato | Metadatos | Licencia | Calidad | Estado Descarga |
|---------|--------|---------|-----------|----------|---------|-----------------|
| **SVG-1M** | 1M pares texto-SVG | SVG | Instrucciones, CoT, colores | Académico/No-comercial | Profesional | ✅ Disponible (HuggingFace) |
| **L3D** | 770K imágenes | PNG (256x256) | Vienna Classification | Abierto | Profesional (EUIPO) | ✅ Disponible (Zenodo) |
| **LLD-logo** | 122,920 logos | PNG (alta res) | Básico | Abierto | Mixto | ✅ Disponible |
| **LLD-icon** | 548,210 favicons | PNG/HDF5 (32x32) | Básico | Abierto | Amateur/Mixto | ✅ Disponible |
| **LogoDet-3K** | 3K clases, 200K+ imgs | JPG + bbox | Industria, empresa | MIT | Profesional | ✅ Disponible (GitHub/HF) |
| **QMUL-OpenLogo** | 27,083 imágenes | JPG + bbox | 352 clases | Académico | Profesional | ✅ Disponible (4.7GB) |
| **FlickrLogos-32** | 8,240 imágenes | JPG | 32 clases de marcas | Flickr ToS | Real-world | ⚠️ Requiere email |
| **WebLogo-2M** | 1.87M imágenes | JPG | 194 clases | N/A | Mixto | ❌ Enlaces rotos |
| **SVG-Icons8** | 100K iconos | SVG (tensor) | 56 categorías | Icons8 ToS | Profesional | ✅ Disponible (3GB) |
| **The Noun Project** | 8M+ iconos | SVG/PNG | Tags, categorías | Freemium API | Profesional | 🔑 API ($150/mes SVG) |

---

## 1. Datasets Principales

### 1.1 SVG-1M (2024-2025) ⭐ RECOMENDADO para Fine-tuning

**Descripción:** Dataset más reciente y completo para generación de logos en formato SVG.

**Especificaciones:**
- **Tamaño total:** ~1 millón de pares texto-SVG
  - 826,326 pares monocromáticos
  - 137,460 pares multicolor
  - 65,745 pares con anotaciones Chain-of-Thought
- **Formato:** SVG (código vectorial real)
- **Fuente:** Iconfont (Alibaba Vector Icon Library)
- **Canvas:** 1024×1024 pixels, estándares SVG namespace
- **Metadatos:**
  - Instrucciones en lenguaje natural
  - Descripciones detalladas
  - Anotaciones CoT para razonamiento
  - Categorización por tipo (monocromo/multicolor)

**Descarga:**
```bash
# Hugging Face
from datasets import load_dataset
dataset = load_dataset("SVG-1M-Json")
```
- **Repositorio:** https://github.com/gitcat-404/SVGen
- **HuggingFace:** SVG-1M-Json repository

**Licencia:** Uso académico/no-comercial (scraping de contenido público de Iconfont)

**Ventajas:**
- ✅ Formato SVG nativo (código vectorial)
- ✅ Pares texto-código alineados
- ✅ Anotaciones CoT para interpretabilidad
- ✅ Dataset más moderno (2024-2025)
- ✅ Normalizado y listo para LLMs

**Limitaciones:**
- ❌ Solo para uso no-comercial
- ❌ Principalmente iconos (no logos complejos de marca)

**Calificación para fine-tuning:** 10/10 (Ideal para modelos generativos)

---

### 1.2 L3D - Large Labelled Logo Dataset

**Descripción:** Dataset masivo de logos profesionales del registro de propiedad intelectual europeo.

**Especificaciones:**
- **Tamaño:** ~770,000 imágenes
- **Formato:** PNG RGB 256×256
- **Fuente:** European Union Intellectual Property Office (EUIPO)
- **Metadatos:**
  - Clasificación Vienna (elementos figurativos/textuales)
  - Etiquetas múltiples por imagen
  - Anotaciones profesionales de evaluadores EUIPO

**Descarga:**
- **Zenodo:** https://zenodo.org/records/5771006
- **GitHub:** https://github.com/lhf-labs/tm-dataset
- **Website:** https://lhf-labs.github.io/tm-dataset/

**Licencia:** Dataset abierto (verificar términos EUIPO)

**Ventajas:**
- ✅ Logos profesionales reales
- ✅ Anotaciones de alta calidad
- ✅ Clasificación estructurada (Vienna)
- ✅ Gran volumen
- ✅ Diseñado para clasificación y generación

**Limitaciones:**
- ❌ Formato raster (PNG), no vectorial
- ❌ Resolución limitada (256×256)
- ❌ Puede contener marcas registradas

**Calificación para fine-tuning:** 8/10 (Excelente para clasificación y embeddings)

---

### 1.3 LLD - Large Logo Dataset

**Descripción:** Dataset histórico de logos crawleados de la web.

**Versiones:**

#### LLD-logo (alta resolución)
- **Tamaño:** 122,920 logos
- **Formato:** PNG (resoluciones variables)
- **Descarga:** https://data.vision.ee.ethz.ch/sagea/lld/

#### LLD-icon (favicons)
- **Tamaño:** 548,210 favicons
- **Formato:**
  - Python Pickle (100K logos/archivo)
  - HDF5 (formato único)
  - PNG individuales
- **Dimensiones:** 32×32×3 RGB (TensorFlow-ready)
- **Fuente:** Alexa Top 1M websites (2017)

**Ventajas:**
- ✅ Gran volumen
- ✅ Múltiples formatos (HDF5, PKL, PNG)
- ✅ Listo para TensorFlow/PyTorch
- ✅ Acceso público directo

**Limitaciones:**
- ❌ Baja resolución (especialmente favicons)
- ❌ Calidad mixta (amateur + profesional)
- ❌ Sin metadatos semánticos
- ❌ Dataset antiguo (2017)

**Calificación para fine-tuning:** 5/10 (Útil para aumentar volumen, pero baja calidad)

---

### 1.4 LogoDet-3K

**Descripción:** Dataset para detección de logos con bounding boxes.

**Especificaciones:**
- **Tamaño:** 3,000+ clases de logos, 200,000+ imágenes
- **Formato:** JPG + anotaciones XML/JSON
- **Metadatos:**
  - Nombre de industria
  - Nombre de empresa
  - Bounding boxes (xmin, ymin, xmax, ymax)
- **Descarga:**
  - **GitHub:** https://github.com/Wangjing1551/LogoDet-3K-Dataset
  - **HuggingFace:** `load_dataset("PodYapolsky/LogoDet-3K")`
  - **Kaggle:** LogoDet3K
  - **Servidor:** 123.57.42.89/Dataset_ict/LogoDet-3K.zip (password: 1234)

**Licencia:** MIT

**Ventajas:**
- ✅ Licencia MIT (permisiva)
- ✅ Metadatos ricos (industria, empresa)
- ✅ Múltiples fuentes de descarga
- ✅ Logos en contexto (no aislados)

**Limitaciones:**
- ❌ Formato raster
- ❌ Orientado a detección, no generación
- ❌ Logos en escenas reales (no aislados)

**Calificación para fine-tuning:** 6/10 (Mejor para detección que generación)

---

### 1.5 QMUL-OpenLogo

**Descripción:** Dataset para logo detection con protocolo de evaluación abierto.

**Especificaciones:**
- **Tamaño:** 27,083 imágenes
- **Clases:** 352 logos de marcas conocidas
- **Formato:** JPG + bounding boxes
- **Fuente:** Agregación de 7 datasets existentes refinados
- **Descarga:**
  - Google Drive (4.7 GB)
  - Baidu Cloud
  - Tencent Cloud
  - **Website:** https://hangsu0730.github.io/qmul-openlogo/

**Publicación:** BMVC 2018

**Licencia:** Solo uso académico

**Ventajas:**
- ✅ Dataset curado de alta calidad
- ✅ 352 clases bien balanceadas
- ✅ Protocolo de evaluación estándar
- ✅ Logos profesionales reconocibles

**Limitaciones:**
- ❌ Solo uso académico
- ❌ Formato raster
- ❌ Orientado a detección

**Calificación para fine-tuning:** 7/10 (Bueno para clasificación y embedding)

---

### 1.6 FlickrLogos-32/47

**Descripción:** Datasets de logos en imágenes del mundo real de Flickr.

**Versiones:**
- **FlickrLogos-32:** 8,240 imágenes, 32 clases (Adidas, Apple, BMW, Coca-Cola, Google, etc.)
- **FlickrLogos-47:** Mismas imágenes, re-anotadas con 47 clases

**Descarga:** Requiere solicitud por email a request_flickrlogos@informatik.uni-augsburg.de

**Licencia:** Sujeto a Flickr Terms of Service

**Ventajas:**
- ✅ Logos en contexto real
- ✅ Marcas reconocibles
- ✅ Útil para evaluación

**Limitaciones:**
- ❌ Proceso de descarga manual
- ❌ Tamaño pequeño
- ❌ Licencia restrictiva (Flickr ToS)

**Calificación para fine-tuning:** 4/10 (Mejor para evaluación)

---

### 1.7 WebLogo-2M ❌ NO DISPONIBLE

**Descripción:** Dataset masivo de logos de Twitter (histórico).

**Especificaciones:**
- **Tamaño:** 1,867,177 imágenes
- **Clases:** 194 logos
- **Etiquetado:** Débil (a nivel imagen, no bounding box)
- **Website:** https://weblogo2m.github.io/

**Estado:** ⚠️ Enlaces de descarga rotos, dataset no accesible

**Calificación:** N/A (No disponible actualmente)

---

### 1.8 SVG-Icons8 (DeepSVG Dataset) ⭐ RECOMENDADO para SVG

**Descripción:** Dataset de 100K iconos en formato SVG para deep learning.

**Especificaciones:**
- **Tamaño:** 100,000 iconos
- **Formato:** SVG convertido a PyTorch tensors
- **Categorías:** 56 diferentes
- **Fuente:** https://icons8.com
- **Paper:** DeepSVG (NeurIPS 2020)

**Descarga:**
- **icons_meta.csv** (9 MB): [Google Drive](https://drive.google.com/file/d/10Zx4TB1-BEdWv1GbwcSUl2-uRFiqgUP1/view)
- **icons_tensor.zip** (3 GB): [Google Drive](https://drive.google.com/file/d/1gTuO3k98u_Y1rvpSbJFbqgCf6AJi2qIA/view)
- **GitHub:** https://github.com/alexandre01/deepsvg

**Licencia:** Icons8 Terms of Service (verificar para uso comercial)

**Ventajas:**
- ✅ Formato SVG nativo
- ✅ Pre-procesado para PyTorch
- ✅ Dataset usado en paper NeurIPS
- ✅ Biblioteca DeepSVG incluida

**Limitaciones:**
- ❌ Pre-procesado (tensors, no SVG raw)
- ❌ Iconos, no logos complejos
- ❌ Requiere plan pago de Icons8 para SVG originales

**Calificación para fine-tuning:** 9/10 (Excelente para aprendizaje de representaciones SVG)

---

## 2. Recursos Complementarios

### 2.1 The Noun Project

**Descripción:** Biblioteca masiva de iconos con API.

**Especificaciones:**
- **Tamaño:** 8+ millones de iconos
- **Formato:** SVG, PNG
- **Metadatos:** Tags, categorías, colecciones
- **API:** REST API con OAuth

**Acceso:**
- **Free Tier:**
  - 5,000 queries/mes
  - Solo iconos dominio público (PNG + SVG)
- **Paid Tier:**
  - Desde $150/mes
  - Acceso completo a SVG de todos los iconos
  - Query param: `?include_svg=1`

**API Docs:** https://api.thenounproject.com/documentation.html

**Ventajas:**
- ✅ Volumen masivo
- ✅ API bien documentada
- ✅ Metadatos ricos
- ✅ SVG de alta calidad

**Limitaciones:**
- ❌ Costoso para acceso completo
- ❌ URLs temporales (1 hora de expiración)
- ❌ Rate limits
- ❌ Principalmente iconos, no logos

**Uso recomendado:** Aumentación de datos, referencia visual

---

### 2.2 Brands of the World

**Descripción:** Biblioteca gratuita de logos vectoriales.

- **Website:** https://www.brandsoftheworld.com/
- **Formatos:** SVG, AI, EPS, PDF, CDR
- **Licencia:** Varía por logo (verificar individualmente)
- **Tamaño:** Miles de logos de marcas

**Ventajas:**
- ✅ Logos vectoriales profesionales
- ✅ Múltiples formatos
- ✅ Descarga gratuita

**Limitaciones:**
- ❌ No es un dataset estructurado
- ❌ Scraping manual requerido
- ❌ Problemas de copyright (marcas registradas)
- ❌ Sin API oficial

**Uso recomendado:** Referencia, ejemplos de estilos

---

### 2.3 LogoBook

**Descripción:** Galería curada de logos profesionales.

- **Website:** https://logobook.com/
- **Contenido:** 5,000+ logos
- **Filtros:** Diseñador, forma, objeto, estilo
- **Formato:** Visualización web (no dataset descargable)

**Limitaciones:**
- ❌ No es un dataset de ML
- ❌ Sin descarga masiva
- ❌ Solo galería visual

**Uso recomendado:** Inspiración, análisis de tendencias

---

### 2.4 Repositorios GitHub con Logos SVG

#### gilbarbara/logos
- **URL:** https://github.com/gilbarbara/logos
- **Contenido:** Colección masiva de logos SVG
- **Licencia:** Varía por logo
- **Uso:** Logos de tecnología, empresas conocidas

#### valohai/ml-logos
- **URL:** https://github.com/valohai/ml-logos
- **Contenido:** Logos SVG de bibliotecas ML/AI
- **Logos:** Caffe, Keras, NumPy, PyTorch, TensorFlow, etc.
- **Licencia:** Verificar por logo

**Ventajas:**
- ✅ Formato SVG
- ✅ Fácil de clonar
- ✅ Logos de alta calidad

**Limitaciones:**
- ❌ Volumen limitado
- ❌ Sin metadatos estructurados
- ❌ Posibles problemas de copyright

---

### 2.5 Otros Recursos SVG

#### SVG Repo
- **URL:** https://www.svgrepo.com/
- **Contenido:** 6000+ colecciones, 500K+ iconos SVG
- **Licencia:** Abierto/CC
- **Filtros:** Por color, estilo

#### Flaticon
- **URL:** https://www.flaticon.com/
- **Contenido:** 50,400+ iconos vectoriales
- **Formatos:** SVG, EPS, PSD, BASE64, Web Font
- **Licencia:** Freemium (atribución requerida)

#### FreeSVGIcons
- **URL:** https://freesvgicons.com/
- **Contenido:** 250,000+ iconos SVG
- **Fuente:** Bibliotecas open source agregadas
- **Licencia:** Open source

---

## 3. Análisis y Recomendaciones

### 3.1 Mejores Datasets para Fine-tuning

#### Top 3 para Generación de Logos:

**1. SVG-1M (Primera elección)** ⭐⭐⭐⭐⭐
- **Por qué:** Único dataset con código SVG real y pares texto-SVG
- **Uso ideal:** Fine-tuning de LLMs para generación SVG (Llama, GPT, Claude)
- **Modelo objetivo:** Seq2Seq, LLM-to-SVG
- **Limitación:** Solo iconos, no logos de marca complejos

**2. SVG-Icons8 (Segunda elección)** ⭐⭐⭐⭐
- **Por qué:** Formato SVG nativo, pre-procesado para deep learning
- **Uso ideal:** Modelos de generación jerárquica (DeepSVG-style)
- **Modelo objetivo:** VAE, GAN, Diffusion models para SVG
- **Limitación:** Tensors pre-procesados, no código SVG directo

**3. L3D (Tercera elección)** ⭐⭐⭐⭐
- **Por qué:** Logos profesionales reales, gran volumen, metadatos ricos
- **Uso ideal:** Fine-tuning de modelos de difusión (Stable Diffusion, FLUX)
- **Modelo objetivo:** Text-to-image, image-to-image
- **Limitación:** Formato raster, no vectorial

---

### 3.2 Datasets con Código/Paths SVG Reales

| Dataset | SVG Nativo | Formato Código | Accesibilidad |
|---------|------------|----------------|---------------|
| **SVG-1M** | ✅ Sí | Código SVG texto | HuggingFace |
| **SVG-Icons8** | ✅ Sí | Tensors PyTorch | Google Drive |
| **Icons8 (API)** | ✅ Sí | SVG descargable | API paga |
| **The Noun Project** | ✅ Sí | SVG vía API | API paga |
| **GitHub repos** | ✅ Sí | SVG archivos | Git clone |
| L3D | ❌ No | PNG raster | Zenodo |
| LLD | ❌ No | PNG/HDF5 | Web directo |
| LogoDet-3K | ❌ No | JPG | GitHub/HF |
| QMUL-OpenLogo | ❌ No | JPG | Google Drive |

**Conclusión:** Solo SVG-1M y SVG-Icons8 proporcionan datos SVG estructurados para entrenamiento directo de modelos generativos.

---

### 3.3 Estrategias por Tipo de Modelo

#### Para Modelos LLM (GPT-4, Claude, Llama):
```
Dataset recomendado: SVG-1M
Formato: Pares (texto, código SVG)
Approach: Fine-tuning con LoRA/QLoRA
Pipeline: Prompt → LLM → SVG code → Render
```

#### Para Modelos de Difusión (Stable Diffusion, FLUX):
```
Dataset recomendado: L3D + LogoDet-3K
Formato: Imágenes PNG/JPG + captions
Approach: DreamBooth/LoRA sobre SDXL
Pipeline: Prompt → Diffusion → PNG → Vectorización
```

#### Para Modelos VAE/GAN para SVG:
```
Dataset recomendado: SVG-Icons8
Formato: SVG tensors jerárquicos
Approach: Arquitectura DeepSVG-style
Pipeline: Latent → Decoder → SVG paths
```

#### Para Modelos Híbridos:
```
Dataset primario: SVG-1M (generación)
Dataset secundario: L3D (estilo/embedding)
Approach: Two-stage (diffusion → SVG conversion)
Pipeline: Prompt → Raster → Vectorization model → SVG
```

---

### 3.4 Limitaciones y Consideraciones Legales

#### Copyright y Marcas Registradas

**⚠️ ADVERTENCIA:** Muchos datasets contienen logos de marcas registradas.

**Riesgos legales:**
1. **Uso comercial:** Generar logos similares a marcas existentes puede violar trademark
2. **Dataset training:** Entrenar en logos protegidos está en zona gris legal
3. **Distribución:** Compartir modelos entrenados puede implicar licencias

**Datasets con más riesgo:**
- ❌ LogoDet-3K (marcas reales: Adidas, Apple, BMW, etc.)
- ❌ QMUL-OpenLogo (352 marcas conocidas)
- ❌ FlickrLogos (marcas específicas)
- ❌ Brands of the World (marcas registradas)

**Datasets más seguros:**
- ✅ SVG-1M (iconos genéricos, uso académico)
- ✅ L3D (logos del registro EUIPO, posible fair use)
- ✅ LLD (favicons genéricos)

#### Fair Use en Machine Learning

**Situación legal (2024-2025):**

**Estados Unidos:**
- Uso de copyrighted works para training **probablemente es Fair Use**
- Factores: transformativo, no-comercial, educacional
- Jurisprudencia aún en desarrollo

**Unión Europea:**
- Text-and-Data Mining (TDM) exception para investigación
- TDM comercial permitido con **opt-out** de rightholders
- Directiva DSM 2019

**Recomendaciones:**
1. **Uso académico:** Generalmente seguro con datasets abiertos
2. **Uso comercial:** Preferir datasets con licencias permisivas (MIT, CC)
3. **Generación:** No generar logos que imiten marcas registradas existentes
4. **Atribución:** Siempre dar crédito a fuentes de datos

#### Licencias Ambiguas

**Problema:** Muchos datasets tienen licencias poco claras.

**Ejemplos:**
- "Academic use only" → ¿Qué pasa con startups en incubadoras universitarias?
- "Non-commercial" → ¿Se puede usar en modelos open-source usados comercialmente?
- Flickr ToS → Cada imagen tiene su propia licencia

**Mejores prácticas:**
1. Documentar fuentes de todos los datos
2. Contactar a autores para clarificaciones
3. Tener políticas de uso responsable
4. Considerar legal counsel para lanzamiento comercial

---

### 3.5 Dataset Híbrido Recomendado

Para un proyecto de generación de logos robusto:

**Composición:**
```
Core training (70%): SVG-1M
  → Capacidad de generación SVG directa
  → Pares texto-código para fine-tuning LLM

Style reference (20%): L3D
  → Logos profesionales para estética
  → Embeddings de estilo via CLIP/DINOv2

Evaluation (10%): QMUL-OpenLogo
  → Benchmark contra logos reales conocidos
  → Métricas de similitud y clasificación
```

**Pipeline sugerido:**
1. Pre-train en SVG-1M para arquitectura SVG
2. Fine-tune en L3D para estilo de logos profesionales
3. Post-process con vectorización si es necesario
4. Evaluar en QMUL-OpenLogo para calidad

---

## 4. Recursos Adicionales

### Papers Relevantes (2024-2025)

1. **SVGen: Interpretable Vector Graphics Generation with LLMs** (2024)
   - Introduce SVG-1M dataset
   - ArXiv: https://arxiv.org/html/2508.09168v1
   - GitHub: https://github.com/gitcat-404/SVGen

2. **LogoSticker: Inserting Logos into Diffusion Models** (ECCV 2024)
   - Generación contextual de logos
   - ArXiv: https://arxiv.org/html/2407.13752v1

3. **DeepSVG: Hierarchical Generative Network for Vector Graphics** (NeurIPS 2020)
   - Introduce SVG-Icons8
   - GitHub: https://github.com/alexandre01/deepsvg

4. **L3D: Large Labelled Logo Dataset** (2021)
   - ArXiv: https://arxiv.org/abs/2112.05404
   - Zenodo: https://zenodo.org/records/5771006

### Tools y Frameworks

**SVG Processing:**
- **svgwrite** (Python): Generación SVG programática
- **svgpathtools** (Python): Manipulación de paths SVG
- **SVGO**: Optimización de SVG

**Vectorización:**
- **vtracer**: Bitmap → SVG vectorization
- **potrace**: Tracing de bitmaps
- **StarVector** (2025): SOTA model para vectorización

**Evaluación:**
- **SVG-Bench**: Benchmark para modelos SVG
- **FID Score**: Frechet Inception Distance
- **CLIP-Score**: Similaridad texto-imagen

---

## 5. Conclusiones

### TL;DR Recomendaciones

**Para fine-tuning de modelos generativos de logos:**

1. **Mejor opción SVG nativo:** SVG-1M
   - Único con código SVG + instrucciones
   - Ideal para LLMs (GPT, Llama, Claude)

2. **Mejor opción raster profesional:** L3D
   - Logos reales profesionales
   - Gran volumen, buena calidad
   - Ideal para Stable Diffusion/FLUX

3. **Mejor opción para research:** SVG-Icons8
   - Dataset académico establecido
   - Pre-procesado para deep learning
   - Paper NeurIPS de referencia

4. **Aumentación de datos:** The Noun Project API
   - 8M+ iconos SVG
   - Metadatos ricos
   - Requiere inversión ($150/mes)

### Roadmap de Implementación

**Fase 1: Proof of Concept**
```
Dataset: SVG-1M (subset 10K samples)
Modelo: Llama-3-8B + LoRA
Objetivo: Generar SVG simple desde texto
Timeline: 1-2 semanas
```

**Fase 2: Scaling**
```
Dataset: SVG-1M completo (1M samples)
Modelo: Llama-3-70B + QLoRA / GPT-4 fine-tune
Objetivo: Generación de alta calidad
Timeline: 1-2 meses
```

**Fase 3: Style Transfer**
```
Dataset: SVG-1M + L3D
Modelo: Hybrid (LLM + Diffusion)
Objetivo: Logos profesionales estilizados
Timeline: 2-3 meses
```

**Fase 4: Production**
```
Dataset: Custom curated + augmentation
Modelo: Ensemble + post-processing
Objetivo: Sistema comercial robusto
Timeline: 3-6 meses
```

---

## 6. Próximos Pasos

- [ ] Descargar SVG-1M desde HuggingFace
- [ ] Explorar estructura del dataset (análisis de distribución)
- [ ] Implementar pipeline de preprocessing para SVG
- [ ] Evaluar calidad de código SVG (validez, complejidad)
- [ ] Descargar subset de L3D para referencia visual
- [ ] Configurar métricas de evaluación (FID, CLIP-score)
- [ ] Implementar baseline: Fine-tuning Llama-3-8B en SVG-1M
- [ ] Benchmark contra modelos existentes (StarVector, SVGen)

---

**Contacto para datasets:**
- SVG-1M: GitHub issues en gitcat-404/SVGen
- L3D: https://lhf-labs.github.io/tm-dataset/
- QMUL-OpenLogo: Solicitud a autores (QMUL)
- FlickrLogos: request_flickrlogos@informatik.uni-augsburg.de

---

**Referencias adicionales:**
- Papers with Code - Logo Datasets: https://paperswithcode.com/
- Awesome SVG: https://github.com/willianjusten/awesome-svg
- Computer Vision Datasets: https://www.v7labs.com/open-datasets
