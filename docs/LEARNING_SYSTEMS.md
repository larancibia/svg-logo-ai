# LEARNING SYSTEMS - Sistemas de Aprendizaje Continuo para Generación de Logos con IA

**Fecha:** 2025-11-25
**Estado:** Research & Architecture Document
**Objetivo:** Diseñar un sistema que APRENDA de logos generados y mejore con el tiempo usando embeddings y feedback loops.

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema de Aprendizaje Continuo](#arquitectura-del-sistema-de-aprendizaje-continuo)
3. [Embedding-Based Learning](#embedding-based-learning)
4. [Feedback Loop Systems](#feedback-loop-systems)
5. [Vector Databases para Logos](#vector-databases-para-logos)
6. [Quality Prediction Models](#quality-prediction-models)
7. [Case Studies: Estrategias de Mejora](#case-studies-estrategias-de-mejora)
8. [Pipeline de Mejora Iterativa](#pipeline-de-mejora-iterativa)
9. [Implementación en Python](#implementación-en-python)
10. [Métricas de Mejora Real](#métricas-de-mejora-real)

---

## Resumen Ejecutivo

Los sistemas modernos de generación de imágenes con IA (Midjourney, DALL-E 3, Stable Diffusion) no se quedan estáticos: **aprenden continuamente de las interacciones humanas** para mejorar la calidad de sus outputs. Este documento presenta una arquitectura completa para implementar un sistema de aprendizaje continuo en generación de logos SVG con IA.

### Principios Clave

1. **Embeddings como Representación**: Usar CLIP y modelos multimodales para crear representaciones latentes de logos que capturen calidad, estilo e industria.

2. **Feedback Loops Activos**: Implementar sistemas de active learning donde ratings humanos guían la mejora del modelo.

3. **Vector Databases**: Usar ChromaDB/Pinecone para indexar logos por calidad y recuperar ejemplos relevantes (RAG).

4. **Predicción de Calidad**: Entrenar modelos que predicen la calidad ANTES de generar, ahorrando recursos.

5. **Mejora Iterativa**: Curriculum learning para empezar con prompts simples y progresar a diseños complejos.

---

## Arquitectura del Sistema de Aprendizaje Continuo

### Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    USUARIO / DISEÑADOR                          │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ Prompt + Preferencias              │ Ratings/Feedback
             ▼                                    ▼
┌─────────────────────────────┐    ┌──────────────────────────────┐
│   PROMPT OPTIMIZER          │    │   FEEDBACK COLLECTOR         │
│   (Bandit Algorithms)       │    │   (Human-in-the-Loop)        │
│                             │    │                              │
│ • Multi-armed bandit        │    │ • Rating system (1-5)        │
│ • Contextual bandits        │    │ • Pairwise comparisons       │
│ • A/B testing automático    │    │ • Feature preferences        │
└─────────────┬───────────────┘    └───────────┬──────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                               │
│                    (CLIP + Custom)                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Text Encoder │  │ Image Encoder│  │ Logo-Specific│         │
│  │   (CLIP)     │  │    (CLIP)    │  │   Features   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Output: 512-dim vector representation                          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VECTOR DATABASE (ChromaDB)                      │
│                                                                  │
│  Collections:                                                    │
│  • logos_generated    - Todos los logos generados               │
│  • logos_high_quality - Ratings > 4.0                           │
│  • logos_by_industry  - Indexados por industria                 │
│  • prompt_templates   - Prompts que funcionan bien              │
│                                                                  │
│  Indexing Strategy:                                              │
│  • HNSW (Hierarchical Navigable Small World)                    │
│  • Cosine similarity                                             │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              RAG-ENHANCED GENERATION PIPELINE                    │
│                                                                  │
│  1. Query similar high-quality logos (k=5)                      │
│  2. Extract successful patterns                                 │
│  3. Augment prompt with retrieved context                       │
│  4. Predict quality BEFORE generation                           │
│  5. Generate logo (if predicted quality > threshold)            │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOGO GENERATOR                                 │
│            (Flux Pro / Stable Diffusion 3.5)                    │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│               QUALITY PREDICTION MODEL                           │
│                                                                  │
│  • Bradley-Terry preference model                               │
│  • Elo rating system                                            │
│  • Multi-dimensional quality assessment                         │
│                                                                  │
│  Dimensions:                                                     │
│  1. Technical Quality (sharpness, composition)                  │
│  2. Aesthetic Appeal (beauty, harmony)                          │
│  3. Brand Fit (appropriateness for industry)                    │
│  4. Text-Image Correspondence (matches prompt)                  │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTINUOUS LEARNING LOOP                            │
│                                                                  │
│  1. Collect feedback (ratings, selections, rejections)          │
│  2. Update embeddings with human preferences                    │
│  3. Retrain quality predictor (weekly)                          │
│  4. Fine-tune prompt optimizer (daily)                          │
│  5. Update vector database with new high-quality examples       │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

#### 1. Prompt Optimizer (Bandit-Based)
- **Función**: Selecciona la mejor estrategia de prompt en tiempo real
- **Algoritmo**: Multi-armed bandit con upper confidence bound (UCB)
- **Entrada**: Contexto (industria, estilo deseado, complejidad)
- **Salida**: Template de prompt optimizado

#### 2. Embedding Layer
- **CLIP Encoder**: Representaciones text-image multimodales
- **Custom Features**: Métricas específicas de logos (simetría, balance, simplicidad)
- **Dimensión**: 512-dim vector

#### 3. Vector Database (ChromaDB)
- **Propósito**: Almacenar y recuperar logos similares
- **Indexing**: HNSW para búsqueda eficiente
- **Metadata**: ratings, industria, estilo, timestamp

#### 4. Quality Predictor
- **Modelo**: Bradley-Terry + Features de CLIP embeddings
- **Entrenamiento**: Pairwise comparisons de usuarios
- **Output**: Score 0-1 de calidad esperada

#### 5. Continuous Learning Loop
- **Frecuencia**: Actualización diaria de bandit, semanal de quality predictor
- **Datos**: Feedback acumulado + métricas de engagement

---

## Embedding-Based Learning

### CLIP Embeddings para Logos

**CLIP (Contrastive Language-Image Pre-training)** de OpenAI es el estándar actual para representaciones multimodales. Para logos, necesitamos adaptar CLIP con conocimiento específico del dominio.

#### Arquitectura de CLIP para Logos

```python
import torch
import clip
from PIL import Image

class LogoEmbedder:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_image(self, image_path):
        """Genera embedding de 512-dim para una imagen de logo"""
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def encode_text(self, text):
        """Genera embedding para descripción textual"""
        text_input = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()

    def similarity(self, image_path, text):
        """Calcula similitud coseno entre imagen y texto"""
        image_emb = self.encode_image(image_path)
        text_emb = self.encode_text(text)

        similarity = (image_emb @ text_emb.T).item()
        return similarity
```

### FashionLOGO: State-of-the-Art en Logo Embeddings (2024)

Investigación reciente (FashionLOGO, arXiv 2024) muestra cómo mejorar CLIP para logos:

**Key Innovations:**
1. **Cross-Attention Transformer**: Permite que embeddings visuales aprendan de conocimiento textual suplementario
2. **Multimodal LLM Integration**: Usa LLaVA para generar descripciones textuales ricas
3. **Logo-Specific Fine-tuning**: Entrenado en datasets de logos con metadatos de industria

```python
class EnhancedLogoEmbedder(LogoEmbedder):
    """Versión mejorada con features específicas de logos"""

    def extract_logo_features(self, image_path):
        """Extrae features adicionales específicas de logos"""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)

        features = {
            'symmetry': self._calculate_symmetry(img),
            'color_diversity': self._calculate_color_diversity(img),
            'edge_density': self._calculate_edge_density(img),
            'simplicity_score': self._calculate_simplicity(img),
            'contrast': self._calculate_contrast(img)
        }

        return features

    def _calculate_symmetry(self, img):
        """Calcula simetría vertical y horizontal"""
        h, w = img.shape[:2]
        left = img[:, :w//2]
        right = cv2.flip(img[:, w//2:], 1)

        if left.shape != right.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        diff = cv2.absdiff(left, right)
        symmetry_score = 1 - (diff.mean() / 255.0)

        return symmetry_score

    def _calculate_color_diversity(self, img):
        """Número de colores únicos (simplificado)"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))

        # Normalizar (logos simples = 2-10 colores)
        return min(unique_colors / 20.0, 1.0)

    def _calculate_edge_density(self, img):
        """Densidad de bordes (logos simples tienen menos)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = edges.sum() / (img.shape[0] * img.shape[1] * 255)

        return density

    def _calculate_simplicity(self, img):
        """Score de simplicidad (inverso de complejidad)"""
        # Logos simples: pocos colores + baja densidad de bordes
        colors = self._calculate_color_diversity(img)
        edges = self._calculate_edge_density(img)

        simplicity = 1 - ((colors + edges) / 2)
        return simplicity

    def _calculate_contrast(self, img):
        """Contraste de la imagen"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.std() / 128.0

    def get_combined_embedding(self, image_path, text_description):
        """Combina CLIP embedding + features específicas"""
        # CLIP embeddings
        clip_emb = self.encode_image(image_path)

        # Logo-specific features
        logo_features = self.extract_logo_features(image_path)
        logo_vec = np.array(list(logo_features.values()))

        # Concatenar (512 + 5 = 517 dimensiones)
        combined = np.concatenate([clip_emb.flatten(), logo_vec])

        return combined, logo_features
```

### Similarity Search en Espacio de Embeddings

```python
class LogoSimilaritySearch:
    """Búsqueda de logos similares en espacio de embeddings"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.embeddings_db = []
        self.metadata_db = []

    def add_logo(self, image_path, metadata):
        """Añade logo a la base de datos"""
        embedding, features = self.embedder.get_combined_embedding(
            image_path,
            metadata.get('description', '')
        )

        self.embeddings_db.append(embedding)
        self.metadata_db.append({
            **metadata,
            **features,
            'image_path': image_path
        })

    def find_similar(self, query_path, k=5, filters=None):
        """Encuentra k logos más similares"""
        query_emb, _ = self.embedder.get_combined_embedding(query_path, "")

        # Calcular similitudes
        similarities = []
        for i, emb in enumerate(self.embeddings_db):
            # Aplicar filtros
            if filters and not self._match_filters(self.metadata_db[i], filters):
                continue

            # Cosine similarity
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((sim, i))

        # Ordenar por similitud
        similarities.sort(reverse=True)

        # Retornar top-k
        results = []
        for sim, idx in similarities[:k]:
            results.append({
                'similarity': sim,
                'metadata': self.metadata_db[idx]
            })

        return results

    def _match_filters(self, metadata, filters):
        """Verifica si metadata cumple con filtros"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
```

### Uso de Embeddings para Guiar Generación

**Estrategia: Retrieval-Augmented Generation (RAG)**

En lugar de generar logos desde cero, recuperamos ejemplos exitosos y los usamos para augmentar el prompt:

```python
class RAGLogoGenerator:
    """Generador de logos con RAG"""

    def __init__(self, embedder, similarity_search, generator):
        self.embedder = embedder
        self.search = similarity_search
        self.generator = generator

    def generate_with_rag(self, prompt, industry, style, k=3):
        """Genera logo usando ejemplos similares"""

        # 1. Buscar logos similares exitosos
        filters = {
            'industry': industry,
            'rating': lambda x: x >= 4.0  # Solo alta calidad
        }

        # Crear query embedding desde el prompt
        similar_logos = self.search.find_similar_by_text(
            prompt,
            k=k,
            filters=filters
        )

        # 2. Extraer patrones exitosos
        patterns = self._extract_patterns(similar_logos)

        # 3. Augmentar prompt con contexto
        augmented_prompt = self._augment_prompt(prompt, patterns)

        # 4. Generar logo
        logo = self.generator.generate(augmented_prompt)

        return logo, augmented_prompt, similar_logos

    def _extract_patterns(self, similar_logos):
        """Extrae patrones comunes de logos exitosos"""
        patterns = {
            'avg_symmetry': np.mean([l['metadata']['symmetry']
                                     for l in similar_logos]),
            'avg_colors': np.mean([l['metadata']['color_diversity']
                                   for l in similar_logos]),
            'avg_simplicity': np.mean([l['metadata']['simplicity_score']
                                       for l in similar_logos]),
            'common_styles': self._get_common_values(
                [l['metadata'].get('style', '') for l in similar_logos]
            )
        }

        return patterns

    def _augment_prompt(self, original_prompt, patterns):
        """Aumenta prompt con información de patrones"""
        augmentation = []

        if patterns['avg_symmetry'] > 0.7:
            augmentation.append("highly symmetric design")

        if patterns['avg_simplicity'] > 0.6:
            augmentation.append("minimalist, clean, simple")

        if patterns['avg_colors'] < 0.3:
            augmentation.append("limited color palette (2-4 colors)")

        if patterns['common_styles']:
            augmentation.append(f"style: {patterns['common_styles'][0]}")

        # Combinar
        if augmentation:
            return f"{original_prompt}, {', '.join(augmentation)}"
        return original_prompt

    def _get_common_values(self, values, top_n=3):
        """Obtiene valores más comunes"""
        from collections import Counter
        counter = Counter([v for v in values if v])
        return [item for item, count in counter.most_common(top_n)]
```

---

## Feedback Loop Systems

### Active Learning para Diseño

**Active Learning** es una técnica donde el modelo elige qué ejemplos son más informativos para su entrenamiento. En generación de logos:

1. El sistema genera múltiples variantes
2. Predice cuáles son más "inciertos" o "informativos"
3. Solicita feedback humano solo para esos
4. Aprende más rápido con menos datos etiquetados

#### Implementación de Active Learning

```python
class ActiveLogoLearner:
    """Sistema de aprendizaje activo para logos"""

    def __init__(self, quality_predictor, uncertainty_threshold=0.3):
        self.quality_predictor = quality_predictor
        self.uncertainty_threshold = uncertainty_threshold
        self.pending_feedback = []

    def generate_batch(self, prompts, n_variants=4):
        """Genera múltiples variantes y selecciona las más informativas"""

        all_logos = []

        for prompt in prompts:
            variants = []
            for i in range(n_variants):
                logo = self.generate_variant(prompt, seed=i)

                # Predecir calidad y calcular incertidumbre
                quality_pred, uncertainty = self.quality_predictor.predict_with_uncertainty(logo)

                variants.append({
                    'logo': logo,
                    'prompt': prompt,
                    'quality_pred': quality_pred,
                    'uncertainty': uncertainty,
                    'variant_id': i
                })

            all_logos.extend(variants)

        # Ordenar por incertidumbre (los más inciertos primero)
        all_logos.sort(key=lambda x: x['uncertainty'], reverse=True)

        # Seleccionar top-k para feedback humano
        need_feedback = [
            logo for logo in all_logos
            if logo['uncertainty'] > self.uncertainty_threshold
        ]

        auto_accept = [
            logo for logo in all_logos
            if logo['uncertainty'] <= self.uncertainty_threshold
            and logo['quality_pred'] > 0.7
        ]

        return {
            'need_feedback': need_feedback,
            'auto_accept': auto_accept,
            'stats': {
                'total_generated': len(all_logos),
                'need_human_review': len(need_feedback),
                'auto_accepted': len(auto_accept)
            }
        }

    def collect_feedback(self, logo_id, rating, comments=None):
        """Recolecta feedback humano"""
        feedback = {
            'logo_id': logo_id,
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now()
        }

        self.pending_feedback.append(feedback)

        # Si acumulamos suficiente feedback, reentrenar
        if len(self.pending_feedback) >= 50:
            self.retrain()

    def retrain(self):
        """Reentrena el modelo con nuevo feedback"""
        print(f"Reentrenando con {len(self.pending_feedback)} ejemplos nuevos...")

        # Aquí iría la lógica de reentrenamiento
        # Por ahora, solo simulamos

        self.pending_feedback = []
        print("Modelo actualizado!")
```

### Human-in-the-Loop Feedback System

```python
class FeedbackCollector:
    """Sistema de recolección de feedback humano"""

    def __init__(self, storage_backend='chromadb'):
        self.storage = storage_backend
        self.feedback_history = []

    def collect_rating(self, logo_id, user_id, rating, dimensions=None):
        """
        Recolecta rating multi-dimensional

        dimensions puede incluir:
        - technical_quality: 1-5
        - aesthetic_appeal: 1-5
        - brand_fit: 1-5
        - text_correspondence: 1-5
        """

        feedback = {
            'logo_id': logo_id,
            'user_id': user_id,
            'overall_rating': rating,
            'dimensions': dimensions or {},
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_history.append(feedback)
        self._update_logo_score(logo_id, rating, dimensions)

        return feedback

    def collect_pairwise_comparison(self, logo_a_id, logo_b_id,
                                   winner, user_id):
        """
        Recolecta comparación por pares (para Bradley-Terry)

        Args:
            winner: 'a', 'b', or 'tie'
        """

        comparison = {
            'logo_a': logo_a_id,
            'logo_b': logo_b_id,
            'winner': winner,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_history.append(comparison)
        self._update_pairwise_scores(logo_a_id, logo_b_id, winner)

        return comparison

    def collect_implicit_feedback(self, logo_id, action, user_id):
        """
        Recolecta feedback implícito

        Actions:
        - 'download': +2 puntos
        - 'share': +1.5 puntos
        - 'save': +1 punto
        - 'skip': -0.5 puntos
        - 'report': -2 puntos
        """

        action_weights = {
            'download': 2.0,
            'share': 1.5,
            'save': 1.0,
            'skip': -0.5,
            'report': -2.0
        }

        weight = action_weights.get(action, 0)

        feedback = {
            'logo_id': logo_id,
            'user_id': user_id,
            'action': action,
            'implicit_score': weight,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_history.append(feedback)
        self._update_logo_score(logo_id, weight, is_implicit=True)

        return feedback

    def _update_logo_score(self, logo_id, score, dimensions=None,
                          is_implicit=False):
        """Actualiza score agregado del logo"""
        # Implementación específica del backend de almacenamiento
        pass

    def get_logo_statistics(self, logo_id):
        """Obtiene estadísticas agregadas de un logo"""
        logo_feedbacks = [
            f for f in self.feedback_history
            if f.get('logo_id') == logo_id
        ]

        if not logo_feedbacks:
            return None

        ratings = [f['overall_rating'] for f in logo_feedbacks
                  if 'overall_rating' in f]

        stats = {
            'avg_rating': np.mean(ratings) if ratings else None,
            'num_ratings': len(ratings),
            'num_comparisons': len([f for f in logo_feedbacks
                                   if 'winner' in f]),
            'num_implicit': len([f for f in logo_feedbacks
                                if 'action' in f]),
            'total_feedback': len(logo_feedbacks)
        }

        return stats
```

### Bandit Algorithms para A/B Testing de Prompts

**Multi-Armed Bandit (MAB)** es superior a A/B testing tradicional porque:
- **Minimiza "regret"**: No desperdicia tiempo en variantes malas
- **Exploración dinámica**: Balancea exploración vs explotación
- **Convergencia más rápida**: Encuentra la mejor variante más rápido

#### Implementación de Thompson Sampling

```python
import numpy as np
from scipy.stats import beta

class ThompsonSamplingBandit:
    """
    Thompson Sampling para selección de prompt templates

    Cada "arm" es un template de prompt diferente.
    Reward es el rating del usuario (0-1 normalizado).
    """

    def __init__(self, n_arms, arm_names=None):
        self.n_arms = n_arms
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]

        # Prior: Beta(1, 1) = Uniform
        self.alpha = np.ones(n_arms)  # Successes
        self.beta_params = np.ones(n_arms)  # Failures

        self.total_pulls = np.zeros(n_arms)
        self.total_reward = np.zeros(n_arms)

    def select_arm(self):
        """Selecciona arm usando Thompson Sampling"""

        # Sample de cada distribución Beta
        samples = [
            np.random.beta(self.alpha[i], self.beta_params[i])
            for i in range(self.n_arms)
        ]

        # Seleccionar el arm con mayor sample
        selected_arm = np.argmax(samples)

        return selected_arm

    def update(self, arm, reward):
        """
        Actualiza creencias después de observar reward

        Args:
            arm: índice del arm seleccionado
            reward: valor 0-1 (rating normalizado)
        """

        self.total_pulls[arm] += 1
        self.total_reward[arm] += reward

        # Actualizar parámetros Beta
        # Reward = 1 → success, Reward = 0 → failure
        self.alpha[arm] += reward
        self.beta_params[arm] += (1 - reward)

    def get_arm_statistics(self):
        """Obtiene estadísticas de cada arm"""

        stats = []
        for i in range(self.n_arms):
            if self.total_pulls[i] > 0:
                mean_reward = self.total_reward[i] / self.total_pulls[i]
            else:
                mean_reward = 0.5  # Prior mean

            # Intervalo de credibilidad 95%
            lower, upper = beta.ppf(
                [0.025, 0.975],
                self.alpha[i],
                self.beta_params[i]
            )

            stats.append({
                'arm': i,
                'name': self.arm_names[i],
                'pulls': int(self.total_pulls[i]),
                'mean_reward': mean_reward,
                'credible_interval': (lower, upper),
                'alpha': self.alpha[i],
                'beta': self.beta_params[i]
            })

        return stats

    def get_best_arm(self):
        """Retorna el arm con mayor reward esperado"""
        expected_rewards = self.alpha / (self.alpha + self.beta_params)
        best_arm = np.argmax(expected_rewards)

        return best_arm, self.arm_names[best_arm]

# Ejemplo de uso
class PromptOptimizer:
    """Optimizador de prompts usando bandits"""

    def __init__(self, prompt_templates):
        self.templates = prompt_templates
        self.bandit = ThompsonSamplingBandit(
            n_arms=len(prompt_templates),
            arm_names=[t['name'] for t in prompt_templates]
        )

    def select_prompt_template(self, context):
        """Selecciona el mejor template para el contexto dado"""

        arm = self.bandit.select_arm()
        template = self.templates[arm]

        return arm, template

    def update_with_feedback(self, arm, rating):
        """Actualiza bandit con feedback del usuario"""

        # Normalizar rating (1-5) a (0-1)
        normalized_reward = (rating - 1) / 4

        self.bandit.update(arm, normalized_reward)

    def get_best_template(self):
        """Retorna el template con mejor performance"""

        best_arm, best_name = self.bandit.get_best_arm()
        return self.templates[best_arm]

    def get_performance_report(self):
        """Genera reporte de performance de templates"""

        stats = self.bandit.get_arm_statistics()

        # Ordenar por mean_reward
        stats.sort(key=lambda x: x['mean_reward'], reverse=True)

        return stats
```

#### Ejemplo de Uso Completo

```python
# Definir templates de prompts
prompt_templates = [
    {
        'name': 'minimalist',
        'template': 'minimalist {industry} logo, simple geometric shapes, {colors}, vector art, clean design'
    },
    {
        'name': 'modern_gradient',
        'template': 'modern {industry} logo with gradient colors, sleek design, {colors}, professional'
    },
    {
        'name': 'vintage',
        'template': 'vintage {industry} logo, retro style, {colors}, badge design, classic'
    },
    {
        'name': 'abstract',
        'template': 'abstract {industry} logo, creative shapes, {colors}, unique, artistic'
    }
]

# Inicializar optimizador
optimizer = PromptOptimizer(prompt_templates)

# Simular uso
for iteration in range(100):
    # Usuario solicita logo
    context = {'industry': 'tech', 'colors': 'blue and white'}

    # Seleccionar template
    arm, template = optimizer.select_prompt_template(context)

    # Generar logo (simulated)
    # logo = generate_logo(template, context)

    # Usuario da rating (simulated)
    rating = np.random.randint(1, 6)  # En producción, esto viene del usuario

    # Actualizar bandit
    optimizer.update_with_feedback(arm, rating)

# Ver resultados
print("Performance Report:")
for stat in optimizer.get_performance_report():
    print(f"{stat['name']}: {stat['mean_reward']:.3f} "
          f"({stat['pulls']} pulls) "
          f"95% CI: [{stat['credible_interval'][0]:.3f}, "
          f"{stat['credible_interval'][1]:.3f}]")

# Mejor template
best = optimizer.get_best_template()
print(f"\nBest template: {best['name']}")
print(f"Template: {best['template']}")
```

### Curriculum Learning

**Curriculum Learning** entrena modelos empezando con ejemplos simples y progresando a complejos, imitando cómo aprenden los humanos.

Para generación de logos:
1. **Nivel 1 (Simple)**: Logos monocromáticos, formas geométricas básicas
2. **Nivel 2 (Intermedio)**: 2-3 colores, tipografía simple
3. **Nivel 3 (Avanzado)**: Gradientes, efectos, composiciones complejas
4. **Nivel 4 (Experto)**: Logos con ilustraciones detalladas, efectos 3D

```python
class CurriculumManager:
    """Gestor de curriculum learning para logos"""

    def __init__(self):
        self.current_level = 1
        self.level_thresholds = {
            1: {'accuracy': 0.7, 'min_samples': 50},
            2: {'accuracy': 0.75, 'min_samples': 100},
            3: {'accuracy': 0.8, 'min_samples': 150},
            4: {'accuracy': 0.85, 'min_samples': 200}
        }

        self.level_characteristics = {
            1: {
                'name': 'Simple',
                'constraints': {
                    'max_colors': 1,
                    'shapes': ['circle', 'square', 'triangle'],
                    'complexity': 'low',
                    'effects': []
                }
            },
            2: {
                'name': 'Intermediate',
                'constraints': {
                    'max_colors': 3,
                    'shapes': ['circle', 'square', 'triangle', 'polygon'],
                    'complexity': 'medium',
                    'effects': ['solid_fill']
                }
            },
            3: {
                'name': 'Advanced',
                'constraints': {
                    'max_colors': 5,
                    'shapes': 'any',
                    'complexity': 'high',
                    'effects': ['gradient', 'shadow']
                }
            },
            4: {
                'name': 'Expert',
                'constraints': {
                    'max_colors': 'unlimited',
                    'shapes': 'any',
                    'complexity': 'very_high',
                    'effects': ['gradient', 'shadow', '3d', 'texture']
                }
            }
        }

        self.performance_history = []

    def get_current_constraints(self):
        """Retorna constraints del nivel actual"""
        return self.level_characteristics[self.current_level]['constraints']

    def record_performance(self, accuracy, n_samples):
        """Registra performance y decide si subir de nivel"""

        self.performance_history.append({
            'level': self.current_level,
            'accuracy': accuracy,
            'n_samples': n_samples,
            'timestamp': datetime.now()
        })

        # Verificar si podemos avanzar al siguiente nivel
        if self.current_level < 4:
            threshold = self.level_thresholds[self.current_level]

            if (accuracy >= threshold['accuracy'] and
                n_samples >= threshold['min_samples']):

                self.current_level += 1
                print(f"✓ Advanced to Level {self.current_level}: "
                      f"{self.level_characteristics[self.current_level]['name']}")

                return True

        return False

    def generate_curriculum_prompt(self, base_prompt):
        """Adapta prompt según nivel de curriculum"""

        constraints = self.get_current_constraints()

        adaptations = []

        if constraints['max_colors'] != 'unlimited':
            adaptations.append(f"using maximum {constraints['max_colors']} colors")

        if constraints['complexity'] == 'low':
            adaptations.append("simple geometric design")
        elif constraints['complexity'] == 'medium':
            adaptations.append("moderately detailed design")

        if constraints['effects']:
            adaptations.append(f"with {', '.join(constraints['effects'])}")

        if adaptations:
            adapted_prompt = f"{base_prompt}, {', '.join(adaptations)}"
        else:
            adapted_prompt = base_prompt

        return adapted_prompt
```

---

## Vector Databases para Logos

### ChromaDB: Setup y Configuración

**ChromaDB** es ideal para prototipos y sistemas de RAG. Es open-source, fácil de usar, y se integra perfectamente con embeddings.

#### Instalación y Setup

```bash
pip install chromadb
```

```python
import chromadb
from chromadb.config import Settings

class LogoVectorDatabase:
    """Vector database para logos usando ChromaDB"""

    def __init__(self, persist_directory="./chroma_db"):
        # Cliente persistente
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

        # Crear colecciones
        self.logos_collection = self.client.get_or_create_collection(
            name="logos_generated",
            metadata={"description": "All generated logos"}
        )

        self.high_quality_collection = self.client.get_or_create_collection(
            name="logos_high_quality",
            metadata={"description": "High quality logos (rating >= 4.0)"}
        )

    def add_logo(self, logo_id, embedding, metadata, image_path=None):
        """Añade logo a la base de datos"""

        # Añadir a colección principal
        self.logos_collection.add(
            ids=[logo_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

        # Si es alta calidad, añadir también a high_quality
        if metadata.get('rating', 0) >= 4.0:
            self.high_quality_collection.add(
                ids=[logo_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )

        print(f"Added logo {logo_id} to database")

    def query_similar(self, query_embedding, n_results=5,
                     collection='all', filters=None):
        """
        Busca logos similares

        Args:
            query_embedding: embedding del query
            n_results: número de resultados
            collection: 'all' o 'high_quality'
            filters: dict con filtros de metadata
        """

        # Seleccionar colección
        if collection == 'high_quality':
            coll = self.high_quality_collection
        else:
            coll = self.logos_collection

        # Query
        results = coll.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filters
        )

        return results

    def update_logo_rating(self, logo_id, new_rating):
        """Actualiza rating de un logo"""

        # Obtener metadata actual
        result = self.logos_collection.get(ids=[logo_id])

        if result['ids']:
            metadata = result['metadatas'][0]
            metadata['rating'] = new_rating

            # Actualizar en colección principal
            self.logos_collection.update(
                ids=[logo_id],
                metadatas=[metadata]
            )

            # Si ahora es alta calidad, añadir a high_quality
            if new_rating >= 4.0:
                embedding = result['embeddings'][0]
                self.high_quality_collection.add(
                    ids=[logo_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )

            print(f"Updated logo {logo_id} rating to {new_rating}")

    def get_statistics(self):
        """Obtiene estadísticas de la base de datos"""

        total_logos = self.logos_collection.count()
        high_quality_logos = self.high_quality_collection.count()

        stats = {
            'total_logos': total_logos,
            'high_quality_logos': high_quality_logos,
            'quality_ratio': high_quality_logos / total_logos if total_logos > 0 else 0
        }

        return stats
```

### Indexación por Calidad, Estilo, Industria

```python
class AdvancedLogoDatabase(LogoVectorDatabase):
    """Base de datos avanzada con múltiples índices"""

    def __init__(self, persist_directory="./chroma_db"):
        super().__init__(persist_directory)

        # Colecciones especializadas por industria
        self.industry_collections = {}

        # Colecciones por estilo
        self.style_collections = {}

    def add_logo_with_indexing(self, logo_id, embedding, metadata):
        """Añade logo con indexación avanzada"""

        # Añadir a colección principal
        self.add_logo(logo_id, embedding, metadata)

        # Indexar por industria
        industry = metadata.get('industry')
        if industry:
            self._add_to_industry_collection(
                industry, logo_id, embedding, metadata
            )

        # Indexar por estilo
        style = metadata.get('style')
        if style:
            self._add_to_style_collection(
                style, logo_id, embedding, metadata
            )

    def _add_to_industry_collection(self, industry, logo_id,
                                    embedding, metadata):
        """Añade a colección específica de industria"""

        collection_name = f"logos_industry_{industry.lower()}"

        if collection_name not in self.industry_collections:
            self.industry_collections[collection_name] = \
                self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"industry": industry}
                )

        self.industry_collections[collection_name].add(
            ids=[logo_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def _add_to_style_collection(self, style, logo_id,
                                embedding, metadata):
        """Añade a colección específica de estilo"""

        collection_name = f"logos_style_{style.lower()}"

        if collection_name not in self.style_collections:
            self.style_collections[collection_name] = \
                self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"style": style}
                )

        self.style_collections[collection_name].add(
            ids=[logo_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def query_by_industry(self, industry, query_embedding, n_results=5):
        """Query específico para una industria"""

        collection_name = f"logos_industry_{industry.lower()}"

        if collection_name in self.industry_collections:
            return self.industry_collections[collection_name].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )

        return None

    def get_industry_statistics(self, industry):
        """Obtiene estadísticas de una industria específica"""

        collection_name = f"logos_industry_{industry.lower()}"

        if collection_name in self.industry_collections:
            count = self.industry_collections[collection_name].count()

            # Obtener ratings promedio
            # (requeriría query de todos los items)

            return {
                'industry': industry,
                'total_logos': count
            }

        return None
```

### Retrieval-Augmented Generation (RAG) para Logos

Ya implementamos una versión básica arriba. Aquí una versión más avanzada:

```python
class AdvancedRAGSystem:
    """Sistema RAG avanzado para generación de logos"""

    def __init__(self, vector_db, embedder, quality_predictor):
        self.db = vector_db
        self.embedder = embedder
        self.quality_predictor = quality_predictor

    def generate_with_rag(self, prompt, industry, style_preference,
                         n_examples=5, quality_threshold=4.0):
        """
        Generación RAG avanzada

        Pipeline:
        1. Retrieve: Buscar ejemplos similares de alta calidad
        2. Analyze: Extraer patrones exitosos
        3. Augment: Enriquecer prompt con patrones
        4. Predict: Estimar calidad antes de generar
        5. Generate: Solo si predicción es buena
        6. Verify: Verificar calidad post-generación
        """

        # 1. RETRIEVE: Buscar ejemplos similares
        query_embedding = self.embedder.encode_text(prompt)

        filters = {
            'industry': industry,
            'rating': {'$gte': quality_threshold}
        }

        similar_logos = self.db.query_similar(
            query_embedding,
            n_results=n_examples,
            collection='high_quality',
            filters=filters
        )

        # 2. ANALYZE: Extraer patrones
        patterns = self._analyze_patterns(similar_logos)

        # 3. AUGMENT: Enriquecer prompt
        augmented_prompt = self._smart_augment(
            prompt, patterns, style_preference
        )

        # 4. PREDICT: Estimar calidad
        predicted_quality = self.quality_predictor.predict_from_prompt(
            augmented_prompt
        )

        if predicted_quality < quality_threshold:
            # Intentar mejorar el prompt
            augmented_prompt = self._improve_prompt(
                augmented_prompt, patterns, predicted_quality
            )

            predicted_quality = self.quality_predictor.predict_from_prompt(
                augmented_prompt
            )

        # 5. GENERATE
        generation_result = {
            'augmented_prompt': augmented_prompt,
            'predicted_quality': predicted_quality,
            'similar_examples': similar_logos,
            'patterns_used': patterns,
            'should_generate': predicted_quality >= quality_threshold - 0.5
        }

        return generation_result

    def _analyze_patterns(self, similar_logos):
        """Análisis profundo de patrones"""

        if not similar_logos['ids']:
            return {}

        metadatas = similar_logos['metadatas'][0]

        patterns = {
            'visual_features': {
                'avg_symmetry': np.mean([m.get('symmetry', 0.5)
                                        for m in metadatas]),
                'avg_simplicity': np.mean([m.get('simplicity_score', 0.5)
                                          for m in metadatas]),
                'avg_colors': np.mean([m.get('color_diversity', 0.3)
                                      for m in metadatas]),
                'avg_contrast': np.mean([m.get('contrast', 0.5)
                                        for m in metadatas])
            },
            'common_styles': self._extract_common_values(
                [m.get('style', '') for m in metadatas]
            ),
            'common_keywords': self._extract_keywords_from_prompts(
                [m.get('prompt', '') for m in metadatas]
            ),
            'avg_rating': np.mean([m.get('rating', 3.0)
                                  for m in metadatas])
        }

        return patterns

    def _smart_augment(self, base_prompt, patterns, style_preference):
        """Augmentación inteligente del prompt"""

        augmentations = []

        # Visual features
        vf = patterns.get('visual_features', {})

        if vf.get('avg_symmetry', 0) > 0.7:
            augmentations.append("symmetric balanced composition")

        if vf.get('avg_simplicity', 0) > 0.6:
            augmentations.append("clean minimalist design")

        if vf.get('avg_contrast', 0) > 0.6:
            augmentations.append("high contrast")

        # Style
        if patterns.get('common_styles'):
            top_style = patterns['common_styles'][0]
            if top_style and top_style == style_preference:
                augmentations.append(f"{top_style} style")

        # Keywords from successful prompts
        if patterns.get('common_keywords'):
            top_keywords = patterns['common_keywords'][:3]
            augmentations.extend(top_keywords)

        # Combine
        if augmentations:
            return f"{base_prompt}, {', '.join(augmentations)}"

        return base_prompt

    def _improve_prompt(self, prompt, patterns, current_quality):
        """Mejora prompt si la predicción es baja"""

        improvements = []

        if current_quality < 0.5:
            # Calidad muy baja, agregar elementos fuertes
            improvements.append("professional design")
            improvements.append("high quality")
            improvements.append("award winning")

        # Agregar características de ejemplos exitosos
        if patterns.get('avg_rating', 0) > 4.5:
            improvements.append("premium quality")

        if improvements:
            return f"{prompt}, {', '.join(improvements)}"

        return prompt

    def _extract_common_values(self, values, top_n=3):
        """Extrae valores más comunes"""
        from collections import Counter
        counter = Counter([v for v in values if v])
        return [item for item, count in counter.most_common(top_n)]

    def _extract_keywords_from_prompts(self, prompts):
        """Extrae keywords comunes de prompts exitosos"""

        # Palabras a ignorar
        stopwords = {'a', 'an', 'the', 'and', 'or', 'for', 'with', 'logo'}

        all_words = []
        for prompt in prompts:
            words = prompt.lower().split()
            filtered = [w for w in words if w not in stopwords and len(w) > 3]
            all_words.extend(filtered)

        from collections import Counter
        common = Counter(all_words).most_common(5)

        return [word for word, count in common if count > 1]
```

### Few-Shot Selection Automática

```python
class FewShotSelector:
    """Selector automático de ejemplos para few-shot learning"""

    def __init__(self, vector_db, embedder):
        self.db = vector_db
        self.embedder = embedder

    def select_diverse_examples(self, query, n_examples=5,
                               diversity_weight=0.3):
        """
        Selecciona ejemplos balanceando similitud y diversidad

        Estrategia:
        1. Retrieve top-k*2 candidatos por similitud
        2. Re-rank considerando diversidad
        3. Seleccionar top-n que maximicen similitud + diversidad
        """

        # Query embedding
        query_emb = self.embedder.encode_text(query)

        # Retrieve candidatos (2x más de lo necesario)
        candidates = self.db.query_similar(
            query_emb,
            n_results=n_examples * 2,
            collection='high_quality'
        )

        if not candidates['ids']:
            return []

        # Extraer embeddings de candidatos
        candidate_embeddings = candidates['embeddings'][0]
        candidate_ids = candidates['ids'][0]
        candidate_metadatas = candidates['metadatas'][0]

        # Seleccionar con diversidad
        selected_indices = self._select_with_diversity(
            query_emb,
            candidate_embeddings,
            n_examples,
            diversity_weight
        )

        # Construir resultado
        selected_examples = []
        for idx in selected_indices:
            selected_examples.append({
                'id': candidate_ids[idx],
                'metadata': candidate_metadatas[idx],
                'embedding': candidate_embeddings[idx]
            })

        return selected_examples

    def _select_with_diversity(self, query_emb, candidate_embs,
                              n_select, diversity_weight):
        """
        Greedy selection maximizando similitud + diversidad

        Algoritmo:
        1. Seleccionar el más similar al query
        2. Iterar: seleccionar candidato que maximice:
           score = (1-λ)*sim(query, cand) + λ*min_dist(cand, selected)
        """

        selected = []
        remaining = list(range(len(candidate_embs)))

        # 1. Seleccionar el más similar
        similarities = [
            self._cosine_similarity(query_emb, emb)
            for emb in candidate_embs
        ]

        best_idx = np.argmax(similarities)
        selected.append(best_idx)
        remaining.remove(best_idx)

        # 2. Seleccionar resto con diversidad
        while len(selected) < n_select and remaining:
            scores = []

            for idx in remaining:
                # Similitud al query
                query_sim = similarities[idx]

                # Distancia mínima a ya seleccionados
                min_dist = min([
                    1 - self._cosine_similarity(
                        candidate_embs[idx],
                        candidate_embs[sel_idx]
                    )
                    for sel_idx in selected
                ])

                # Score combinado
                score = (1 - diversity_weight) * query_sim + \
                       diversity_weight * min_dist

                scores.append(score)

            # Seleccionar mejor score
            best_remaining_idx = remaining[np.argmax(scores)]
            selected.append(best_remaining_idx)
            remaining.remove(best_remaining_idx)

        return selected

    def _cosine_similarity(self, emb1, emb2):
        """Calcula similitud coseno"""
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)

        return np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
```

---

## Quality Prediction Models

### Bradley-Terry Preference Model

**Bradley-Terry** es un modelo probabilístico para comparaciones por pares. Es usado por Chatbot Arena (LMSYS) para rankear LLMs.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

class BradleyTerryModel:
    """
    Bradley-Terry model para ranking de logos

    Modelo: P(logo_i beats logo_j) = σ(θ_i - θ_j)
    donde σ es la función sigmoide y θ son los strength parameters
    """

    def __init__(self, n_items):
        self.n_items = n_items
        self.theta = np.zeros(n_items)  # Strength parameters
        self.is_fitted = False

    def fit(self, comparisons):
        """
        Fit model from pairwise comparisons

        Args:
            comparisons: list of (i, j, result) where
                         i, j are item indices
                         result = 1 if i wins, 0 if j wins, 0.5 if tie
        """

        # Maximum likelihood estimation
        def neg_log_likelihood(theta):
            """Negative log-likelihood to minimize"""
            ll = 0

            for i, j, result in comparisons:
                # P(i beats j) = σ(θ_i - θ_j)
                p_i_beats_j = expit(theta[i] - theta[j])

                # Log-likelihood contribution
                if result == 1:  # i wins
                    ll += np.log(p_i_beats_j + 1e-10)
                elif result == 0:  # j wins
                    ll += np.log(1 - p_i_beats_j + 1e-10)
                else:  # tie
                    ll += np.log(0.5)

            return -ll

        # Optimize
        # Constraint: sum of theta = 0 (for identifiability)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)}

        result = minimize(
            neg_log_likelihood,
            x0=np.zeros(self.n_items),
            constraints=constraints,
            method='SLSQP'
        )

        self.theta = result.x
        self.is_fitted = True

        return self

    def predict_prob(self, i, j):
        """Predict probability that item i beats item j"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        return expit(self.theta[i] - self.theta[j])

    def get_rankings(self):
        """Get items ranked by strength"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Ordenar por theta (mayor = más fuerte)
        rankings = np.argsort(-self.theta)

        return rankings, self.theta[rankings]

    def get_item_score(self, i):
        """Get strength score for item i"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        return self.theta[i]


class LogoRankingSystem:
    """Sistema de ranking de logos usando Bradley-Terry"""

    def __init__(self):
        self.logo_ids = []
        self.logo_id_to_index = {}
        self.comparisons = []
        self.model = None

    def add_logo(self, logo_id):
        """Añade nuevo logo al sistema"""
        if logo_id not in self.logo_id_to_index:
            idx = len(self.logo_ids)
            self.logo_ids.append(logo_id)
            self.logo_id_to_index[logo_id] = idx

    def add_comparison(self, logo_a_id, logo_b_id, winner):
        """
        Añade comparación

        Args:
            winner: 'a', 'b', or 'tie'
        """

        # Asegurar que logos existen
        self.add_logo(logo_a_id)
        self.add_logo(logo_b_id)

        # Convertir a índices
        idx_a = self.logo_id_to_index[logo_a_id]
        idx_b = self.logo_id_to_index[logo_b_id]

        # Convertir winner a resultado
        if winner == 'a':
            result = 1
        elif winner == 'b':
            result = 0
        else:  # tie
            result = 0.5

        self.comparisons.append((idx_a, idx_b, result))

    def fit_model(self):
        """Entrena modelo Bradley-Terry"""

        if len(self.comparisons) < 10:
            print("Warning: Too few comparisons for reliable estimates")

        self.model = BradleyTerryModel(len(self.logo_ids))
        self.model.fit(self.comparisons)

        print(f"Fitted Bradley-Terry model on {len(self.comparisons)} comparisons")

    def get_rankings(self):
        """Obtiene ranking de logos"""

        if self.model is None:
            raise ValueError("Model not fitted yet")

        indices, scores = self.model.get_rankings()

        rankings = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            rankings.append({
                'rank': rank,
                'logo_id': self.logo_ids[idx],
                'score': score
            })

        return rankings

    def predict_winner(self, logo_a_id, logo_b_id):
        """Predice quién ganaría entre dos logos"""

        if self.model is None:
            raise ValueError("Model not fitted yet")

        idx_a = self.logo_id_to_index[logo_a_id]
        idx_b = self.logo_id_to_index[logo_b_id]

        prob_a_wins = self.model.predict_prob(idx_a, idx_b)

        return {
            'logo_a': logo_a_id,
            'logo_b': logo_b_id,
            'prob_a_wins': prob_a_wins,
            'prob_b_wins': 1 - prob_a_wins,
            'predicted_winner': logo_a_id if prob_a_wins > 0.5 else logo_b_id
        }
```

### Elo Rating System

**Elo** es más simple que Bradley-Terry y actualiza ratings online (sin reentrenar todo).

```python
class EloRatingSystem:
    """
    Elo rating system para logos

    Basado en el sistema usado en ajedrez y Chatbot Arena
    """

    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor  # Learning rate
        self.initial_rating = initial_rating
        self.ratings = {}  # logo_id -> rating
        self.match_history = []

    def get_rating(self, logo_id):
        """Obtiene rating actual de un logo"""
        if logo_id not in self.ratings:
            self.ratings[logo_id] = self.initial_rating

        return self.ratings[logo_id]

    def expected_score(self, rating_a, rating_b):
        """
        Calcula score esperado

        E_A = 1 / (1 + 10^((R_B - R_A)/400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, logo_a_id, logo_b_id, winner):
        """
        Actualiza ratings después de un match

        Args:
            winner: 'a', 'b', or 'tie'
        """

        # Obtener ratings actuales
        rating_a = self.get_rating(logo_a_id)
        rating_b = self.get_rating(logo_b_id)

        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # Actual scores
        if winner == 'a':
            score_a, score_b = 1, 0
        elif winner == 'b':
            score_a, score_b = 0, 1
        else:  # tie
            score_a, score_b = 0.5, 0.5

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.ratings[logo_a_id] = new_rating_a
        self.ratings[logo_b_id] = new_rating_b

        # Record match
        self.match_history.append({
            'logo_a': logo_a_id,
            'logo_b': logo_b_id,
            'winner': winner,
            'rating_change_a': new_rating_a - rating_a,
            'rating_change_b': new_rating_b - rating_b,
            'timestamp': datetime.now()
        })

        return new_rating_a, new_rating_b

    def get_rankings(self):
        """Obtiene ranking de logos por Elo"""

        ranked = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        rankings = []
        for rank, (logo_id, rating) in enumerate(ranked, 1):
            # Calcular percentil
            percentile = (len(ranked) - rank + 1) / len(ranked) * 100

            rankings.append({
                'rank': rank,
                'logo_id': logo_id,
                'rating': rating,
                'percentile': percentile
            })

        return rankings

    def predict_winner(self, logo_a_id, logo_b_id):
        """Predice ganador basado en Elo"""

        rating_a = self.get_rating(logo_a_id)
        rating_b = self.get_rating(logo_b_id)

        prob_a_wins = self.expected_score(rating_a, rating_b)

        return {
            'logo_a': logo_a_id,
            'logo_b': logo_b_id,
            'rating_a': rating_a,
            'rating_b': rating_b,
            'prob_a_wins': prob_a_wins,
            'prob_b_wins': 1 - prob_a_wins,
            'predicted_winner': logo_a_id if prob_a_wins > 0.5 else logo_b_id,
            'confidence': abs(prob_a_wins - 0.5) * 2  # 0-1
        }
```

### Multi-Dimensional Quality Assessment

Basado en research 2024 (Multi-dimensional Human Preference dataset):

```python
class MultiDimensionalQualityPredictor:
    """
    Predictor de calidad multi-dimensional

    Dimensiones (basado en MHP dataset 2024):
    1. Technical Quality (sharpness, resolution, artifacts)
    2. Aesthetic Appeal (beauty, harmony, composition)
    3. Brand Fit (appropriateness for industry/brand)
    4. Text-Image Correspondence (matches prompt)
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.dimension_models = {
            'technical': None,
            'aesthetic': None,
            'brand_fit': None,
            'correspondence': None
        }
        self.is_trained = False

    def extract_features(self, logo_path, prompt):
        """Extrae features para predicción"""

        # CLIP embeddings
        image_emb = self.embedder.encode_image(logo_path)
        text_emb = self.embedder.encode_text(prompt)

        # Logo-specific features
        logo_features = self.embedder.extract_logo_features(logo_path)

        # Combined features
        features = {
            'image_embedding': image_emb,
            'text_embedding': text_emb,
            'clip_similarity': np.dot(image_emb.flatten(), text_emb.flatten()),
            **logo_features
        }

        return features

    def predict(self, logo_path, prompt):
        """
        Predice calidad en múltiples dimensiones

        Returns:
            dict con scores 0-1 para cada dimensión
        """

        features = self.extract_features(logo_path, prompt)

        # Por ahora, heurísticas (en producción serían modelos entrenados)
        predictions = {
            'technical_quality': self._predict_technical(features),
            'aesthetic_appeal': self._predict_aesthetic(features),
            'brand_fit': self._predict_brand_fit(features),
            'text_correspondence': features['clip_similarity'],
            'overall': 0.0
        }

        # Overall es promedio ponderado
        weights = {
            'technical_quality': 0.2,
            'aesthetic_appeal': 0.3,
            'brand_fit': 0.25,
            'text_correspondence': 0.25
        }

        predictions['overall'] = sum(
            predictions[dim] * weight
            for dim, weight in weights.items()
        )

        return predictions

    def _predict_technical(self, features):
        """Predice calidad técnica"""

        # Heurística: alto contraste + alta simplicidad = buena calidad
        technical_score = (
            features['contrast'] * 0.5 +
            features['simplicity_score'] * 0.3 +
            features['symmetry'] * 0.2
        )

        return min(max(technical_score, 0), 1)

    def _predict_aesthetic(self, features):
        """Predice appeal estético"""

        # Heurística: simetría + balance de colores
        aesthetic_score = (
            features['symmetry'] * 0.6 +
            (1 - abs(features['color_diversity'] - 0.3)) * 0.4
        )

        return min(max(aesthetic_score, 0), 1)

    def _predict_brand_fit(self, features):
        """Predice fit con marca/industria"""

        # Placeholder: en producción sería un modelo entrenado
        # que considera industria, competidores, etc.

        brand_fit_score = 0.7  # Default neutral

        return brand_fit_score

    def train_from_feedback(self, training_data):
        """
        Entrena modelos desde feedback humano

        Args:
            training_data: list of dicts con:
                - logo_path
                - prompt
                - ratings: dict con ratings para cada dimensión
        """

        print(f"Training quality predictors on {len(training_data)} examples...")

        # Aquí iría entrenamiento real de modelos
        # Por ejemplo, regresión con RandomForest o neural network

        self.is_trained = True

        return self
```

### Predictor de Calidad Pre-Generación

```python
class PreGenerationQualityPredictor:
    """
    Predice calidad ANTES de generar la imagen

    Basado solo en el prompt y metadata
    """

    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.prompt_quality_map = self._build_prompt_quality_map()

    def _build_prompt_quality_map(self):
        """Construye mapa de características de prompts → calidad"""

        quality_map = {
            'keywords': {},  # keyword → avg quality
            'length': {},    # prompt length → avg quality
            'industries': {}  # industry → avg quality
        }

        for data in self.historical_data:
            prompt = data['prompt']
            quality = data['rating']
            industry = data.get('industry', 'unknown')

            # Keywords
            words = set(prompt.lower().split())
            for word in words:
                if word not in quality_map['keywords']:
                    quality_map['keywords'][word] = []
                quality_map['keywords'][word].append(quality)

            # Length
            length_bucket = len(prompt) // 20 * 20
            if length_bucket not in quality_map['length']:
                quality_map['length'][length_bucket] = []
            quality_map['length'][length_bucket].append(quality)

            # Industry
            if industry not in quality_map['industries']:
                quality_map['industries'][industry] = []
            quality_map['industries'][industry].append(quality)

        # Calcular promedios
        for category in quality_map:
            for key in quality_map[category]:
                quality_map[category][key] = np.mean(
                    quality_map[category][key]
                )

        return quality_map

    def predict_from_prompt(self, prompt, industry='unknown'):
        """Predice calidad esperada desde el prompt"""

        # Extraer features del prompt
        words = set(prompt.lower().split())
        length = len(prompt)
        length_bucket = length // 20 * 20

        # Scores parciales
        scores = []

        # Score por keywords
        keyword_scores = [
            self.prompt_quality_map['keywords'].get(word, 3.0)
            for word in words
            if word in self.prompt_quality_map['keywords']
        ]
        if keyword_scores:
            scores.append(np.mean(keyword_scores))

        # Score por length
        length_score = self.prompt_quality_map['length'].get(
            length_bucket, 3.0
        )
        scores.append(length_score)

        # Score por industria
        industry_score = self.prompt_quality_map['industries'].get(
            industry, 3.0
        )
        scores.append(industry_score)

        # Promedio
        predicted_quality = np.mean(scores) if scores else 3.0

        # Normalizar a 0-1
        normalized = (predicted_quality - 1) / 4

        return normalized
```

---

## Case Studies: Estrategias de Mejora

### Midjourney

**Estrategias de Mejora Identificadas (2024-2025):**

1. **Parámetros de Consistencia Estilística:**
   - `--sref` (style reference): Mantiene estilo consistente
   - `--cref` (character reference): Mantiene personajes consistentes
   - Permite a usuarios crear visual identity coherente

2. **Mejora en Comprensión de Prompts Largos:**
   - V6/V7 procesa prompts largos mejor que versiones anteriores
   - Entiende matices sutiles y contexto complejo

3. **Generación 25% Más Rápida:**
   - Optimizaciones de infraestructura
   - Mejor uso de GPUs

4. **Parámetros de Control:**
   - `--ar` (aspect ratio): Control preciso de dimensiones
   - `--v` (version): Permite comparar versiones del modelo
   - `--chaos`: Controla variabilidad de resultados

**Implementación para Logos:**

```python
class MidjourneyInspiredGenerator:
    """Generador inspirado en estrategias de Midjourney"""

    def __init__(self):
        self.style_references = {}  # Style memory
        self.version_history = {}   # Track model versions

    def generate_with_style_consistency(self, prompt, style_ref_id=None):
        """Genera logo manteniendo consistencia de estilo"""

        if style_ref_id and style_ref_id in self.style_references:
            # Recuperar estilo de referencia
            style_params = self.style_references[style_ref_id]

            # Augmentar prompt con parámetros de estilo
            augmented_prompt = self._apply_style_params(
                prompt, style_params
            )
        else:
            augmented_prompt = prompt

        # Generar
        logo = self.generate(augmented_prompt)

        # Si es nuevo estilo, guardarlo
        if style_ref_id and style_ref_id not in self.style_references:
            style_params = self._extract_style_params(logo)
            self.style_references[style_ref_id] = style_params

        return logo

    def _apply_style_params(self, prompt, style_params):
        """Aplica parámetros de estilo al prompt"""

        style_descriptors = []

        if 'color_palette' in style_params:
            colors = ', '.join(style_params['color_palette'])
            style_descriptors.append(f"using colors: {colors}")

        if 'visual_style' in style_params:
            style_descriptors.append(style_params['visual_style'])

        if style_descriptors:
            return f"{prompt}, {', '.join(style_descriptors)}"

        return prompt

    def _extract_style_params(self, logo):
        """Extrae parámetros de estilo de un logo generado"""

        # Placeholder: en producción, usaría análisis de imagen
        params = {
            'color_palette': ['blue', 'white'],
            'visual_style': 'minimalist geometric'
        }

        return params
```

### DALL-E 3

**Estrategias de Mejora (2024):**

1. **Integración con ChatGPT:**
   - Usa GPT-4 para reformular prompts automáticamente
   - 94% accuracy en interpretar instrucciones conversacionales
   - Usuarios pueden iterar en lenguaje natural

2. **Mejor Comprensión de Matices:**
   - Entiende contexto y subtlezas mejor que DALL-E 2
   - Menos necesidad de "prompt engineering" experto

3. **Prompt Rewriting Automático:**
   - GPT-4 expande prompts cortos a descripciones detalladas
   - Mejora calidad sin esfuerzo del usuario

**Implementación:**

```python
class GPTEnhancedPromptRewriter:
    """Reescribe prompts usando LLM para mejorar calidad"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def rewrite_prompt(self, user_prompt, context):
        """Reescribe prompt del usuario a versión optimizada"""

        system_prompt = """You are an expert prompt engineer for logo generation.

Your task: Rewrite user prompts to be more effective for AI logo generation.

Guidelines:
- Keep core intent of user
- Add specific visual details (shapes, colors, composition)
- Include style keywords (minimalist, modern, vintage, etc.)
- Mention logo-specific requirements (scalable, simple, memorable)
- Keep under 100 words

User context:
- Industry: {industry}
- Target audience: {audience}
- Style preference: {style}
"""

        formatted_system = system_prompt.format(
            industry=context.get('industry', 'unknown'),
            audience=context.get('audience', 'general'),
            style=context.get('style', 'modern')
        )

        # Call LLM
        rewritten = self.llm.complete(
            system=formatted_system,
            user=f"Rewrite this logo prompt: {user_prompt}"
        )

        return rewritten

    def iterative_refinement(self, initial_prompt, feedback_history):
        """Refina prompt basado en feedback previo"""

        refinement_prompt = f"""Original prompt: {initial_prompt}

Previous attempts and feedback:
{self._format_feedback_history(feedback_history)}

Based on this feedback, rewrite the prompt to address the issues mentioned.
Focus on fixing what didn't work while keeping what did work.
"""

        refined = self.llm.complete(refinement_prompt)

        return refined

    def _format_feedback_history(self, history):
        """Formatea historial de feedback"""

        formatted = []
        for i, entry in enumerate(history, 1):
            formatted.append(
                f"Attempt {i}: {entry['prompt']}\n"
                f"Feedback: {entry['feedback']}\n"
                f"Rating: {entry['rating']}/5"
            )

        return "\n\n".join(formatted)
```

### Stable Diffusion

**Estrategias de Mejora:**

1. **Community Fine-Tuning:**
   - Cientos de modelos especializados (LoRA, DreamBooth)
   - Modelos específicos para estilos (photorealistic, anime, painting)
   - Supera modelos universales en nichos específicos

2. **ControlNet:**
   - Control preciso con condiciones adicionales (pose, depth, edges)
   - Permite consistencia mayor

3. **Textual Inversion:**
   - Aprende nuevos conceptos con pocas imágenes
   - Ideal para estilos de marca específicos

**Implementación para Logos:**

```python
class LogoStyleFineTuner:
    """Fine-tuning de modelos para estilos específicos de logos"""

    def __init__(self, base_model):
        self.base_model = base_model
        self.lora_adapters = {}  # Style-specific LoRA adapters

    def fine_tune_on_brand(self, brand_name, example_logos,
                           n_epochs=100):
        """
        Fine-tune en logos de una marca específica

        Usa LoRA (Low-Rank Adaptation) para fine-tuning eficiente
        """

        print(f"Fine-tuning model for {brand_name} style...")
        print(f"Training on {len(example_logos)} examples")

        # Aquí iría el código real de fine-tuning
        # Por ahora, simulamos

        lora_adapter = self._train_lora_adapter(
            example_logos,
            n_epochs
        )

        self.lora_adapters[brand_name] = lora_adapter

        print(f"✓ Created LoRA adapter for {brand_name}")

        return lora_adapter

    def generate_with_brand_style(self, prompt, brand_name):
        """Genera logo usando estilo de marca fine-tuned"""

        if brand_name not in self.lora_adapters:
            raise ValueError(f"No adapter found for {brand_name}")

        # Load adapter
        adapter = self.lora_adapters[brand_name]

        # Generate with adapted model
        logo = self._generate_with_adapter(prompt, adapter)

        return logo

    def _train_lora_adapter(self, examples, n_epochs):
        """Entrena LoRA adapter (placeholder)"""

        # En producción, usaría diffusers library:
        # from diffusers import StableDiffusionPipeline, LoRAModel

        adapter = {
            'type': 'lora',
            'n_examples': len(examples),
            'n_epochs': n_epochs,
            'timestamp': datetime.now()
        }

        return adapter

    def _generate_with_adapter(self, prompt, adapter):
        """Genera con adapter cargado (placeholder)"""

        # Simulated generation
        logo = {
            'prompt': prompt,
            'adapter_used': adapter,
            'generated_at': datetime.now()
        }

        return logo
```

### LogoAI, Looka, Brandmark

Aunque detalles técnicos no son públicos, podemos inferir estrategias:

**Estrategias Comunes:**

1. **Machine Learning en Preferencias:**
   - Aprenden de selecciones de usuarios
   - Ajustan recomendaciones en tiempo real

2. **Principios de Diseño Codificados:**
   - Aplican reglas de teoría de color
   - Font pairings basados en datos
   - Visual hierarchy automática

3. **Generación Variante Rápida:**
   - Múltiples opciones en segundos
   - Variaciones con cambios controlados

4. **Brand Kit Completo:**
   - No solo logos, sino paletas, fonts, templates
   - Consistencia cross-platform

**Implementación Inspirada:**

```python
class CompleteBrandingSystem:
    """Sistema completo de branding inspirado en LogoAI/Looka"""

    def __init__(self, logo_generator, preference_model):
        self.logo_gen = logo_generator
        self.preference_model = preference_model
        self.design_rules = DesignRulesEngine()

    def generate_brand_kit(self, company_name, industry,
                          style_preferences):
        """Genera kit completo de branding"""

        # 1. Generate logo variants
        logo_variants = self._generate_logo_variants(
            company_name, industry, style_preferences
        )

        # 2. Rank by predicted preference
        ranked_logos = self.preference_model.rank(logo_variants)

        # 3. Select best logo
        best_logo = ranked_logos[0]

        # 4. Generate color palette from logo
        color_palette = self.design_rules.extract_palette(best_logo)

        # 5. Select matching fonts
        font_pairing = self.design_rules.select_fonts(
            best_logo, industry
        )

        # 6. Create brand kit
        brand_kit = {
            'primary_logo': best_logo,
            'logo_variants': ranked_logos[:5],
            'color_palette': color_palette,
            'typography': font_pairing,
            'style_guide': self._generate_style_guide(
                best_logo, color_palette, font_pairing
            )
        }

        return brand_kit

    def _generate_logo_variants(self, company_name, industry,
                               style_prefs, n_variants=10):
        """Genera múltiples variantes de logo"""

        variants = []

        # Base prompt
        base_prompt = f"{industry} logo for {company_name}"

        # Generar variaciones
        for i in range(n_variants):
            # Variar estilo
            style_variation = self._sample_style_variation(style_prefs)

            variant_prompt = f"{base_prompt}, {style_variation}"

            logo = self.logo_gen.generate(variant_prompt)
            variants.append(logo)

        return variants

    def _sample_style_variation(self, base_style):
        """Genera variación de estilo"""

        style_options = {
            'minimalist': [
                'clean lines',
                'geometric shapes',
                'negative space'
            ],
            'modern': [
                'gradient colors',
                'sleek design',
                'contemporary'
            ],
            'vintage': [
                'retro style',
                'classic typography',
                'badge design'
            ]
        }

        if base_style in style_options:
            variations = style_options[base_style]
            return np.random.choice(variations)

        return base_style

    def _generate_style_guide(self, logo, palette, fonts):
        """Genera guía de estilo"""

        guide = {
            'logo_usage': {
                'min_size': '40px',
                'clear_space': '20% of logo width',
                'backgrounds': ['white', 'dark', 'color']
            },
            'colors': {
                'primary': palette[0],
                'secondary': palette[1] if len(palette) > 1 else None,
                'accent': palette[2] if len(palette) > 2 else None
            },
            'typography': {
                'headings': fonts['heading'],
                'body': fonts['body']
            }
        }

        return guide


class DesignRulesEngine:
    """Motor de reglas de diseño"""

    def extract_palette(self, logo, n_colors=5):
        """Extrae paleta de colores del logo"""

        # Placeholder: en producción usaría K-means clustering
        palette = [
            '#2C3E50',  # Dark blue
            '#3498DB',  # Blue
            '#ECF0F1',  # Light gray
            '#E74C3C',  # Red (accent)
            '#FFFFFF'   # White
        ]

        return palette[:n_colors]

    def select_fonts(self, logo, industry):
        """Selecciona font pairing apropiado"""

        # Font pairings basados en industria
        industry_fonts = {
            'tech': {
                'heading': 'Roboto',
                'body': 'Open Sans'
            },
            'fashion': {
                'heading': 'Playfair Display',
                'body': 'Lato'
            },
            'food': {
                'heading': 'Lobster',
                'body': 'Raleway'
            }
        }

        return industry_fonts.get(industry, {
            'heading': 'Montserrat',
            'body': 'Roboto'
        })

    def validate_contrast(self, color1, color2, min_ratio=4.5):
        """Valida contraste WCAG entre dos colores"""

        # Placeholder: en producción calcularía contraste real
        return True
```

---

## Pipeline de Mejora Iterativa

### Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT PIPELINE               │
└─────────────────────────────────────────────────────────────────┘

DAILY:
1. Collect Feedback
   ├─ Ratings (1-5 stars)
   ├─ Pairwise comparisons
   ├─ Implicit signals (downloads, shares)
   └─ Comments and reports

2. Update Prompt Optimizer (Bandit)
   ├─ Update arm statistics
   ├─ Re-calculate best templates
   └─ Adjust exploration rate

3. Update Vector Database
   ├─ Add new high-quality logos
   ├─ Update metadata with ratings
   └─ Re-index if needed

WEEKLY:
4. Retrain Quality Predictor
   ├─ Collect new training data (feedback from week)
   ├─ Retrain Bradley-Terry / Elo model
   ├─ Validate on held-out set
   └─ Deploy if improvement > threshold

5. Analyze Performance Metrics
   ├─ User satisfaction trend
   ├─ Quality score trend
   ├─ Prompt optimizer performance
   └─ Generate report

MONTHLY:
6. Fine-tune Generation Model
   ├─ Collect top-rated logos (rating >= 4.5)
   ├─ Create LoRA adapter
   ├─ Validate quality
   └─ Deploy to production

7. Curriculum Advancement
   ├─ Check if ready for next level
   ├─ Unlock new complexity
   └─ Update constraints
```

### Implementación Completa

```python
from datetime import datetime, timedelta
import schedule
import time

class ContinuousImprovementPipeline:
    """Pipeline completo de mejora continua"""

    def __init__(self, config):
        # Components
        self.vector_db = LogoVectorDatabase()
        self.embedder = EnhancedLogoEmbedder()
        self.feedback_collector = FeedbackCollector()
        self.prompt_optimizer = PromptOptimizer(config['prompt_templates'])
        self.quality_predictor = MultiDimensionalQualityPredictor(self.embedder)
        self.elo_system = EloRatingSystem()
        self.curriculum_manager = CurriculumManager()

        # State
        self.metrics_history = []
        self.last_model_update = None

    def run(self):
        """Inicia el pipeline continuo"""

        # Schedule daily tasks
        schedule.every().day.at("02:00").do(self.daily_update)

        # Schedule weekly tasks
        schedule.every().monday.at("03:00").do(self.weekly_retrain)

        # Schedule monthly tasks
        schedule.every().month.do(self.monthly_fine_tune)

        print("Continuous improvement pipeline started")

        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

    def daily_update(self):
        """Actualización diaria"""

        print(f"\n=== DAILY UPDATE: {datetime.now()} ===")

        # 1. Collect feedback from last 24h
        feedback = self.feedback_collector.get_feedback_since(
            datetime.now() - timedelta(days=1)
        )

        print(f"Collected {len(feedback)} feedback items")

        # 2. Update prompt optimizer
        self._update_prompt_optimizer(feedback)

        # 3. Update vector database
        self._update_vector_database(feedback)

        # 4. Update Elo ratings
        self._update_elo_ratings(feedback)

        # 5. Log metrics
        self._log_daily_metrics()

        print("Daily update completed\n")

    def weekly_retrain(self):
        """Reentrenamiento semanal"""

        print(f"\n=== WEEKLY RETRAIN: {datetime.now()} ===")

        # 1. Collect training data from last week
        training_data = self.feedback_collector.get_feedback_since(
            datetime.now() - timedelta(days=7)
        )

        print(f"Collected {len(training_data)} training examples")

        # 2. Retrain quality predictor
        if len(training_data) >= 50:
            self.quality_predictor.train_from_feedback(training_data)

            # 3. Validate
            validation_score = self._validate_quality_predictor()

            print(f"Validation score: {validation_score:.3f}")

            # 4. Deploy if improved
            if validation_score > 0.75:
                self._deploy_quality_predictor()
                print("✓ New quality predictor deployed")
            else:
                print("✗ Quality predictor not deployed (score too low)")
        else:
            print("✗ Not enough training data, skipping retrain")

        # 5. Generate performance report
        self._generate_weekly_report()

        print("Weekly retrain completed\n")

    def monthly_fine_tune(self):
        """Fine-tuning mensual del modelo generativo"""

        print(f"\n=== MONTHLY FINE-TUNE: {datetime.now()} ===")

        # 1. Collect top-rated logos from last month
        top_logos = self.vector_db.query_similar(
            query_embedding=None,  # Get all
            n_results=1000,
            collection='high_quality',
            filters={'rating': {'$gte': 4.5}}
        )

        print(f"Collected {len(top_logos['ids'])} top-rated logos")

        # 2. Fine-tune model (LoRA)
        if len(top_logos['ids']) >= 100:
            self._fine_tune_generator(top_logos)
            print("✓ Generator fine-tuned")
        else:
            print("✗ Not enough high-quality examples")

        # 3. Check curriculum advancement
        self._check_curriculum_advancement()

        # 4. Generate monthly report
        self._generate_monthly_report()

        print("Monthly fine-tune completed\n")

    def _update_prompt_optimizer(self, feedback):
        """Actualiza bandit de optimización de prompts"""

        for item in feedback:
            if 'prompt_template_arm' in item:
                arm = item['prompt_template_arm']
                rating = item['rating']

                self.prompt_optimizer.update_with_feedback(arm, rating)

        print(f"Updated prompt optimizer with {len(feedback)} items")

    def _update_vector_database(self, feedback):
        """Actualiza base de datos vectorial"""

        for item in feedback:
            logo_id = item['logo_id']
            rating = item['rating']

            # Update rating in database
            self.vector_db.update_logo_rating(logo_id, rating)

        print("Vector database updated")

    def _update_elo_ratings(self, feedback):
        """Actualiza ratings Elo"""

        comparisons = [
            item for item in feedback
            if 'comparison' in item
        ]

        for comp in comparisons:
            self.elo_system.update_ratings(
                comp['logo_a_id'],
                comp['logo_b_id'],
                comp['winner']
            )

        print(f"Updated Elo ratings with {len(comparisons)} comparisons")

    def _log_daily_metrics(self):
        """Registra métricas diarias"""

        metrics = {
            'date': datetime.now(),
            'db_stats': self.vector_db.get_statistics(),
            'elo_top_10': self.elo_system.get_rankings()[:10],
            'prompt_optimizer_stats':
                self.prompt_optimizer.get_performance_report()
        }

        self.metrics_history.append(metrics)

        print("Daily metrics logged")

    def _validate_quality_predictor(self):
        """Valida predictor de calidad en conjunto de validación"""

        # Placeholder: en producción usaría conjunto de validación real
        validation_score = 0.8

        return validation_score

    def _deploy_quality_predictor(self):
        """Deploys nuevo predictor de calidad"""

        # Save model
        # Update production endpoint

        self.last_model_update = datetime.now()

    def _generate_weekly_report(self):
        """Genera reporte semanal de performance"""

        # Get last 7 days of metrics
        recent_metrics = [
            m for m in self.metrics_history
            if m['date'] > datetime.now() - timedelta(days=7)
        ]

        if not recent_metrics:
            return

        report = {
            'period': 'last_7_days',
            'total_logos_generated': sum(
                m['db_stats']['total_logos'] for m in recent_metrics
            ),
            'quality_ratio_trend': [
                m['db_stats']['quality_ratio'] for m in recent_metrics
            ],
            'best_prompt_template':
                self.prompt_optimizer.get_best_template()['name']
        }

        print("\n--- WEEKLY REPORT ---")
        print(f"Total logos: {report['total_logos_generated']}")
        print(f"Quality ratio: {report['quality_ratio_trend'][-1]:.2%}")
        print(f"Best prompt: {report['best_prompt_template']}")
        print("-------------------\n")

    def _generate_monthly_report(self):
        """Genera reporte mensual"""

        print("\n--- MONTHLY REPORT ---")
        print(f"Database size: {self.vector_db.get_statistics()['total_logos']}")
        print(f"High quality ratio: {self.vector_db.get_statistics()['quality_ratio']:.2%}")
        print(f"Curriculum level: {self.curriculum_manager.current_level}")
        print("--------------------\n")

    def _fine_tune_generator(self, top_logos):
        """Fine-tune el generador con mejores logos"""

        # Placeholder: aquí iría fine-tuning real
        print("Fine-tuning generator (simulated)...")

    def _check_curriculum_advancement(self):
        """Verifica si avanzar en curriculum"""

        # Get performance from last month
        recent_metrics = [
            m for m in self.metrics_history
            if m['date'] > datetime.now() - timedelta(days=30)
        ]

        if not recent_metrics:
            return

        avg_quality = np.mean([
            m['db_stats']['quality_ratio'] for m in recent_metrics
        ])

        n_samples = len(recent_metrics)

        advanced = self.curriculum_manager.record_performance(
            accuracy=avg_quality,
            n_samples=n_samples
        )

        if advanced:
            print(f"✓ Advanced to curriculum level {self.curriculum_manager.current_level}")
```

### Script de Ejecución

```python
# run_pipeline.py

if __name__ == "__main__":
    config = {
        'prompt_templates': [
            {
                'name': 'minimalist',
                'template': 'minimalist {industry} logo, simple geometric shapes, {colors}'
            },
            {
                'name': 'modern',
                'template': 'modern {industry} logo with gradient, sleek design, {colors}'
            },
            {
                'name': 'vintage',
                'template': 'vintage {industry} logo, retro style, {colors}, badge design'
            }
        ],
        'vector_db_path': './chroma_db',
        'model_checkpoint_path': './models/checkpoints'
    }

    pipeline = ContinuousImprovementPipeline(config)

    # Run continuous improvement
    pipeline.run()
```

---

## Métricas de Mejora Real

### Métricas Clave (NO solo scores técnicos)

```python
class RealWorldMetrics:
    """Métricas de mejora en el mundo real"""

    def __init__(self):
        self.metrics = {
            # User Satisfaction
            'user_satisfaction': {
                'avg_rating': [],
                'rating_distribution': [],
                'nps_score': []  # Net Promoter Score
            },

            # Engagement
            'engagement': {
                'download_rate': [],
                'share_rate': [],
                'time_to_decision': [],  # Tiempo hasta elegir logo
                'iterations_per_session': []  # Cuántas veces regeneran
            },

            # Quality Improvement
            'quality': {
                'high_quality_ratio': [],  # % con rating >= 4
                'avg_quality_score': [],
                'quality_variance': []  # Consistencia
            },

            # Efficiency
            'efficiency': {
                'time_to_generate': [],
                'api_cost_per_logo': [],
                'rejection_rate': []  # % de logos rechazados
            },

            # Business Impact
            'business': {
                'conversion_rate': [],  # % que compran/descargan
                'retention_rate': [],  # % que vuelven
                'customer_lifetime_value': []
            }
        }

    def record_session(self, session_data):
        """Registra métricas de una sesión de usuario"""

        # User Satisfaction
        if 'final_rating' in session_data:
            self.metrics['user_satisfaction']['avg_rating'].append(
                session_data['final_rating']
            )

        # Engagement
        self.metrics['engagement']['download_rate'].append(
            1 if session_data.get('downloaded') else 0
        )

        self.metrics['engagement']['time_to_decision'].append(
            session_data.get('decision_time_seconds', 0)
        )

        self.metrics['engagement']['iterations_per_session'].append(
            session_data.get('n_generations', 1)
        )

        # Efficiency
        self.metrics['efficiency']['rejection_rate'].append(
            session_data.get('n_rejections', 0) /
            max(session_data.get('n_generations', 1), 1)
        )

    def get_improvement_report(self, window_days=30):
        """
        Genera reporte de mejora comparando períodos

        Compara últimos window_days vs período anterior
        """

        now = datetime.now()
        cutoff = now - timedelta(days=window_days)
        previous_cutoff = cutoff - timedelta(days=window_days)

        report = {}

        # User Satisfaction Improvement
        recent_ratings = self._get_metrics_in_window(
            'user_satisfaction', 'avg_rating', cutoff, now
        )
        previous_ratings = self._get_metrics_in_window(
            'user_satisfaction', 'avg_rating', previous_cutoff, cutoff
        )

        if recent_ratings and previous_ratings:
            improvement = (
                np.mean(recent_ratings) - np.mean(previous_ratings)
            )
            report['satisfaction_improvement'] = {
                'absolute_change': improvement,
                'percent_change': improvement / np.mean(previous_ratings) * 100,
                'current_avg': np.mean(recent_ratings),
                'previous_avg': np.mean(previous_ratings)
            }

        # Efficiency Improvement
        recent_rejections = self._get_metrics_in_window(
            'efficiency', 'rejection_rate', cutoff, now
        )
        previous_rejections = self._get_metrics_in_window(
            'efficiency', 'rejection_rate', previous_cutoff, cutoff
        )

        if recent_rejections and previous_rejections:
            improvement = (
                np.mean(previous_rejections) - np.mean(recent_rejections)
            )
            report['efficiency_improvement'] = {
                'rejection_rate_reduction': improvement,
                'percent_improvement':
                    improvement / np.mean(previous_rejections) * 100,
                'current_rejection_rate': np.mean(recent_rejections),
                'previous_rejection_rate': np.mean(previous_rejections)
            }

        # Engagement Improvement
        recent_downloads = self._get_metrics_in_window(
            'engagement', 'download_rate', cutoff, now
        )
        previous_downloads = self._get_metrics_in_window(
            'engagement', 'download_rate', previous_cutoff, cutoff
        )

        if recent_downloads and previous_downloads:
            improvement = (
                np.mean(recent_downloads) - np.mean(previous_downloads)
            )
            report['engagement_improvement'] = {
                'download_rate_increase': improvement,
                'percent_improvement':
                    improvement / np.mean(previous_downloads) * 100,
                'current_download_rate': np.mean(recent_downloads),
                'previous_download_rate': np.mean(previous_downloads)
            }

        return report

    def _get_metrics_in_window(self, category, metric, start, end):
        """Obtiene métricas en ventana de tiempo"""

        # Placeholder: en producción, filtrar por timestamp
        all_values = self.metrics[category][metric]

        # Simular filtrado temporal
        # (asumir que métricas están en orden cronológico)
        window_size = len(all_values) // 4

        if len(all_values) < window_size * 2:
            return all_values

        return all_values[-window_size:]

    def print_dashboard(self):
        """Imprime dashboard de métricas"""

        print("\n" + "="*60)
        print("           REAL-WORLD METRICS DASHBOARD")
        print("="*60 + "\n")

        # User Satisfaction
        if self.metrics['user_satisfaction']['avg_rating']:
            avg_rating = np.mean(
                self.metrics['user_satisfaction']['avg_rating']
            )
            print(f"📊 User Satisfaction")
            print(f"   Average Rating: {avg_rating:.2f}/5.0")
            print()

        # Engagement
        if self.metrics['engagement']['download_rate']:
            download_rate = np.mean(
                self.metrics['engagement']['download_rate']
            )
            print(f"🎯 Engagement")
            print(f"   Download Rate: {download_rate:.1%}")

            if self.metrics['engagement']['iterations_per_session']:
                avg_iterations = np.mean(
                    self.metrics['engagement']['iterations_per_session']
                )
                print(f"   Avg Iterations: {avg_iterations:.1f}")
            print()

        # Quality
        if self.metrics['quality']['high_quality_ratio']:
            quality_ratio = np.mean(
                self.metrics['quality']['high_quality_ratio']
            )
            print(f"⭐ Quality")
            print(f"   High Quality Ratio: {quality_ratio:.1%}")
            print()

        # Efficiency
        if self.metrics['efficiency']['rejection_rate']:
            rejection_rate = np.mean(
                self.metrics['efficiency']['rejection_rate']
            )
            print(f"⚡ Efficiency")
            print(f"   Rejection Rate: {rejection_rate:.1%}")
            print()

        print("="*60 + "\n")


class ABTestMetrics:
    """Métricas para A/B testing de mejoras"""

    def __init__(self):
        self.experiments = {}

    def create_experiment(self, name, control_group, treatment_group):
        """Crea nuevo experimento A/B"""

        self.experiments[name] = {
            'control': control_group,
            'treatment': treatment_group,
            'start_date': datetime.now(),
            'metrics': {
                'control': [],
                'treatment': []
            }
        }

    def record_metric(self, experiment_name, group, value):
        """Registra métrica para un grupo"""

        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        self.experiments[experiment_name]['metrics'][group].append(value)

    def analyze_experiment(self, experiment_name, metric='conversion'):
        """Analiza resultados de experimento"""

        exp = self.experiments[experiment_name]

        control_values = exp['metrics']['control']
        treatment_values = exp['metrics']['treatment']

        if not control_values or not treatment_values:
            return None

        # Calculate statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)

        # Effect size
        lift = (treatment_mean - control_mean) / control_mean * 100

        # Simple significance test (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)

        is_significant = p_value < 0.05

        result = {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift': lift,
            'p_value': p_value,
            'is_significant': is_significant,
            'sample_sizes': {
                'control': len(control_values),
                'treatment': len(treatment_values)
            }
        }

        return result

    def print_experiment_results(self, experiment_name):
        """Imprime resultados de experimento"""

        result = self.analyze_experiment(experiment_name)

        if result is None:
            print("Not enough data to analyze")
            return

        print(f"\n{'='*60}")
        print(f"   A/B TEST RESULTS: {experiment_name}")
        print(f"{'='*60}\n")

        print(f"Control Mean:    {result['control_mean']:.3f}")
        print(f"Treatment Mean:  {result['treatment_mean']:.3f}")
        print(f"Lift:            {result['lift']:+.1f}%")
        print(f"P-value:         {result['p_value']:.4f}")
        print(f"Significant:     {'✓ YES' if result['is_significant'] else '✗ NO'}")
        print(f"\nSample Sizes:")
        print(f"  Control:   {result['sample_sizes']['control']}")
        print(f"  Treatment: {result['sample_sizes']['treatment']}")
        print(f"\n{'='*60}\n")
```

### Ejemplo de Uso Completo

```python
# Ejemplo de tracking de métricas en producción

metrics_tracker = RealWorldMetrics()
ab_test = ABTestMetrics()

# Setup A/B test: Nueva estrategia de prompts
ab_test.create_experiment(
    name='prompt_strategy_v2',
    control_group='original_prompts',
    treatment_group='rag_enhanced_prompts'
)

# Simular sesiones de usuarios
for session_id in range(100):
    # Asignar a grupo
    group = 'control' if session_id % 2 == 0 else 'treatment'

    # Simular generación de logo
    if group == 'control':
        rating = np.random.normal(3.5, 0.8)
        downloaded = np.random.random() < 0.4
    else:  # treatment (con RAG)
        rating = np.random.normal(4.2, 0.6)  # Mejor
        downloaded = np.random.random() < 0.6  # Más descargas

    # Registrar métricas
    session_data = {
        'session_id': session_id,
        'group': group,
        'final_rating': max(1, min(5, rating)),
        'downloaded': downloaded,
        'decision_time_seconds': np.random.uniform(30, 180),
        'n_generations': np.random.randint(1, 5),
        'n_rejections': np.random.randint(0, 3)
    }

    metrics_tracker.record_session(session_data)
    ab_test.record_metric('prompt_strategy_v2', group,
                         1 if downloaded else 0)

# Analizar resultados
metrics_tracker.print_dashboard()
ab_test.print_experiment_results('prompt_strategy_v2')

# Improvement report
improvement = metrics_tracker.get_improvement_report(window_days=30)
print("\nIMPROVEMENT OVER LAST 30 DAYS:")
for metric, data in improvement.items():
    print(f"\n{metric}:")
    for key, value in data.items():
        print(f"  {key}: {value}")
```

---

## Conclusión

Este documento presenta una arquitectura completa para un sistema de aprendizaje continuo en generación de logos con IA. Los componentes clave son:

1. **Embeddings (CLIP + Custom)**: Representaciones ricas de logos
2. **Vector Database (ChromaDB)**: Almacenamiento y retrieval eficiente
3. **RAG Pipeline**: Generación aumentada con ejemplos exitosos
4. **Feedback Loops**: Active learning y bandit algorithms
5. **Quality Prediction**: Bradley-Terry y Elo para ranking
6. **Continuous Improvement**: Pipeline automatizado de mejora
7. **Real-World Metrics**: Métricas que importan (no solo FID/CLIP scores)

### Próximos Pasos

1. **Implementar embedding pipeline** con CLIP
2. **Setup ChromaDB** y empezar a indexar logos
3. **Implementar feedback collector** en UI
4. **Entrenar primer quality predictor** con datos iniciales
5. **Deploy pipeline de mejora continua**
6. **Monitorear métricas reales** y ajustar

### Referencias

- FashionLOGO (arXiv 2024): State-of-the-art logo embeddings
- NTIRE 2024 Challenge: Quality assessment for AI-generated images
- Bradley-Terry Extensions (LMSYS 2024): Preference modeling
- Multi-dimensional Human Preference (CVPR 2024): Quality dimensions
- Thompson Sampling for Prompt Optimization (arXiv 2024)

---

**Última actualización:** 2025-11-25
**Versión:** 1.0
**Autor:** Sistema de Investigación IA
