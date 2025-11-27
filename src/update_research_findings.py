"""
Actualiza la base de conocimiento con los nuevos hallazgos de investigación
"""

from knowledge_base import SVGKnowledgeBase


def add_design_principles(kb: SVGKnowledgeBase):
    """Agrega principios de diseño profesional"""

    kb.add_technique(
        name="Golden Ratio Logo Design",
        description="Aplicación de la proporción áurea (φ=1.618) en construcción de logos profesionales. Usado en Apple, Twitter, Pepsi.",
        category="Design-Principles",
        difficulty="Medium",
        use_cases=[
            "Construcción geométrica de logos",
            "Proporciones armónicas en elementos",
            "Grid systems basados en φ",
            "Logos que necesitan balance matemático"
        ]
    )

    kb.add_technique(
        name="Gestalt Principles for Logos",
        description="Aplicación de 5 principios de Gestalt: Closure, Proximity, Similarity, Figure-Ground, Continuation. FedEx usa figure-ground para flecha oculta.",
        category="Design-Principles",
        difficulty="Medium",
        use_cases=[
            "Crear logos memorables con espacio negativo",
            "Diseños que necesitan percepción visual efectiva",
            "Logos con elementos ocultos o doble significado",
            "Optimización de reconocimiento visual"
        ]
    )

    kb.add_technique(
        name="Color Psychology for Branding",
        description="El color aumenta reconocimiento de marca en 80%. Azul lidera con 33% (confianza), rojo 29% (energía), amarillo 13% (optimismo). Máximo 1-3 colores.",
        category="Design-Principles",
        difficulty="Low",
        use_cases=[
            "Selección de paleta de colores según industria",
            "Branding emocional y psicológico",
            "Logos para mercados específicos",
            "Optimización de memorabilidad"
        ]
    )

    kb.add_technique(
        name="Simplicity Sweet Spot",
        description="Logos profesionales promedian 32 puntos de complejidad (categoría simple). Rango óptimo: 20-40. Nike Swoosh: ~15 (ultra simple).",
        category="Design-Principles",
        difficulty="Low",
        use_cases=[
            "Evaluar complejidad de logos generados",
            "Guiar simplificación de diseños",
            "Benchmarking contra logos profesionales",
            "Optimización para escalabilidad"
        ]
    )

    kb.add_technique(
        name="SVG Path Optimization",
        description="Reducción de 50-80% en tamaño de archivo. Técnicas: simplificación de Bézier, reducción de precisión (2-3 decimales), merge de paths.",
        category="Technical-SVG",
        difficulty="Medium",
        use_cases=[
            "Optimizar SVG generados por IA",
            "Mejorar performance de carga",
            "Clean up de código SVG",
            "Preparación para producción"
        ]
    )


def add_datasets_info(kb: SVGKnowledgeBase):
    """Agrega información sobre datasets"""

    kb.add_paper(
        title="SVG-1M: 1 Million SVG-Text Pairs Dataset",
        authors="Various (2024-2025)",
        summary="Dataset de 1 millón de pares texto-SVG con código vectorial real. Único dataset con formato SVG nativo para fine-tuning de LLMs.",
        key_findings=[
            "Único dataset con código SVG como texto (no raster)",
            "Disponible en HuggingFace",
            "Ideal para fine-tuning de LLMs (GPT, Llama, Claude)",
            "Mejor opción actual para generación directa de SVG"
        ],
        url="https://huggingface.co/datasets/svg-1m"
    )

    kb.add_paper(
        title="L3D - Large Labelled Logo Dataset",
        authors="EUIPO Registry (2024)",
        summary="770K logos profesionales del registro europeo EUIPO. Formato PNG 256x256 con clasificación Vienna.",
        key_findings=[
            "770K logos de marcas reales registradas",
            "Mejor para fine-tuning de modelos de difusión",
            "Clasificación Vienna (taxonomía profesional)",
            "Calidad profesional garantizada (registro oficial)"
        ],
        url="https://euipo.europa.eu"
    )

    kb.add_paper(
        title="SVG-Icons8: DeepSVG Dataset",
        authors="Carlier et al. (NeurIPS 2020)",
        summary="100K iconos en formato SVG tensor para investigación. Paper NeurIPS 2020 sobre generación jerárquica de SVG.",
        key_findings=[
            "100K iconos vectoriales en formato tensor",
            "Arquitectura VAE para latent space de SVG",
            "Excelente para investigación académica",
            "Base del paper DeepSVG (altamente citado)"
        ],
        url="https://github.com/alexandre01/deepsvg"
    )


def add_prompt_engineering(kb: SVGKnowledgeBase):
    """Agrega técnicas de prompt engineering"""

    kb.add_technique(
        name="Drawing-with-Thought (DwT)",
        description="Paradigma de 6 etapas para generación de SVG: Concept → Rationale → Structure → Geometric → SVG Code → Validation. Del paper Reason-SVG.",
        category="Prompt-Engineering",
        difficulty="Medium",
        use_cases=[
            "Generación de logos complejos con razonamiento",
            "Mejorar coherencia geométrica",
            "Explicabilidad del proceso de diseño",
            "Logos que requieren justificación conceptual"
        ]
    )

    kb.add_technique(
        name="Chain-of-Thought for SVG",
        description="Mejora 17.8% accuracy vs generación directa. Variantes: CD-CoT (concept-driven), DD-CoT (detail-driven). Mejor con 3+ ejemplos (few-shot).",
        category="Prompt-Engineering",
        difficulty="Low",
        use_cases=[
            "Mejorar calidad de SVG generado",
            "Reducir errores de sintaxis",
            "Logos con requisitos complejos",
            "Iteración rápida con feedback"
        ]
    )

    kb.add_technique(
        name="Few-Shot SVG Examples",
        description="Incluir 2-3 ejemplos de SVG en el prompt mejora precisión en +28% vs zero-shot. Los ejemplos deben ser similares en complejidad al target.",
        category="Prompt-Engineering",
        difficulty="Low",
        use_cases=[
            "Establecer estilo consistente",
            "Guiar complejidad del output",
            "Enseñar patrones específicos de SVG",
            "Mejorar validez del código generado"
        ]
    )

    kb.add_technique(
        name="Semantic SVG Tokens",
        description="Sistema de 55 tokens: 15 tags SVG, 30 atributos clave, 10 comandos path. Usado en LLM4SVG (CVPR 2025) para 89.7% validity.",
        category="Prompt-Engineering",
        difficulty="High",
        use_cases=[
            "Fine-tuning de LLMs para SVG",
            "Constrained generation",
            "Maximizar validez del código",
            "Sistemas de producción robustos"
        ]
    )


def add_advanced_models(kb: SVGKnowledgeBase):
    """Agrega modelos avanzados de la investigación"""

    kb.add_model(
        name="LLM4SVG",
        description="Sistema de CVPR 2025 con 55 tokens semánticos especializados. Dataset de 250K SVGs. Logra 89.7% validity en generación.",
        capabilities=[
            "Generación de SVG con alta validez",
            "Sistema de tokens especializados",
            "Fine-tuning específico para SVG",
            "Understanding, editing, generation"
        ],
        limitations=[
            "Requiere fine-tuning costoso",
            "Dataset propietario de 250K ejemplos",
            "No disponible comercialmente aún",
            "Paper reciente (CVPR 2025)"
        ],
        implementation="Research paper, código pendiente de release"
    )

    kb.add_model(
        name="OmniSVG v2",
        description="Versión 2025 con 2M SVGs anotados (MMSVG-2M). Generación end-to-end con VLMs pre-entrenados. NeurIPS 2025.",
        capabilities=[
            "Dataset masivo: 2 millones de SVGs",
            "Multi-modal: texto, imagen, sketch",
            "Transferencia desde VLMs grandes",
            "State-of-the-art en diversidad"
        ],
        limitations=[
            "Dataset MMSVG-2M aún no público",
            "Requiere GPUs potentes (A100+)",
            "Costs altos de inferencia",
            "Licencia académica por ahora"
        ],
        implementation="Research paper NeurIPS 2025"
    )

    kb.add_model(
        name="Claude 3.7 Sonnet",
        description="Líder actual en SVG generation según SVGenius benchmark. 87.3% understanding, 81.2% editing, 76.4% generation. Disponible comercialmente.",
        capabilities=[
            "Mejor modelo comercial disponible HOY",
            "Excellent chain-of-thought reasoning",
            "API accesible vía Anthropic",
            "Acepta few-shot prompting"
        ],
        limitations=[
            "No especializado en SVG (modelo general)",
            "54% accuracy en logos complejos",
            "Costo: $3/M input tokens",
            "Requiere prompt engineering cuidadoso"
        ],
        implementation="Disponible comercialmente: https://anthropic.com"
    )


def main():
    """Actualiza la base de conocimiento con todos los nuevos hallazgos"""
    print("="*60)
    print("Actualizando base de conocimiento con investigación avanzada")
    print("="*60)

    kb = SVGKnowledgeBase(persist_directory="../data/chroma_db")

    print("\n🎨 Agregando principios de diseño profesional...")
    add_design_principles(kb)

    print("\n📊 Agregando información de datasets...")
    add_datasets_info(kb)

    print("\n🔧 Agregando técnicas de prompt engineering...")
    add_prompt_engineering(kb)

    print("\n🤖 Agregando modelos avanzados...")
    add_advanced_models(kb)

    print("\n✅ Base de conocimiento actualizada!")
    stats = kb.get_stats()
    print(f"\nEstadísticas actualizadas:")
    print(f"  Papers:    {stats['papers']}")
    print(f"  Modelos:   {stats['models']}")
    print(f"  Técnicas:  {stats['techniques']}")
    print(f"  Total:     {sum(stats.values())} documentos")

    # Demo de búsqueda con nuevos datos
    print("\n" + "="*60)
    print("DEMO: Búsquedas con conocimiento actualizado")
    print("="*60)

    print("\n🔍 Buscar: 'golden ratio logo design'")
    results = kb.search_techniques("golden ratio logo design", n_results=2)
    for r in results[:2]:
        print(f"  • {r['metadata'].get('name', 'N/A')}")

    print("\n🔍 Buscar: 'best dataset for training logos'")
    results = kb.search_papers("best dataset for training logos", n_results=2)
    for r in results[:2]:
        print(f"  • {r['metadata'].get('title', 'N/A')}")

    print("\n🔍 Buscar: 'chain of thought prompting SVG'")
    results = kb.search_techniques("chain of thought prompting SVG", n_results=2)
    for r in results[:2]:
        print(f"  • {r['metadata'].get('name', 'N/A')}")

    print("\n🔍 Buscar: 'best commercial model available today'")
    results = kb.search_models("best commercial model available today", n_results=2)
    for r in results[:2]:
        print(f"  • {r['metadata'].get('name', 'N/A')}")


if __name__ == "__main__":
    main()
