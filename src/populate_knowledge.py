"""
Script para poblar la base de conocimiento con información sobre generación SVG con IA
"""

from knowledge_base import SVGKnowledgeBase


def populate_research_papers(kb: SVGKnowledgeBase):
    """Agrega papers de investigación encontrados"""

    # RoboSVG
    kb.add_paper(
        title="RoboSVG: Unified Framework for Vector Graphic Generation",
        authors="Jiuniu Wang et al.",
        summary="Framework unificado que soporta texto, imagen y guía numérica para generación interactiva de gráficos vectoriales.",
        key_findings=[
            "Introduce el dataset RoboDraw con 1 millón de pares SVG-condición",
            "Soporta múltiples modalidades de entrada (texto, imagen, numérico)",
            "Permite generación interactiva y control fino",
            "Estado del arte en generación de íconos y gráficos vectoriales"
        ],
        url="https://arxiv.org/search/?query=RoboSVG"
    )

    # InternSVG
    kb.add_paper(
        title="InternSVG: Multimodal SVG Understanding, Editing and Generation",
        authors="Haomin Wang et al.",
        summary="Modelo multimodal que maneja comprensión, edición y generación de SVG en múltiples dominios.",
        key_findings=[
            "Cubre íconos, ilustraciones de secuencia larga, diagramas científicos y animaciones",
            "Capacidades multimodales de entrada y salida",
            "Edición estructurada de elementos SVG existentes",
            "Comprensión semántica de gráficos vectoriales"
        ],
        url="https://arxiv.org/search/?query=InternSVG"
    )

    # SVGThinker
    kb.add_paper(
        title="SVGThinker: Reasoning-Driven SVG Generation Framework",
        authors="Hanqi Chen et al.",
        summary="Framework basado en razonamiento que alinea la producción de código SVG con el proceso creativo de visualización.",
        key_findings=[
            "Implementa razonamiento chain-of-thought para SVG",
            "Mejora la precisión estructural del código generado",
            "Proceso creativo guiado por razonamiento explícito",
            "Superior en coherencia geométrica vs modelos directos"
        ],
        url="https://arxiv.org/search/?query=SVGThinker"
    )

    # Reason-SVG
    kb.add_paper(
        title="Reason-SVG: Drawing-with-Thought Paradigm",
        authors="Ximing Xing et al.",
        summary="Paradigma que combina rationales de diseño explícitos con generación de código SVG usando reinforcement learning.",
        key_findings=[
            "Usa funciones de recompensa híbridas",
            "Evalúa validez estructural y alineación semántica",
            "Genera rationales de diseño antes del código",
            "Approach de RL mejora calidad iterativamente"
        ],
        url="https://arxiv.org/search/?query=Reason-SVG"
    )

    # OmniSVG
    kb.add_paper(
        title="OmniSVG: Unified Framework with Pre-trained VLMs",
        authors="Yiying Yang et al.",
        summary="Framework unificado que aprovecha modelos de visión-lenguaje pre-entrenados para generación SVG.",
        key_findings=[
            "Introduce dataset MMSVG-2M con 2 millones de assets anotados",
            "Aprovecha VLMs pre-entrenados (Vision-Language Models)",
            "Transferencia de conocimiento desde modelos multimodales",
            "Datos masivos mejoran generalización"
        ],
        url="https://arxiv.org/search/?query=OmniSVG"
    )

    # SliDer
    kb.add_paper(
        title="SliDer: Semantic Document Derendering",
        authors="Adam Hazimeh, Ke Wang, Mark Collier et al.",
        summary="Convierte imágenes raster de slides en representaciones vectoriales editables preservando estructura semántica.",
        key_findings=[
            "Vision-Language Models para conversión raster-a-vector",
            "Preserva estructura semántica del documento",
            "Aplicación específica a presentaciones y documentos",
            "Mantiene editabilidad post-conversión"
        ],
        url="https://arxiv.org/search/?query=SliDer+semantic+document"
    )

    # SVGauge
    kb.add_paper(
        title="SVGauge: First Human-Aligned SVG Quality Metric",
        authors="Various researchers",
        summary="Primera métrica alineada con evaluación humana que combina fidelidad visual y consistencia semántica.",
        key_findings=[
            "Correlación alta con juicios humanos",
            "Combina fidelidad visual y semántica",
            "Benchmark para evaluación de modelos generativos",
            "Permite comparación objetiva entre sistemas"
        ]
    )


def populate_models(kb: SVGKnowledgeBase):
    """Agrega información sobre modelos específicos"""

    kb.add_model(
        name="RoboSVG",
        description="Framework unificado para generación de SVG con múltiples modalidades de entrada",
        capabilities=[
            "Generación desde texto descriptivo",
            "Generación desde imagen de referencia",
            "Guía numérica para control preciso",
            "Generación interactiva iterativa",
            "Entiende íconos y gráficos simples"
        ],
        limitations=[
            "Diseñado principalmente para íconos y gráficos simples",
            "Complejidad limitada en ilustraciones orgánicas",
            "Requiere dataset RoboDraw específico para entrenamiento"
        ],
        implementation="Disponible como research paper, implementación académica"
    )

    kb.add_model(
        name="InternSVG",
        description="Modelo multimodal completo para understanding, editing y generation de SVG",
        capabilities=[
            "Comprensión semántica de SVG existentes",
            "Edición estructurada de elementos",
            "Generación de novo",
            "Manejo de secuencias largas (ilustraciones complejas)",
            "Diagramas científicos y animaciones"
        ],
        limitations=[
            "Requiere entrenamiento multimodal extenso",
            "Complejidad computacional alta",
            "Modelo research, disponibilidad comercial incierta"
        ],
        implementation="Research model, paper disponible"
    )

    kb.add_model(
        name="SVGThinker",
        description="Framework de generación SVG basado en razonamiento chain-of-thought",
        capabilities=[
            "Razonamiento explícito sobre estructura",
            "Mejor coherencia geométrica",
            "Código SVG más limpio y estructurado",
            "Proceso creativo interpretable"
        ],
        limitations=[
            "Mayor latencia por etapa de reasoning",
            "Requiere LLM potente como base",
            "Puede ser over-engineered para logos simples"
        ],
        implementation="Research framework, approach replicable con LLMs actuales"
    )

    kb.add_model(
        name="OmniSVG",
        description="Aprovecha Vision-Language Models pre-entrenados con dataset masivo MMSVG-2M",
        capabilities=[
            "Generalización superior por dataset grande",
            "Transferencia desde VLMs como CLIP/GPT-4V",
            "Diversidad de estilos y dominios",
            "2 millones de ejemplos de entrenamiento"
        ],
        limitations=[
            "Requiere acceso a dataset MMSVG-2M",
            "VLMs subyacentes son caros (API costs)",
            "Datos de entrenamiento pueden no ser públicos"
        ],
        implementation="Research model con dataset propietario"
    )

    kb.add_model(
        name="Gemini Pro",
        description="LLM de Google con capacidades multimodales, puede generar código SVG como texto",
        capabilities=[
            "Generación de código SVG directo",
            "Razonamiento sobre geometría",
            "Múltiples iteraciones y refinamiento",
            "Disponible vía Vertex AI (GCP)",
            "Bueno para formas geométricas simples"
        ],
        limitations=[
            "No es específico para SVG (modelo general)",
            "Calidad variable en logos complejos",
            "Mejor en geometría que en orgánico",
            "Requiere prompting cuidadoso"
        ],
        implementation="Disponible comercialmente en GCP Vertex AI"
    )


def populate_techniques(kb: SVGKnowledgeBase):
    """Agrega técnicas y métodos"""

    kb.add_technique(
        name="Chain-of-Thought SVG Generation",
        description="Genera código SVG usando razonamiento paso a paso explícito antes de escribir el código.",
        category="Reasoning-Based",
        difficulty="Medium",
        use_cases=[
            "Logos con estructura geométrica compleja",
            "Diseños que requieren simetría o proporciones específicas",
            "Casos donde la explicabilidad es importante"
        ]
    )

    kb.add_technique(
        name="Multi-Modal Conditioning",
        description="Combina múltiples señales de entrada (texto, imagen, sketch) para guiar la generación.",
        category="Input-Fusion",
        difficulty="High",
        use_cases=[
            "Refinamiento iterativo desde boceto",
            "Transferencia de estilo desde imagen referencia",
            "Control fino con guías numéricas"
        ]
    )

    kb.add_technique(
        name="Reinforcement Learning with Design Rewards",
        description="Usa RL con funciones de recompensa que evalúan calidad de diseño, no solo similitud visual.",
        category="RL-Based",
        difficulty="High",
        use_cases=[
            "Optimización de balance y composición",
            "Aprendizaje de principios de diseño",
            "Mejora iterativa de calidad estética"
        ]
    )

    kb.add_technique(
        name="VLM-to-SVG Direct Generation",
        description="Usa Vision-Language Models pre-entrenados para generar SVG directamente desde descripciones.",
        category="Direct-Generation",
        difficulty="Medium",
        use_cases=[
            "Prototipado rápido de conceptos",
            "Generación de íconos simples",
            "Variaciones rápidas de diseños base"
        ]
    )

    kb.add_technique(
        name="Semantic Structure Preservation",
        description="Mantiene jerarquía semántica y estructura lógica del SVG, no solo apariencia visual.",
        category="Structure-Aware",
        difficulty="High",
        use_cases=[
            "SVGs que necesitan ser editados posteriormente",
            "Logos con variantes (colores, tamaños)",
            "Diseños que deben animarse"
        ]
    )

    kb.add_technique(
        name="Geometric Primitive Composition",
        description="Construye diseños complejos combinando primitivas geométricas básicas (círculos, paths, polígonos).",
        category="Constructive",
        difficulty="Low",
        use_cases=[
            "Logos minimalistas geométricos",
            "Íconos de interfaz",
            "Diseños con estética modernista"
        ]
    )


def main():
    """Puebla la base de conocimiento completa"""
    print("=== Iniciando población de base de conocimiento ===\n")

    kb = SVGKnowledgeBase(persist_directory="../data/chroma_db")

    print("\n📚 Agregando papers de investigación...")
    populate_research_papers(kb)

    print("\n🤖 Agregando modelos de IA...")
    populate_models(kb)

    print("\n🛠️  Agregando técnicas y métodos...")
    populate_techniques(kb)

    print("\n✅ Base de conocimiento poblada exitosamente!")
    print(f"\nEstadísticas finales: {kb.get_stats()}")

    # Demo de búsqueda
    print("\n" + "="*50)
    print("DEMO: Búsqueda de ejemplo")
    print("="*50)

    query = "generar logos profesionales con IA"
    print(f"\nQuery: '{query}'\n")

    results = kb.search_all(query, n_results=2)

    print("Top 2 Papers:")
    for paper in results['papers']:
        print(f"  - {paper['metadata'].get('title', 'N/A')}")

    print("\nTop 2 Modelos:")
    for model in results['models']:
        print(f"  - {model['metadata'].get('name', 'N/A')}")

    print("\nTop 2 Técnicas:")
    for tech in results['techniques']:
        print(f"  - {tech['metadata'].get('name', 'N/A')}")


if __name__ == "__main__":
    main()
