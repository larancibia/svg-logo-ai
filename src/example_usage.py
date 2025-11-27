"""
Ejemplo de uso de la base de conocimiento para consultas específicas
"""

from knowledge_base import SVGKnowledgeBase


def demo_search_capabilities():
    """Demuestra las capacidades de búsqueda del sistema"""

    kb = SVGKnowledgeBase(persist_directory="../data/chroma_db")

    print("="*60)
    print("DEMO: Búsquedas en Base de Conocimiento SVG-AI")
    print("="*60)

    # Caso 1: Buscar modelos para producción
    print("\n🔍 CASO 1: ¿Qué modelos puedo usar en producción HOY?")
    print("-" * 60)
    results = kb.search_models("commercial available production ready", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['metadata'].get('name', 'N/A')}")
        print(f"   Implementación: {r['metadata'].get('has_implementation', 'N/A')}")
        print(f"   Relevancia: {r['distance']:.3f}")

    # Caso 2: Técnicas para logos geométricos
    print("\n\n🔍 CASO 2: ¿Qué técnicas sirven para logos geométricos?")
    print("-" * 60)
    results = kb.search_techniques("geometric simple logos", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['metadata'].get('name', 'N/A')}")
        print(f"   Categoría: {r['metadata'].get('category', 'N/A')}")
        print(f"   Dificultad: {r['metadata'].get('difficulty', 'N/A')}")

    # Caso 3: Papers sobre reasoning
    print("\n\n🔍 CASO 3: Papers sobre razonamiento en generación SVG")
    print("-" * 60)
    results = kb.search_papers("reasoning chain of thought", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['metadata'].get('title', 'N/A')}")
        print(f"   Autores: {r['metadata'].get('authors', 'N/A')}")
        print(f"   Key findings: {r['metadata'].get('key_findings_count', 0)} hallazgos")

    # Caso 4: Búsqueda completa sobre datasets
    print("\n\n🔍 CASO 4: Información completa sobre datasets")
    print("-" * 60)
    results = kb.search_all("large dataset millions examples", n_results=2)

    print("\nPapers:")
    for r in results['papers']:
        print(f"  - {r['metadata'].get('title', 'N/A')}")

    print("\nModelos:")
    for r in results['models']:
        print(f"  - {r['metadata'].get('name', 'N/A')}")

    # Estadísticas
    print("\n\n📊 ESTADÍSTICAS DE LA BASE DE CONOCIMIENTO")
    print("-" * 60)
    stats = kb.get_stats()
    print(f"Papers: {stats['papers']}")
    print(f"Modelos: {stats['models']}")
    print(f"Técnicas: {stats['techniques']}")
    print(f"TOTAL: {sum(stats.values())} documentos indexados")

    print("\n" + "="*60)
    print("Demo completado ✓")
    print("="*60)


def search_interactive():
    """Modo interactivo de búsqueda"""
    kb = SVGKnowledgeBase(persist_directory="../data/chroma_db")

    print("\n🤖 Modo Interactivo - Base de Conocimiento SVG-AI")
    print("Escribe 'salir' para terminar\n")

    while True:
        query = input("Tu consulta: ").strip()

        if query.lower() in ['salir', 'exit', 'quit']:
            print("¡Hasta luego!")
            break

        if not query:
            continue

        print("\n🔎 Buscando...\n")

        results = kb.search_all(query, n_results=2)

        print("📚 Papers relevantes:")
        for r in results['papers'][:2]:
            print(f"  • {r['metadata'].get('title', 'N/A')}")

        print("\n🤖 Modelos relevantes:")
        for r in results['models'][:2]:
            print(f"  • {r['metadata'].get('name', 'N/A')}")

        print("\n🛠️  Técnicas relevantes:")
        for r in results['techniques'][:2]:
            print(f"  • {r['metadata'].get('name', 'N/A')}")

        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        search_interactive()
    else:
        demo_search_capabilities()
