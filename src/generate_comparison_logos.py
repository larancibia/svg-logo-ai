"""
Genera logos con diferentes técnicas para comparar en la galería
Simula diferentes approaches: v1 básico, v2 CoT, v2 con Golden Ratio
"""

from logo_metadata import LogoMetadata
from logo_validator import LogoValidator
from pathlib import Path


def generate_logos_comparison():
    """Genera conjunto de logos para comparación"""

    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)

    metadata = LogoMetadata()
    validator = LogoValidator()

    logos_to_generate = [
        # TechFlow - 3 iteraciones con diferentes técnicas
        {
            "filename": "techflow_v1_basic.svg",
            "company": "TechFlow",
            "industry": "Technology",
            "style": "minimalist",
            "version": "v1",
            "iteration": 1,
            "technique": "Zero-shot básico",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="50" fill="#2563eb"/>
  <rect x="85" y="85" width="30" height="30" fill="white"/>
</svg>""",
            "colors": ["#2563eb", "#ffffff"]
        },
        {
            "filename": "techflow_v2_cot.svg",
            "company": "TechFlow",
            "industry": "Technology",
            "style": "minimalist",
            "version": "v2",
            "iteration": 2,
            "technique": "Chain-of-Thought",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="none" stroke="#2563eb" stroke-width="8"/>
  <path d="M70 100 Q100 70 130 100" fill="none" stroke="#2563eb" stroke-width="6" stroke-linecap="round"/>
  <circle cx="70" cy="100" r="4" fill="#2563eb"/>
  <circle cx="130" cy="100" r="4" fill="#2563eb"/>
</svg>""",
            "colors": ["#2563eb"]
        },
        {
            "filename": "techflow_v2_golden.svg",
            "company": "TechFlow",
            "industry": "Technology",
            "style": "minimalist",
            "version": "v2",
            "iteration": 3,
            "technique": "Chain-of-Thought + Golden Ratio",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g transform="translate(100,100)">
    <!-- Outer circle: r=60 -->
    <circle r="60" fill="none" stroke="#2563eb" stroke-width="6"/>
    <!-- Inner circle: 60/1.618 = 37 (Golden Ratio) -->
    <circle r="37" fill="none" stroke="#3b82f6" stroke-width="4"/>
    <!-- Center dot: 37/1.618 = 23 -->
    <circle r="23" fill="#2563eb"/>
    <!-- Flow lines -->
    <path d="M-37 0 Q-20 -20 0 -23" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
    <path d="M37 0 Q20 20 0 23" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
  </g>
</svg>""",
            "colors": ["#2563eb", "#3b82f6"]
        },

        # HealthPlus - Comparación de complejidades
        {
            "filename": "healthplus_v1_simple.svg",
            "company": "HealthPlus",
            "industry": "Healthcare",
            "style": "modern",
            "version": "v1",
            "iteration": 1,
            "technique": "Simple generation",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="85" y="70" width="30" height="60" rx="5" fill="#10b981"/>
  <rect x="70" y="85" width="60" height="30" rx="5" fill="#10b981"/>
</svg>""",
            "colors": ["#10b981"]
        },
        {
            "filename": "healthplus_v2_gestalt.svg",
            "company": "HealthPlus",
            "industry": "Healthcare",
            "style": "modern",
            "version": "v2",
            "iteration": 2,
            "technique": "Gestalt Principles (Figure-Ground)",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Heart with cross using figure-ground -->
  <path d="M100 140 L70 110 Q70 80 85 80 Q100 80 100 95 Q100 80 115 80 Q130 80 130 110 Z" fill="#10b981"/>
  <rect x="85" y="70" width="30" height="10" rx="2" fill="#059669"/>
  <rect x="95" y="60" width="10" height="30" rx="2" fill="#059669"/>
  <circle cx="85" cy="90" r="3" fill="white" opacity="0.3"/>
  <circle cx="115" cy="90" r="3" fill="white" opacity="0.3"/>
</svg>""",
            "colors": ["#10b981", "#059669"]
        },

        # FinVest - Comparación de balance
        {
            "filename": "finvest_v1_basic.svg",
            "company": "FinVest",
            "industry": "Finance",
            "style": "professional",
            "version": "v1",
            "iteration": 1,
            "technique": "Basic geometric",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <polygon points="100,50 150,150 50,150" fill="#1e40af"/>
</svg>""",
            "colors": ["#1e40af"]
        },
        {
            "filename": "finvest_v2_balance.svg",
            "company": "FinVest",
            "industry": "Finance",
            "style": "professional",
            "version": "v2",
            "iteration": 2,
            "technique": "Symmetrical Balance + Structure",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <polygon points="100,60 160,140 40,140" fill="none" stroke="#1e3a8a" stroke-width="6"/>
  <line x1="100" y1="80" x2="100" y2="130" stroke="#1e40af" stroke-width="4"/>
  <line x1="80" y1="110" x2="120" y2="110" stroke="#1e40af" stroke-width="4"/>
  <circle cx="100" cy="80" r="5" fill="#1e40af"/>
  <circle cx="80" cy="110" r="5" fill="#1e40af"/>
  <circle cx="120" cy="110" r="5" fill="#1e40af"/>
</svg>""",
            "colors": ["#1e3a8a", "#1e40af"]
        },

        # DataFlow - AI/ML company
        {
            "filename": "dataflow_v2_network.svg",
            "company": "DataFlow",
            "industry": "AI/ML",
            "style": "geometric",
            "version": "v2",
            "iteration": 1,
            "technique": "Network topology + Golden Ratio",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g transform="translate(100,100)">
    <!-- Central node -->
    <circle r="16" fill="#6366f1"/>
    <!-- Outer nodes with golden ratio spacing: 60 -->
    <circle cx="0" cy="-60" r="10" fill="#8b5cf6"/>
    <circle cx="60" cy="0" r="10" fill="#8b5cf6"/>
    <circle cx="0" cy="60" r="10" fill="#8b5cf6"/>
    <circle cx="-60" cy="0" r="10" fill="#8b5cf6"/>
    <!-- Connections -->
    <line x1="0" y1="-16" x2="0" y2="-50" stroke="#6366f1" stroke-width="3"/>
    <line x1="16" y1="0" x2="50" y2="0" stroke="#6366f1" stroke-width="3"/>
    <line x1="0" y1="16" x2="0" y2="50" stroke="#6366f1" stroke-width="3"/>
    <line x1="-16" y1="0" x2="-50" y2="0" stroke="#6366f1" stroke-width="3"/>
  </g>
</svg>""",
            "colors": ["#6366f1", "#8b5cf6"]
        },

        # CoffeeHub - Food industry
        {
            "filename": "coffeehub_v1_simple.svg",
            "company": "CoffeeHub",
            "industry": "Food/Coffee",
            "style": "friendly",
            "version": "v1",
            "iteration": 1,
            "technique": "Simple shapes",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <ellipse cx="100" cy="110" rx="40" ry="35" fill="#92400e"/>
  <rect x="140" y="100" width="15" height="30" rx="7" fill="none" stroke="#92400e" stroke-width="3"/>
</svg>""",
            "colors": ["#92400e"]
        },
        {
            "filename": "coffeehub_v2_detailed.svg",
            "company": "CoffeeHub",
            "industry": "Food/Coffee",
            "style": "friendly",
            "version": "v2",
            "iteration": 2,
            "technique": "Gestalt Continuation + Details",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Cup -->
  <path d="M60 100 Q60 130 100 130 Q140 130 140 100 L140 80 L60 80 Z" fill="#92400e" stroke="#78350f" stroke-width="3"/>
  <rect x="140" y="90" width="15" height="25" rx="8" fill="none" stroke="#78350f" stroke-width="3"/>
  <!-- Steam with continuation principle -->
  <path d="M80 70 Q80 55 85 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
  <path d="M100 70 Q100 55 105 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
  <path d="M120 70 Q120 55 125 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
  <!-- Saucer -->
  <ellipse cx="100" cy="132" rx="50" ry="8" fill="#78350f" opacity="0.3"/>
</svg>""",
            "colors": ["#92400e", "#78350f"]
        },

        # RetailHub - E-commerce
        {
            "filename": "retailhub_v2_minimal.svg",
            "company": "RetailHub",
            "industry": "E-commerce",
            "style": "modern",
            "version": "v2",
            "iteration": 1,
            "technique": "Minimalist + Color Psychology",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="70" y="85" width="60" height="70" rx="4" fill="#7c3aed" stroke="#6d28d9" stroke-width="3"/>
  <path d="M80 85 Q80 60 100 60 Q120 60 120 85" fill="none" stroke="#6d28d9" stroke-width="3"/>
</svg>""",
            "colors": ["#7c3aed", "#6d28d9"]
        },

        # GreenLeaf - Wellness con complejidad óptima
        {
            "filename": "greenleaf_v2_optimal.svg",
            "company": "GreenLeaf",
            "industry": "Wellness",
            "style": "organic",
            "version": "v2",
            "iteration": 1,
            "technique": "Optimal Complexity (20-40 target)",
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g transform="translate(100,100)">
    <!-- Circle base -->
    <circle r="55" fill="#34d399"/>
    <!-- Leaf shape using golden ratio proportions -->
    <path d="M0 -35 Q22 -18 18 5 Q14 28 0 50 Q-14 28 -18 5 Q-22 -18 0 -35" fill="#065f46"/>
    <!-- Veins -->
    <line x1="0" y1="-35" x2="0" y2="50" stroke="#34d399" stroke-width="2"/>
    <path d="M0 -10 Q8 -5 10 0" fill="none" stroke="#34d399" stroke-width="1.5"/>
    <path d="M0 -10 Q-8 -5 -10 0" fill="none" stroke="#34d399" stroke-width="1.5"/>
    <path d="M0 10 Q8 15 10 20" fill="none" stroke="#34d399" stroke-width="1.5"/>
    <path d="M0 10 Q-8 15 -10 20" fill="none" stroke="#34d399" stroke-width="1.5"/>
  </g>
</svg>""",
            "colors": ["#34d399", "#065f46"]
        },
    ]

    print("="*70)
    print("🎨 GENERANDO LOGOS PARA COMPARACIÓN")
    print("="*70)
    print(f"\nTotal a generar: {len(logos_to_generate)} logos")
    print("Técnicas: v1 básico, v2 CoT, v2 Golden Ratio, v2 Gestalt\n")

    for logo_data in logos_to_generate:
        print(f"\n{'─'*70}")
        print(f"🔨 {logo_data['company']} - {logo_data['technique']}")
        print(f"{'─'*70}")

        # Guardar SVG
        svg_path = output_dir / logo_data['filename']
        with open(svg_path, 'w') as f:
            f.write(logo_data['svg'])
        print(f"   ✓ SVG guardado: {logo_data['filename']}")

        # Validar
        validation = validator.validate_all(logo_data['svg'])
        score = validation['final_score']
        complexity = validation['level3_quality']['complexity']

        print(f"   ✓ Score: {score}/100")
        print(f"   ✓ Complejidad: {complexity}")
        print(f"   ✓ Técnica: {logo_data['technique']}")

        # Agregar a metadata
        is_favorite = score >= 85  # Auto-marcar los mejores

        logo_id = metadata.add_logo(
            filename=logo_data['filename'],
            company_name=logo_data['company'],
            industry=logo_data['industry'],
            style=logo_data['style'],
            score=score,
            complexity=complexity,
            version=logo_data['version'],
            iteration=logo_data['iteration'],
            colors=logo_data['colors'],
            notes=f"Técnica: {logo_data['technique']}",
            is_favorite=is_favorite,
            validation_results=validation
        )

        fav_marker = "⭐" if is_favorite else ""
        print(f"   ✓ Metadata guardada {fav_marker}")
        print(f"   ✓ ID: {logo_id}")

    # Estadísticas finales
    print(f"\n{'='*70}")
    print("📊 ESTADÍSTICAS FINALES")
    print(f"{'='*70}")

    stats = metadata.get_stats()
    print(f"\nTotal logos generados: {stats['total']}")
    print(f"Score promedio: {stats['avg_score']:.1f}/100")
    print(f"Mejor score: {stats['max_score']}")
    print(f"Favoritos automáticos: {stats['favorites']}")

    print(f"\nPor versión:")
    for ver, data in stats['by_version'].items():
        print(f"  {ver}: {data['count']} logos (avg: {data['avg_score']:.1f})")

    print(f"\nPor industria:")
    for ind, data in stats['by_industry'].items():
        print(f"  {ind}: {data['count']} logos (avg: {data['avg_score']:.1f})")

    # Comparación v1 vs v2
    v1_logos = metadata.get_logos(version='v1')
    v2_logos = metadata.get_logos(version='v2')

    if v1_logos and v2_logos:
        v1_avg = sum(l['score'] for l in v1_logos) / len(v1_logos)
        v2_avg = sum(l['score'] for l in v2_logos) / len(v2_logos)
        improvement = ((v2_avg - v1_avg) / v1_avg * 100) if v1_avg > 0 else 0

        print(f"\n{'─'*70}")
        print("📈 COMPARACIÓN V1 vs V2")
        print(f"{'─'*70}")
        print(f"v1 promedio: {v1_avg:.1f}/100")
        print(f"v2 promedio: {v2_avg:.1f}/100")
        print(f"Mejora: {v2_avg - v1_avg:+.1f} puntos ({improvement:+.1f}%)")

    print(f"\n{'='*70}")
    print("✅ GENERACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"\nPróximo paso:")
    print(f"  1. Regenerar galería: ./run.sh gallery")
    print(f"  2. Re-deploy: sudo cp output/deploy/* /var/www/logos.guanacolabs.com/")
    print(f"  3. Ver en: https://logos.guanacolabs.com")


if __name__ == "__main__":
    generate_logos_comparison()
