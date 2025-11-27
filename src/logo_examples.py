"""
Biblioteca de ejemplos de logos profesionales para few-shot prompting
Organizados por industria y estilo
"""

LOGO_EXAMPLES = {
    "tech_minimalist": [
        {
            "description": "Tech startup logo - círculo con línea dinámica",
            "industry": "Technology",
            "style": "minimalist",
            "complexity": 25,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="none" stroke="#2563eb" stroke-width="8"/>
  <path d="M70 100 Q100 70 130 100" fill="none" stroke="#2563eb" stroke-width="6" stroke-linecap="round"/>
</svg>""",
            "rationale": "Círculo = continuidad y completitud. Línea curva = dinamismo y movimiento. Proporción golden ratio en radios."
        },
        {
            "description": "Software company - hexágono geométrico",
            "industry": "Technology",
            "style": "geometric",
            "complexity": 30,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g transform="translate(100,100)">
    <polygon points="0,-60 52,-30 52,30 0,60 -52,30 -52,-30" fill="#1e40af"/>
    <polygon points="0,-40 35,-20 35,20 0,40 -35,20 -35,-20" fill="#3b82f6"/>
  </g>
</svg>""",
            "rationale": "Hexágono = estructura y estabilidad técnica. Dos capas = profundidad. Simetría radial = balance perfecto."
        },
        {
            "description": "AI company - nodos conectados",
            "industry": "AI/ML",
            "style": "minimalist",
            "complexity": 35,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <line x1="60" y1="100" x2="140" y2="100" stroke="#6366f1" stroke-width="4"/>
  <line x1="100" y1="60" x2="100" y2="140" stroke="#6366f1" stroke-width="4"/>
  <circle cx="60" cy="100" r="12" fill="#6366f1"/>
  <circle cx="140" cy="100" r="12" fill="#6366f1"/>
  <circle cx="100" cy="60" r="12" fill="#6366f1"/>
  <circle cx="100" cy="140" r="12" fill="#6366f1"/>
  <circle cx="100" cy="100" r="16" fill="#8b5cf6"/>
</svg>""",
            "rationale": "Red de nodos = conexión e inteligencia distribuida. Centro más grande = núcleo central. Líneas limpias = claridad."
        }
    ],

    "health_modern": [
        {
            "description": "Healthcare - cruz moderna con corazón",
            "industry": "Healthcare",
            "style": "modern",
            "complexity": 32,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="M100 140 L70 110 Q70 80 85 80 Q100 80 100 95 Q100 80 115 80 Q130 80 130 110 Z" fill="#10b981"/>
  <rect x="85" y="70" width="30" height="10" rx="2" fill="#059669"/>
  <rect x="95" y="60" width="10" height="30" rx="2" fill="#059669"/>
</svg>""",
            "rationale": "Corazón = cuidado y empatía. Cruz = medicina. Formas redondeadas = amigable. Verde = salud y vida."
        },
        {
            "description": "Wellness app - hoja con círculo",
            "industry": "Health/Wellness",
            "style": "organic",
            "complexity": 28,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="50" fill="#34d399"/>
  <path d="M100 60 Q120 75 115 95 Q110 115 100 140 Q90 115 85 95 Q80 75 100 60" fill="#065f46"/>
</svg>""",
            "rationale": "Círculo = ciclo y equilibrio. Hoja = naturaleza y crecimiento. Combinación = wellness holístico."
        }
    ],

    "finance_professional": [
        {
            "description": "Financial services - triángulo con líneas",
            "industry": "Finance",
            "style": "professional",
            "complexity": 30,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <polygon points="100,60 160,140 40,140" fill="none" stroke="#1e3a8a" stroke-width="6"/>
  <line x1="100" y1="80" x2="100" y2="130" stroke="#1e40af" stroke-width="4"/>
  <line x1="80" y1="110" x2="120" y2="110" stroke="#1e40af" stroke-width="4"/>
</svg>""",
            "rationale": "Triángulo = estabilidad y crecimiento. Líneas internas = estructura y orden. Azul oscuro = confianza."
        },
        {
            "description": "Investment firm - escalera ascendente",
            "industry": "Investment",
            "style": "minimal",
            "complexity": 25,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <polyline points="50,150 80,120 110,110 140,80 170,50" fill="none" stroke="#1e40af" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="50" cy="150" r="6" fill="#1e40af"/>
  <circle cx="80" cy="120" r="6" fill="#1e40af"/>
  <circle cx="110" cy="110" r="6" fill="#1e40af"/>
  <circle cx="140" cy="80" r="6" fill="#1e40af"/>
  <circle cx="170" cy="50" r="6" fill="#1e40af"/>
</svg>""",
            "rationale": "Línea ascendente = crecimiento de inversión. Puntos = hitos. Continuidad = progreso sostenido."
        }
    ],

    "food_energetic": [
        {
            "description": "Restaurant - fork y knife cruzados",
            "industry": "Food/Restaurant",
            "style": "simple",
            "complexity": 28,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g transform="translate(100,100) rotate(45)">
    <rect x="-4" y="-50" width="8" height="100" rx="2" fill="#dc2626"/>
    <circle cx="0" cy="-35" r="3" fill="#dc2626"/>
    <circle cx="0" cy="-25" r="3" fill="#dc2626"/>
    <circle cx="0" cy="-15" r="3" fill="#dc2626"/>
  </g>
  <g transform="translate(100,100) rotate(-45)">
    <rect x="-4" y="-50" width="8" height="100" rx="2" fill="#dc2626"/>
    <path d="M-10,-35 L-10,-15 L-4,-15 L-4,-45 Z" fill="#dc2626"/>
    <path d="M4,-45 L4,-15 L10,-15 L10,-35 Z" fill="#dc2626"/>
  </g>
</svg>""",
            "rationale": "Tenedor y cuchillo = comida. Cruzados = servicio completo. Rojo = apetito y energía."
        },
        {
            "description": "Coffee shop - taza con vapor",
            "industry": "Coffee/Café",
            "style": "friendly",
            "complexity": 30,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="M60 100 Q60 130 100 130 Q140 130 140 100 L140 80 L60 80 Z" fill="#92400e" stroke="#78350f" stroke-width="3"/>
  <rect x="140" y="90" width="15" height="25" rx="8" fill="none" stroke="#78350f" stroke-width="3"/>
  <path d="M80 70 Q80 55 85 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
  <path d="M100 70 Q100 55 105 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
  <path d="M120 70 Q120 55 125 50" fill="none" stroke="#92400e" stroke-width="2" stroke-linecap="round"/>
</svg>""",
            "rationale": "Taza = café obvio. Vapor en curvas = aroma y calidez. Marrón = café. Formas suaves = acogedor."
        }
    ],

    "retail_modern": [
        {
            "description": "E-commerce - shopping bag",
            "industry": "Retail/E-commerce",
            "style": "modern",
            "complexity": 26,
            "svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="70" y="85" width="60" height="70" rx="4" fill="#7c3aed" stroke="#6d28d9" stroke-width="3"/>
  <path d="M80 85 Q80 60 100 60 Q120 60 120 85" fill="none" stroke="#6d28d9" stroke-width="3"/>
</svg>""",
            "rationale": "Bolsa de compras = retail obvio. Asa redondeada = ergonómico. Púrpura = premium y moderno."
        }
    ]
}


def get_examples_by_industry(industry: str, n: int = 2):
    """Obtiene ejemplos relevantes por industria"""
    examples = []

    # Mapeo de industrias a categorías
    industry_map = {
        "technology": "tech_minimalist",
        "software": "tech_minimalist",
        "ai": "tech_minimalist",
        "ml": "tech_minimalist",
        "healthcare": "health_modern",
        "health": "health_modern",
        "wellness": "health_modern",
        "finance": "finance_professional",
        "fintech": "finance_professional",
        "banking": "finance_professional",
        "investment": "finance_professional",
        "food": "food_energetic",
        "restaurant": "food_energetic",
        "coffee": "food_energetic",
        "retail": "retail_modern",
        "ecommerce": "retail_modern",
        "shopping": "retail_modern"
    }

    industry_lower = industry.lower()
    category = None

    for key, cat in industry_map.items():
        if key in industry_lower:
            category = cat
            break

    if category and category in LOGO_EXAMPLES:
        examples = LOGO_EXAMPLES[category][:n]

    # Si no hay match, usar ejemplos tech como default
    if not examples:
        examples = LOGO_EXAMPLES["tech_minimalist"][:n]

    return examples


def format_examples_for_prompt(examples):
    """Formatea ejemplos para incluir en el prompt"""
    formatted = []

    for i, ex in enumerate(examples, 1):
        formatted.append(f"""
### Ejemplo {i}: {ex['description']}
**Industria:** {ex['industry']}
**Estilo:** {ex['style']}
**Complejidad:** {ex['complexity']} (óptimo: 20-40)

**Razonamiento de Diseño:**
{ex['rationale']}

**Código SVG:**
```xml
{ex['svg']}
```
""")

    return "\n".join(formatted)


if __name__ == "__main__":
    # Demo
    print("Biblioteca de Ejemplos de Logos\n")

    for category, examples in LOGO_EXAMPLES.items():
        print(f"\n{category.upper()}: {len(examples)} ejemplos")
        for ex in examples:
            print(f"  • {ex['description']} (complejidad: {ex['complexity']})")

    print("\n" + "="*60)
    print("\nDemo: Obtener ejemplos para 'healthcare'")
    examples = get_examples_by_industry("healthcare", 2)
    print(format_examples_for_prompt(examples))
