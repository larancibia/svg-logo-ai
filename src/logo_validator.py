"""
Sistema de validación multi-nivel y scoring para logos SVG
Basado en principios profesionales de diseño

VERSIÓN 2.0: Incluye métricas estéticas (Golden Ratio, Color Harmony, Visual Interest)
Nuevo sistema de ponderación: 50% Aesthetic + 35% Professional + 15% Technical
"""

import re
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
from colorsys import rgb_to_hsv


class AestheticMetrics:
    """
    Métricas estéticas profesionales para evaluación de logos
    Incluye: Golden Ratio, Color Harmony, Visual Interest
    """

    PHI = 1.618033988749895  # Golden ratio
    TOLERANCE = 0.15  # 15% tolerance for golden ratio detection

    @staticmethod
    def extract_dimensions(svg_code: str) -> List[float]:
        """Extrae todas las dimensiones numéricas del SVG"""
        numbers = []
        pattern = r'(\d+\.?\d*)'
        matches = re.findall(pattern, svg_code)

        for match in matches:
            try:
                num = float(match)
                if 0 < num < 1000:  # Rango razonable
                    numbers.append(num)
            except ValueError:
                continue

        return numbers

    @classmethod
    def calculate_golden_ratio_score(cls, svg_code: str) -> float:
        """
        Detecta uso del Golden Ratio en proporciones
        Score: 0-100 basado en frecuencia de ratios φ (1.618)
        """
        numbers = cls.extract_dimensions(svg_code)

        if len(numbers) < 2:
            return 50.0  # Neutral si no hay suficientes números

        golden_ratios_found = 0
        total_comparisons = 0

        for i, num1 in enumerate(numbers):
            for num2 in numbers[i+1:]:
                if num2 < 1:  # Skip números muy pequeños
                    continue

                ratio = max(num1, num2) / min(num1, num2)

                # ¿Está cerca del golden ratio?
                if abs(ratio - cls.PHI) / cls.PHI < cls.TOLERANCE:
                    golden_ratios_found += 1

                total_comparisons += 1

        if total_comparisons == 0:
            return 50.0

        # Porcentaje de golden ratios encontrados
        percentage = golden_ratios_found / total_comparisons

        # Escalar a 0-100 (50 = baseline, +50 por uso de φ)
        score = min(100, 50 + percentage * 200)

        return round(score, 1)

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convierte #RRGGBB a (r, g, b) en rango 0-1"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])

        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        return (r, g, b)

    @classmethod
    def extract_colors(cls, svg_code: str) -> List[Dict]:
        """Extrae todos los colores hex del SVG con info RGB/HSV"""
        pattern = r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})'
        matches = set(re.findall(pattern, svg_code))

        colors = []
        for match in matches:
            # Skip white (no aporta a la armonía)
            if match.lower() in ['fff', 'ffffff']:
                continue

            hex_color = f"#{match}"
            rgb = cls.hex_to_rgb(hex_color)
            hsv = rgb_to_hsv(*rgb)

            colors.append({
                'hex': hex_color,
                'rgb': rgb,
                'hsv': hsv
            })

        return colors

    @classmethod
    def calculate_color_harmony(cls, svg_code: str) -> float:
        """
        Evalúa armonía de la paleta de colores
        Detecta: Monocromático, Complementario, Análogo, Triádico
        Score: 0-100
        """
        colors = cls.extract_colors(svg_code)

        if len(colors) == 0:
            return 50.0  # Neutral si no hay colores

        if len(colors) == 1:
            return 95.0  # Monocromático = muy armónico

        # Extraer hues (0-360 grados)
        hues = [c['hsv'][0] * 360 for c in colors]

        # 1. Complementario (2 colores, ~180° aparte)
        if len(colors) == 2:
            diff = abs(hues[0] - hues[1])
            diff = min(diff, 360 - diff)  # Handle wraparound

            if 165 < diff < 195:
                return 95.0  # Perfect complementary
            elif 150 < diff < 210:
                return 85.0  # Close to complementary

        # 2. Análogo (< 60° de rango)
        hue_range = max(hues) - min(hues)
        if hue_range < 60:
            return 90.0  # Analogous harmony

        # 3. Triádico (3 colores, ~120° aparte)
        if len(colors) == 3:
            sorted_hues = sorted(hues)
            diff1 = sorted_hues[1] - sorted_hues[0]
            diff2 = sorted_hues[2] - sorted_hues[1]
            diff3 = (360 - sorted_hues[2]) + sorted_hues[0]

            if all(100 < d < 140 for d in [diff1, diff2, diff3]):
                return 95.0  # Triadic

        # Sin armonía clara detectada
        return 60.0

    @classmethod
    def calculate_visual_interest(cls, svg_code: str) -> float:
        """
        Mide interés visual basado en variedad de elementos
        Score: 0-100 (más variedad = más interesante)
        """
        try:
            tree = ET.fromstring(svg_code)

            # Contar tipos de elementos diferentes
            element_types = set()
            for elem in tree.iter():
                tag = elem.tag.split('}')[-1]  # Remove namespace
                if tag in ['circle', 'rect', 'ellipse', 'line', 'polyline', 'polygon', 'path']:
                    element_types.add(tag)

            # Más variedad = más interesante
            variety_score = min(100, len(element_types) * 20 + 40)

            # Bonus: comentarios (indica diseño pensado)
            has_comments = '<!--' in svg_code
            comment_bonus = 10 if has_comments else 0

            # Bonus: transformaciones (más sofisticado)
            has_transforms = 'transform=' in svg_code
            transform_bonus = 10 if has_transforms else 0

            return float(min(100, variety_score + comment_bonus + transform_bonus))

        except Exception:
            return 50.0  # Neutral en caso de error


class LogoValidator:
    """Validación técnica y de calidad para logos SVG con métricas estéticas"""

    def __init__(self):
        self.namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        self.aesthetic_metrics = AestheticMetrics()

    def validate_all(self, svg_code: str) -> Dict:
        """
        Validación completa en 5 niveles (incluye métricas estéticas)

        Returns:
            Dict con resultados de cada nivel + scores (final y legacy)
        """
        results = {
            'level1_xml': self.validate_xml(svg_code),
            'level2_svg': self.validate_svg_structure(svg_code),
            'level3_quality': self.evaluate_quality(svg_code),
            'level4_professional': self.evaluate_professional_standards(svg_code),
            'level5_aesthetic': self.evaluate_aesthetic_metrics(svg_code)
        }

        # Score final con NUEVAS métricas estéticas (v2.0)
        results['final_score'] = self._calculate_final_score(results)

        # Legacy score (para comparación con sistema anterior)
        results['legacy_score'] = self._calculate_legacy_score(results)

        results['passed'] = results['final_score'] >= 70

        return results

    def validate_xml(self, svg_code: str) -> Dict:
        """Nivel 1: Validación de sintaxis XML"""
        result = {
            'valid': False,
            'error': None,
            'score': 0
        }

        try:
            ET.fromstring(svg_code)
            result['valid'] = True
            result['score'] = 100
        except ET.ParseError as e:
            result['error'] = str(e)

        return result

    def validate_svg_structure(self, svg_code: str) -> Dict:
        """Nivel 2: Validación de estructura SVG"""
        result = {
            'has_svg_root': False,
            'has_viewbox': False,
            'has_xmlns': False,
            'elements_count': 0,
            'warnings': [],
            'score': 0
        }

        try:
            root = ET.fromstring(svg_code)

            # Check SVG root
            if root.tag.endswith('svg'):
                result['has_svg_root'] = True

            # Check viewBox
            if root.get('viewBox'):
                result['has_viewbox'] = True
            else:
                result['warnings'].append("Missing viewBox (scalability issue)")

            # Check xmlns
            if 'xmlns' in root.attrib:
                result['has_xmlns'] = True

            # Count elements
            result['elements_count'] = len(list(root.iter()))

            # Calculate score
            checks_passed = sum([
                result['has_svg_root'],
                result['has_viewbox'],
                result['has_xmlns']
            ])
            result['score'] = int((checks_passed / 3) * 100)

        except Exception as e:
            result['warnings'].append(f"Parse error: {e}")

        return result

    def evaluate_quality(self, svg_code: str) -> Dict:
        """Nivel 3: Evaluación de calidad técnica"""
        result = {
            'complexity': 0,
            'complexity_rating': 'unknown',
            'precision_ok': True,
            'has_comments': False,
            'has_ids': False,
            'color_count': 0,
            'warnings': [],
            'score': 0
        }

        try:
            root = ET.fromstring(svg_code)

            # Complexity (count geometric elements)
            geometric_tags = ['circle', 'rect', 'ellipse', 'line', 'polyline', 'polygon', 'path']
            complexity = sum(1 for elem in root.iter() if any(elem.tag.endswith(tag) for tag in geometric_tags))
            result['complexity'] = complexity

            # Complexity rating
            if complexity < 20:
                result['complexity_rating'] = 'ultra_minimal'
            elif 20 <= complexity <= 40:
                result['complexity_rating'] = 'optimal'
                result['warnings'].append("Complejidad óptima ✓")
            elif 40 < complexity <= 60:
                result['complexity_rating'] = 'moderate'
                result['warnings'].append("Complejidad moderada")
            else:
                result['complexity_rating'] = 'too_complex'
                result['warnings'].append("⚠ Demasiado complejo para logo")

            # Precision check (too many decimals = bloat)
            if re.search(r'\d+\.\d{4,}', svg_code):
                result['precision_ok'] = False
                result['warnings'].append("⚠ Precisión excesiva (>3 decimales)")

            # Comments
            if '<!--' in svg_code:
                result['has_comments'] = True

            # IDs (good for accessibility)
            if 'id=' in svg_code:
                result['has_ids'] = True

            # Color count
            colors = set(re.findall(r'#[0-9a-fA-F]{6}', svg_code))
            colors.update(re.findall(r'#[0-9a-fA-F]{3}', svg_code))
            result['color_count'] = len(colors)

            if len(colors) > 3:
                result['warnings'].append("⚠ Más de 3 colores")

            # Calculate score
            score_components = []

            # Complexity score (optimal = 100, degrading outside)
            if result['complexity_rating'] == 'optimal':
                score_components.append(100)
            elif result['complexity_rating'] == 'ultra_minimal':
                score_components.append(85)
            elif result['complexity_rating'] == 'moderate':
                score_components.append(70)
            else:
                score_components.append(40)

            # Precision score
            score_components.append(100 if result['precision_ok'] else 70)

            # Color score
            if result['color_count'] <= 3:
                score_components.append(100)
            else:
                score_components.append(max(60, 100 - (result['color_count'] - 3) * 10))

            result['score'] = int(sum(score_components) / len(score_components))

        except Exception as e:
            result['warnings'].append(f"Evaluation error: {e}")

        return result

    def evaluate_professional_standards(self, svg_code: str) -> Dict:
        """Nivel 4: Evaluación de estándares profesionales"""
        result = {
            'scalability': 0,
            'memorability': 0,
            'versatility': 0,
            'originality': 0,
            'warnings': [],
            'score': 0
        }

        try:
            root = ET.fromstring(svg_code)

            # Scalability: tiene viewBox + elementos vectoriales puros
            if root.get('viewBox'):
                has_raster = any(elem.tag.endswith('image') for elem in root.iter())
                result['scalability'] = 90 if not has_raster else 60
            else:
                result['scalability'] = 40
                result['warnings'].append("⚠ Sin viewBox (escalabilidad limitada)")

            # Memorability: basado en simplicidad
            element_count = len(list(root.iter()))
            if element_count <= 10:
                result['memorability'] = 95
            elif 10 < element_count <= 40:
                result['memorability'] = 85
            elif 40 < element_count <= 60:
                result['memorability'] = 65
            else:
                result['memorability'] = 45

            # Versatility: pocos colores + sin gradientes complejos
            colors = set(re.findall(r'#[0-9a-fA-F]{6}', svg_code))
            has_gradients = 'gradient' in svg_code.lower()

            if len(colors) <= 2 and not has_gradients:
                result['versatility'] = 95
            elif len(colors) <= 3:
                result['versatility'] = 85
            else:
                result['versatility'] = 70

            # Originality: heurística simple (no puede ser 100% automático)
            # Penalizar uso de clichés comunes
            cliches = ['lightbulb', 'rocket', 'globe']
            has_cliche = any(cliche in svg_code.lower() for cliche in cliches)
            result['originality'] = 75 if not has_cliche else 60

            # Score final = promedio ponderado
            result['score'] = int(
                result['scalability'] * 0.30 +
                result['memorability'] * 0.30 +
                result['versatility'] * 0.25 +
                result['originality'] * 0.15
            )

        except Exception as e:
            result['warnings'].append(f"Professional eval error: {e}")

        return result

    def evaluate_aesthetic_metrics(self, svg_code: str) -> Dict:
        """Nivel 5: Evaluación de métricas estéticas"""
        result = {
            'golden_ratio': 0.0,
            'color_harmony': 0.0,
            'visual_interest': 0.0,
            'score': 0
        }

        try:
            # Calcular cada métrica
            result['golden_ratio'] = self.aesthetic_metrics.calculate_golden_ratio_score(svg_code)
            result['color_harmony'] = self.aesthetic_metrics.calculate_color_harmony(svg_code)
            result['visual_interest'] = self.aesthetic_metrics.calculate_visual_interest(svg_code)

            # Score ponderado: 35% Golden Ratio + 35% Color + 30% Visual Interest
            result['score'] = int(
                result['golden_ratio'] * 0.35 +
                result['color_harmony'] * 0.35 +
                result['visual_interest'] * 0.30
            )

        except Exception as e:
            result['warnings'] = [f"Aesthetic eval error: {e}"]
            result['score'] = 50  # Neutral en caso de error

        return result

    def _calculate_final_score(self, results: Dict) -> int:
        """
        Calcula score final con NUEVAS métricas estéticas (v2.0)
        Ponderación: 50% Aesthetic + 35% Professional + 15% Technical
        """
        if not results['level1_xml']['valid']:
            return 0

        # Technical score (15% total = 40% XML + 30% SVG + 30% Quality)
        technical_score = (
            results['level1_xml']['score'] * 0.40 +
            results['level2_svg']['score'] * 0.30 +
            results['level3_quality']['score'] * 0.30
        )

        # NEW WEIGHTING: Aesthetic-first
        final = int(
            technical_score * 0.15 +                        # 15% Technical
            results['level5_aesthetic']['score'] * 0.50 +   # 50% Aesthetic ⭐
            results['level4_professional']['score'] * 0.35  # 35% Professional
        )

        return final

    def _calculate_legacy_score(self, results: Dict) -> int:
        """
        Calcula score con sistema ANTERIOR (sin métricas estéticas)
        Solo para comparación backward-compatibility
        """
        if not results['level1_xml']['valid']:
            return 0

        # Old weighting (v1.0)
        weights = {
            'level1_xml': 0.15,
            'level2_svg': 0.20,
            'level3_quality': 0.35,
            'level4_professional': 0.30
        }

        final = sum(
            results[level]['score'] * weight
            for level, weight in weights.items()
        )

        return int(final)

    def get_recommendations(self, results: Dict) -> List[str]:
        """Genera recomendaciones de mejora"""
        recommendations = []

        if not results['level1_xml']['valid']:
            recommendations.append("🔴 CRÍTICO: Corregir errores de sintaxis XML")

        if not results['level2_svg']['has_viewbox']:
            recommendations.append("🟡 Agregar viewBox para escalabilidad")

        complexity = results['level3_quality']['complexity']
        if complexity > 60:
            recommendations.append(f"🟡 Simplificar diseño (actual: {complexity}, óptimo: 20-40)")

        color_count = results['level3_quality']['color_count']
        if color_count > 3:
            recommendations.append(f"🟡 Reducir colores (actual: {color_count}, máximo: 3)")

        if results['level4_professional']['scalability'] < 70:
            recommendations.append("🟡 Mejorar escalabilidad (usar elementos vectoriales puros)")

        if results['level4_professional']['memorability'] < 70:
            recommendations.append("🟡 Simplificar para mejor memorabilidad")

        if results['final_score'] >= 85:
            recommendations.append("✅ Logo de excelente calidad profesional")
        elif results['final_score'] >= 70:
            recommendations.append("✅ Logo de buena calidad, listo para uso")
        elif results['final_score'] >= 50:
            recommendations.append("🟡 Logo aceptable, requiere mejoras menores")
        else:
            recommendations.append("🔴 Logo necesita mejoras significativas")

        return recommendations

    def print_report(self, results: Dict):
        """Imprime reporte detallado con métricas estéticas"""
        print("\n" + "="*70)
        print("REPORTE DE VALIDACIÓN Y SCORING v2.0 (Aesthetic Metrics)")
        print("="*70)

        print(f"\n📊 SCORE FINAL (NEW): {results['final_score']}/100")
        print(f"   Legacy Score (old): {results['legacy_score']}/100")
        score_improvement = results['final_score'] - results['legacy_score']
        arrow = "↑" if score_improvement > 0 else "↓" if score_improvement < 0 else "→"
        print(f"   Change: {arrow} {score_improvement:+d} points")
        print(f"   Status: {'✅ PASSED' if results['passed'] else '❌ NEEDS WORK'}")

        print("\n" + "-"*70)
        print("NIVEL 1: Validación XML")
        print("-"*70)
        l1 = results['level1_xml']
        print(f"   Válido: {'✅' if l1['valid'] else '❌'}")
        if l1['error']:
            print(f"   Error: {l1['error']}")
        print(f"   Score: {l1['score']}/100")

        print("\n" + "-"*70)
        print("NIVEL 2: Estructura SVG")
        print("-"*70)
        l2 = results['level2_svg']
        print(f"   SVG root: {'✅' if l2['has_svg_root'] else '❌'}")
        print(f"   viewBox: {'✅' if l2['has_viewbox'] else '❌'}")
        print(f"   xmlns: {'✅' if l2['has_xmlns'] else '❌'}")
        print(f"   Elementos: {l2['elements_count']}")
        print(f"   Score: {l2['score']}/100")

        print("\n" + "-"*70)
        print("NIVEL 3: Calidad Técnica")
        print("-"*70)
        l3 = results['level3_quality']
        print(f"   Complejidad: {l3['complexity']} ({l3['complexity_rating']})")
        print(f"   Colores: {l3['color_count']}")
        print(f"   Precisión OK: {'✅' if l3['precision_ok'] else '❌'}")
        print(f"   Score: {l3['score']}/100")

        print("\n" + "-"*70)
        print("NIVEL 4: Estándares Profesionales")
        print("-"*70)
        l4 = results['level4_professional']
        print(f"   Escalabilidad: {l4['scalability']}/100")
        print(f"   Memorabilidad: {l4['memorability']}/100")
        print(f"   Versatilidad: {l4['versatility']}/100")
        print(f"   Originalidad: {l4['originality']}/100")
        print(f"   Score: {l4['score']}/100")

        print("\n" + "-"*70)
        print("NIVEL 5: Métricas Estéticas ⭐ NEW")
        print("-"*70)
        l5 = results['level5_aesthetic']
        print(f"   Golden Ratio (φ=1.618): {l5['golden_ratio']:.1f}/100")
        print(f"   Color Harmony: {l5['color_harmony']:.1f}/100")
        print(f"   Visual Interest: {l5['visual_interest']:.1f}/100")
        print(f"   Score: {l5['score']}/100")

        print("\n" + "-"*70)
        print("RECOMENDACIONES")
        print("-"*70)
        for rec in self.get_recommendations(results):
            print(f"   {rec}")

        print("\n" + "="*70)


def demo():
    """Demo del validador"""
    validator = LogoValidator()

    # Ejemplo: Logo simple bueno
    svg_good = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="none" stroke="#2563eb" stroke-width="8"/>
  <path d="M70 100 Q100 70 130 100" fill="none" stroke="#2563eb" stroke-width="6"/>
</svg>"""

    print("VALIDANDO LOGO DE EJEMPLO:")
    results = validator.validate_all(svg_good)
    validator.print_report(results)


if __name__ == "__main__":
    demo()
