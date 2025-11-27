"""
Sistema de metadata para tracking de logos generados
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class LogoMetadata:
    """Gestiona metadata de logos generados"""

    def __init__(self, metadata_file: str = "../output/logos_metadata.json"):
        self.metadata_file = metadata_file
        self.logos = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        """Carga metadata existente"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        """Guarda metadata"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.logos, f, indent=2, ensure_ascii=False)

    def add_logo(self,
                 filename: str,
                 company_name: str,
                 industry: str,
                 style: str,
                 score: int,
                 complexity: int,
                 version: str = "v2",
                 iteration: int = 1,
                 colors: List[str] = None,
                 notes: str = "",
                 is_favorite: bool = False,
                 validation_results: Dict = None) -> str:
        """
        Agrega un logo a la metadata

        Returns:
            logo_id: ID único del logo
        """
        logo_id = f"{company_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logo_entry = {
            'id': logo_id,
            'filename': filename,
            'company_name': company_name,
            'industry': industry,
            'style': style,
            'score': score,
            'complexity': complexity,
            'version': version,
            'iteration': iteration,
            'colors': colors or [],
            'notes': notes,
            'is_favorite': is_favorite,
            'timestamp': datetime.now().isoformat(),
            'validation': validation_results or {}
        }

        self.logos.append(logo_entry)
        self._save_metadata()

        return logo_id

    def update_logo(self, logo_id: str, updates: Dict):
        """Actualiza metadata de un logo"""
        for logo in self.logos:
            if logo['id'] == logo_id:
                logo.update(updates)
                self._save_metadata()
                return True
        return False

    def set_favorite(self, logo_id: str, is_favorite: bool = True):
        """Marca/desmarca un logo como favorito"""
        return self.update_logo(logo_id, {'is_favorite': is_favorite})

    def add_rating(self, logo_id: str, rating: int, comment: str = ""):
        """Agrega rating manual a un logo"""
        for logo in self.logos:
            if logo['id'] == logo_id:
                if 'ratings' not in logo:
                    logo['ratings'] = []
                logo['ratings'].append({
                    'rating': rating,
                    'comment': comment,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_metadata()
                return True
        return False

    def get_logos(self,
                  industry: Optional[str] = None,
                  style: Optional[str] = None,
                  version: Optional[str] = None,
                  min_score: Optional[int] = None,
                  favorites_only: bool = False) -> List[Dict]:
        """Obtiene logos con filtros"""
        filtered = self.logos

        if industry:
            filtered = [l for l in filtered if industry.lower() in l['industry'].lower()]

        if style:
            filtered = [l for l in filtered if style.lower() in l['style'].lower()]

        if version:
            filtered = [l for l in filtered if l['version'] == version]

        if min_score is not None:
            filtered = [l for l in filtered if l['score'] >= min_score]

        if favorites_only:
            filtered = [l for l in filtered if l.get('is_favorite', False)]

        return filtered

    def get_best_logos(self, n: int = 10) -> List[Dict]:
        """Obtiene los N mejores logos por score"""
        sorted_logos = sorted(self.logos, key=lambda x: x['score'], reverse=True)
        return sorted_logos[:n]

    def get_by_company(self, company_name: str) -> List[Dict]:
        """Obtiene todas las versiones de logos de una empresa"""
        return [l for l in self.logos
                if l['company_name'].lower() == company_name.lower()]

    def get_iterations_comparison(self, company_name: str) -> List[Dict]:
        """Obtiene iteraciones de una empresa ordenadas"""
        iterations = self.get_by_company(company_name)
        return sorted(iterations, key=lambda x: x['iteration'])

    def get_evolution_timeline(self) -> List[Dict]:
        """Timeline de evolución de scores promedio"""
        if not self.logos:
            return []

        # Agrupar por fecha
        by_date = {}
        for logo in sorted(self.logos, key=lambda x: x['timestamp']):
            date = logo['timestamp'].split('T')[0]
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(logo['score'])

        # Calcular promedios
        timeline = []
        for date, scores in by_date.items():
            timeline.append({
                'date': date,
                'avg_score': sum(scores) / len(scores),
                'count': len(scores),
                'max_score': max(scores),
                'min_score': min(scores)
            })

        return timeline

    def get_stats(self) -> Dict:
        """Estadísticas generales"""
        if not self.logos:
            return {
                'total': 0,
                'avg_score': 0,
                'by_industry': {},
                'by_version': {},
                'favorites': 0
            }

        scores = [l['score'] for l in self.logos]

        # Por industria
        by_industry = {}
        for logo in self.logos:
            ind = logo['industry']
            if ind not in by_industry:
                by_industry[ind] = []
            by_industry[ind].append(logo['score'])

        for ind in by_industry:
            by_industry[ind] = {
                'count': len(by_industry[ind]),
                'avg_score': sum(by_industry[ind]) / len(by_industry[ind])
            }

        # Por versión
        by_version = {}
        for logo in self.logos:
            ver = logo['version']
            if ver not in by_version:
                by_version[ver] = []
            by_version[ver].append(logo['score'])

        for ver in by_version:
            by_version[ver] = {
                'count': len(by_version[ver]),
                'avg_score': sum(by_version[ver]) / len(by_version[ver])
            }

        return {
            'total': len(self.logos),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'by_industry': by_industry,
            'by_version': by_version,
            'favorites': sum(1 for l in self.logos if l.get('is_favorite', False))
        }

    def export_comparison(self, logo_ids: List[str], output_file: str):
        """Exporta comparación de logos específicos"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'logos': [l for l in self.logos if l['id'] in logo_ids]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)


def demo():
    """Demo del sistema de metadata"""
    metadata = LogoMetadata()

    # Agregar logo de ejemplo
    logo_id = metadata.add_logo(
        filename="techflow_logo.svg",
        company_name="TechFlow",
        industry="Technology",
        style="minimalist",
        score=87,
        complexity=28,
        version="v2",
        iteration=1,
        colors=["#2563eb", "#3b82f6"],
        notes="Excelente balance y simplicidad",
        is_favorite=True,
        validation_results={
            'level1_xml': {'score': 100},
            'level2_svg': {'score': 100},
            'level3_quality': {'score': 85},
            'level4_professional': {'score': 90}
        }
    )

    print(f"✓ Logo agregado: {logo_id}")

    # Agregar rating
    metadata.add_rating(logo_id, 9, "Me encanta la simplicidad")

    # Stats
    stats = metadata.get_stats()
    print(f"\n📊 Estadísticas:")
    print(f"   Total logos: {stats['total']}")
    print(f"   Score promedio: {stats['avg_score']:.1f}")
    print(f"   Favoritos: {stats['favorites']}")

    # Mejores
    best = metadata.get_best_logos(5)
    print(f"\n🏆 Top 5:")
    for i, logo in enumerate(best, 1):
        print(f"   {i}. {logo['company_name']} - {logo['score']}/100")


if __name__ == "__main__":
    demo()
