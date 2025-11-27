"""
Generador de galería HTML para visualizar logos generados
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from logo_metadata import LogoMetadata


class GalleryGenerator:
    """Genera galería HTML interactiva de logos"""

    def __init__(self, output_dir: str = "../output"):
        self.output_dir = output_dir
        self.metadata = LogoMetadata()

    def generate_gallery(self, output_file: str = "../output/gallery.html"):
        """Genera archivo HTML de galería"""

        logos = self.metadata.logos
        stats = self.metadata.get_stats()
        best_logos = self.metadata.get_best_logos(10)
        timeline = self.metadata.get_evolution_timeline()

        html = self._generate_html(logos, stats, best_logos, timeline)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ Galería generada: {output_file}")
        return output_file

    def _generate_html(self, logos: List[Dict], stats: Dict,
                      best_logos: List[Dict], timeline: List[Dict]) -> str:
        """Genera HTML completo"""

        # Preparar datos JSON para JavaScript
        logos_json = json.dumps(logos, ensure_ascii=False)
        stats_json = json.dumps(stats, ensure_ascii=False)
        timeline_json = json.dumps(timeline, ensure_ascii=False)

        return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logo Gallery - AI Generated</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--gray-50);
            color: var(--gray-900);
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }}

        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}

        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .stat-card .label {{
            color: var(--gray-700);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}

        .controls {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 2rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .controls-row {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}

        .control-group {{
            flex: 1;
            min-width: 200px;
        }}

        .control-group label {{
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--gray-700);
        }}

        select, input[type="text"], input[type="number"] {{
            width: 100%;
            padding: 0.75rem;
            border: 2px solid var(--gray-200);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s;
        }}

        select:focus, input:focus {{
            outline: none;
            border-color: var(--primary);
        }}

        .btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 1rem;
        }}

        .btn-primary {{
            background: var(--primary);
            color: white;
        }}

        .btn-primary:hover {{
            background: var(--primary-dark);
            transform: translateY(-1px);
        }}

        .tabs {{
            display: flex;
            gap: 1rem;
            border-bottom: 2px solid var(--gray-200);
            margin: 2rem 0;
        }}

        .tab {{
            padding: 1rem 1.5rem;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            font-weight: 600;
            color: var(--gray-700);
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: var(--primary);
        }}

        .tab.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .logos-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}

        .logo-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.3s;
            cursor: pointer;
        }}

        .logo-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }}

        .logo-card.favorite {{
            border: 3px solid var(--warning);
        }}

        .logo-preview {{
            width: 100%;
            height: 250px;
            background: var(--gray-100);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            position: relative;
        }}

        .logo-preview svg, .logo-preview img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .favorite-badge {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--warning);
            color: white;
            padding: 0.5rem;
            border-radius: 50%;
            font-size: 1.2rem;
        }}

        .logo-info {{
            padding: 1.5rem;
        }}

        .logo-title {{
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .logo-meta {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 1rem 0;
        }}

        .tag {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .tag-industry {{
            background: #dbeafe;
            color: #1e40af;
        }}

        .tag-style {{
            background: #dcfce7;
            color: #166534;
        }}

        .tag-version {{
            background: #f3e8ff;
            color: #6b21a8;
        }}

        .score-bar {{
            width: 100%;
            height: 8px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }}

        .score-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}

        .score-fill.excellent {{
            background: var(--success);
        }}

        .score-fill.good {{
            background: var(--primary);
        }}

        .score-fill.fair {{
            background: var(--warning);
        }}

        .score-fill.poor {{
            background: var(--danger);
        }}

        .score-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--gray-700);
            margin-top: 0.5rem;
        }}

        .score-value {{
            font-weight: 700;
            font-size: 1.1rem;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}

        .timeline-chart {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 2rem 0;
        }}

        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}

        .modal.active {{
            display: flex;
        }}

        .modal-content {{
            background: white;
            border-radius: 12px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            padding: 2rem;
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }}

        .close-btn {{
            font-size: 2rem;
            cursor: pointer;
            color: var(--gray-700);
        }}

        .close-btn:hover {{
            color: var(--danger);
        }}

        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--gray-700);
        }}

        .empty-state svg {{
            width: 120px;
            height: 120px;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}

        @media (max-width: 768px) {{
            .logos-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🎨 Logo Gallery</h1>
            <p>AI-Generated Professional Logos</p>
        </div>
    </div>

    <div class="container">
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{stats.get('total', 0)}</div>
                <div class="label">Total Logos</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('avg_score', 0):.0f}</div>
                <div class="label">Avg Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('max_score', 0)}</div>
                <div class="label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('favorites', 0)}</div>
                <div class="label">⭐ Favorites</div>
            </div>
        </div>

        <!-- Controls -->
        <div class="controls">
            <div class="controls-row">
                <div class="control-group">
                    <label>Industria</label>
                    <select id="filter-industry">
                        <option value="">Todas</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Estilo</label>
                    <select id="filter-style">
                        <option value="">Todos</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Versión</label>
                    <select id="filter-version">
                        <option value="">Todas</option>
                        <option value="v1">v1</option>
                        <option value="v2">v2</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Score Mínimo</label>
                    <input type="number" id="filter-score" min="0" max="100" placeholder="0-100">
                </div>
                <div class="control-group">
                    <label>Buscar</label>
                    <input type="text" id="search" placeholder="Nombre de empresa...">
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <div class="tab active" onclick="showTab('all')">Todos ({stats.get('total', 0)})</div>
            <div class="tab" onclick="showTab('best')">🏆 Mejores</div>
            <div class="tab" onclick="showTab('favorites')">⭐ Favoritos</div>
            <div class="tab" onclick="showTab('comparison')">📊 Comparación</div>
            <div class="tab" onclick="showTab('timeline')">📈 Evolución</div>
        </div>

        <!-- Tab: All Logos -->
        <div id="tab-all" class="tab-content active">
            <div id="logos-grid" class="logos-grid"></div>
        </div>

        <!-- Tab: Best -->
        <div id="tab-best" class="tab-content">
            <div id="best-grid" class="logos-grid"></div>
        </div>

        <!-- Tab: Favorites -->
        <div id="tab-favorites" class="tab-content">
            <div id="favorites-grid" class="logos-grid"></div>
        </div>

        <!-- Tab: Comparison -->
        <div id="tab-comparison" class="tab-content">
            <h2>Comparación por Versión</h2>
            <div id="comparison-grid" class="comparison-grid"></div>
        </div>

        <!-- Tab: Timeline -->
        <div id="tab-timeline" class="tab-content">
            <div class="timeline-chart">
                <h2>Evolución de Scores</h2>
                <canvas id="timeline-canvas"></canvas>
            </div>
        </div>
    </div>

    <!-- Modal -->
    <div id="modal" class="modal" onclick="closeModal()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2 id="modal-title"></h2>
                <span class="close-btn" onclick="closeModal()">&times;</span>
            </div>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        // Data
        const logos = {logos_json};
        const stats = {stats_json};
        const timeline = {timeline_json};

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            populateFilters();
            renderLogos(logos, 'logos-grid');
            renderBest();
            renderFavorites();
            renderComparison();
            renderTimeline();
            setupFilters();
        }});

        function populateFilters() {{
            const industries = [...new Set(logos.map(l => l.industry))];
            const styles = [...new Set(logos.map(l => l.style))];

            const industrySelect = document.getElementById('filter-industry');
            industries.forEach(ind => {{
                const option = document.createElement('option');
                option.value = ind;
                option.textContent = ind;
                industrySelect.appendChild(option);
            }});

            const styleSelect = document.getElementById('filter-style');
            styles.forEach(style => {{
                const option = document.createElement('option');
                option.value = style;
                option.textContent = style;
                styleSelect.appendChild(option);
            }});
        }}

        function setupFilters() {{
            ['filter-industry', 'filter-style', 'filter-version', 'filter-score', 'search'].forEach(id => {{
                document.getElementById(id).addEventListener('change', applyFilters);
                document.getElementById(id).addEventListener('input', applyFilters);
            }});
        }}

        function applyFilters() {{
            const industry = document.getElementById('filter-industry').value;
            const style = document.getElementById('filter-style').value;
            const version = document.getElementById('filter-version').value;
            const minScore = parseInt(document.getElementById('filter-score').value) || 0;
            const search = document.getElementById('search').value.toLowerCase();

            const filtered = logos.filter(logo => {{
                if (industry && logo.industry !== industry) return false;
                if (style && logo.style !== style) return false;
                if (version && logo.version !== version) return false;
                if (logo.score < minScore) return false;
                if (search && !logo.company_name.toLowerCase().includes(search)) return false;
                return true;
            }});

            renderLogos(filtered, 'logos-grid');
        }}

        function renderLogos(logosList, containerId) {{
            const container = document.getElementById(containerId);

            if (logosList.length === 0) {{
                container.innerHTML = `
                    <div class="empty-state">
                        <p>No hay logos que mostrar</p>
                    </div>
                `;
                return;
            }}

            container.innerHTML = logosList.map(logo => `
                <div class="logo-card ${{logo.is_favorite ? 'favorite' : ''}}" onclick="showDetails('${{logo.id}}')">
                    <div class="logo-preview">
                        ${{logo.is_favorite ? '<div class="favorite-badge">⭐</div>' : ''}}
                        <img src="${{logo.filename}}" alt="${{logo.company_name}}" />
                    </div>
                    <div class="logo-info">
                        <div class="logo-title">${{logo.company_name}}</div>
                        <div class="logo-meta">
                            <span class="tag tag-industry">${{logo.industry}}</span>
                            <span class="tag tag-style">${{logo.style}}</span>
                            <span class="tag tag-version">${{logo.version}}</span>
                        </div>
                        <div class="score-bar">
                            <div class="score-fill ${{getScoreClass(logo.score)}}" style="width: ${{logo.score}}%"></div>
                        </div>
                        <div class="score-label">
                            <span>Score</span>
                            <span class="score-value">${{logo.score}}/100</span>
                        </div>
                        <div style="font-size: 0.85rem; color: var(--gray-700); margin-top: 0.5rem;">
                            Complejidad: ${{logo.complexity}} | Iteración: ${{logo.iteration}}
                        </div>
                    </div>
                </div>
            `).join('');
        }}

        function renderBest() {{
            const best = [...logos].sort((a, b) => b.score - a.score).slice(0, 10);
            renderLogos(best, 'best-grid');
        }}

        function renderFavorites() {{
            const favorites = logos.filter(l => l.is_favorite);
            renderLogos(favorites, 'favorites-grid');
        }}

        function renderComparison() {{
            const v1 = logos.filter(l => l.version === 'v1');
            const v2 = logos.filter(l => l.version === 'v2');

            const avgV1 = v1.length ? v1.reduce((sum, l) => sum + l.score, 0) / v1.length : 0;
            const avgV2 = v2.length ? v2.reduce((sum, l) => sum + l.score, 0) / v2.length : 0;

            document.getElementById('comparison-grid').innerHTML = `
                <div class="stat-card">
                    <div class="value">${{v1.length}}</div>
                    <div class="label">v1 Logos</div>
                    <div style="margin-top: 1rem; font-size: 1.5rem; font-weight: 700;">
                        ${{avgV1.toFixed(1)}}
                    </div>
                    <div class="label">Avg Score</div>
                </div>
                <div class="stat-card">
                    <div class="value">${{v2.length}}</div>
                    <div class="label">v2 Logos</div>
                    <div style="margin-top: 1rem; font-size: 1.5rem; font-weight: 700;">
                        ${{avgV2.toFixed(1)}}
                    </div>
                    <div class="label">Avg Score</div>
                </div>
                <div class="stat-card">
                    <div class="value" style="color: ${{avgV2 > avgV1 ? 'var(--success)' : 'var(--danger)'}}">
                        ${{avgV2 > avgV1 ? '+' : ''}}${{(avgV2 - avgV1).toFixed(1)}}
                    </div>
                    <div class="label">Mejora</div>
                    <div style="margin-top: 1rem; font-size: 1.5rem; font-weight: 700;">
                        ${{avgV1 > 0 ? ((avgV2 - avgV1) / avgV1 * 100).toFixed(1) : 0}}%
                    </div>
                    <div class="label">Cambio</div>
                </div>
            `;
        }}

        function renderTimeline() {{
            // Simple text timeline for now
            if (timeline.length === 0) {{
                document.getElementById('timeline-canvas').parentElement.innerHTML = '<p>No hay datos suficientes para timeline</p>';
                return;
            }}

            const timelineText = timeline.map(t => `
                <div style="padding: 1rem; border-left: 3px solid var(--primary); margin: 0.5rem 0;">
                    <strong>${{t.date}}</strong>: ${{t.count}} logos generados, promedio ${{t.avg_score.toFixed(1)}}/100
                    (rango: ${{t.min_score}}-${{t.max_score}})
                </div>
            `).join('');

            document.getElementById('timeline-canvas').parentElement.innerHTML = `
                <h2>Evolución de Scores</h2>
                ${{timelineText}}
            `;
        }}

        function getScoreClass(score) {{
            if (score >= 85) return 'excellent';
            if (score >= 70) return 'good';
            if (score >= 50) return 'fair';
            return 'poor';
        }}

        function showTab(tabName) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
        }}

        function showDetails(logoId) {{
            const logo = logos.find(l => l.id === logoId);
            if (!logo) return;

            document.getElementById('modal-title').textContent = logo.company_name;
            document.getElementById('modal-body').innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    <div>
                        <img src="${{logo.filename}}" style="width: 100%; max-width: 400px; border: 1px solid var(--gray-200); border-radius: 8px;" />
                    </div>
                    <div>
                        <h3>Información</h3>
                        <p><strong>Industria:</strong> ${{logo.industry}}</p>
                        <p><strong>Estilo:</strong> ${{logo.style}}</p>
                        <p><strong>Versión:</strong> ${{logo.version}}</p>
                        <p><strong>Iteración:</strong> ${{logo.iteration}}</p>
                        <p><strong>Score:</strong> ${{logo.score}}/100</p>
                        <p><strong>Complejidad:</strong> ${{logo.complexity}}</p>
                        <p><strong>Colores:</strong> ${{logo.colors.join(', ')}}</p>
                        <p><strong>Timestamp:</strong> ${{new Date(logo.timestamp).toLocaleString()}}</p>
                        ${{logo.notes ? `<p><strong>Notas:</strong> ${{logo.notes}}</p>` : ''}}

                        <h3 style="margin-top: 2rem;">Validación</h3>
                        ${{logo.validation && logo.validation.level1_xml ? `
                            <p>XML: ${{logo.validation.level1_xml.score}}/100</p>
                            <p>SVG: ${{logo.validation.level2_svg.score}}/100</p>
                            <p>Quality: ${{logo.validation.level3_quality.score}}/100</p>
                            <p>Professional: ${{logo.validation.level4_professional.score}}/100</p>
                        ` : '<p>No disponible</p>'}}
                    </div>
                </div>
            `;

            document.getElementById('modal').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('modal').classList.remove('active');
        }}

        // Close modal on ESC
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') closeModal();
        }});
    </script>
</body>
</html>"""

    def generate_comparison_report(self, company_name: str,
                                  output_file: str = None) -> str:
        """Genera reporte de comparación de iteraciones"""
        iterations = self.metadata.get_iterations_comparison(company_name)

        if not iterations:
            print(f"No se encontraron logos para {company_name}")
            return None

        if not output_file:
            output_file = f"../output/{company_name.lower().replace(' ', '_')}_comparison.html"

        # Generar HTML de comparación
        html = self._generate_comparison_html(iterations)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ Comparación generada: {output_file}")
        return output_file

    def _generate_comparison_html(self, iterations: List[Dict]) -> str:
        """HTML para comparación de iteraciones"""
        # Implementación simplificada
        return f"<html><body><h1>Comparison Report</h1></body></html>"


def main():
    """Genera galería"""
    generator = GalleryGenerator()
    output = generator.generate_gallery()

    print(f"\n{'='*70}")
    print("GALERÍA GENERADA")
    print(f"{'='*70}")
    print(f"\nAbre en tu navegador:")
    print(f"  file://{os.path.abspath(output)}")
    print(f"\nO ejecuta:")
    print(f"  open {output}  # macOS")
    print(f"  xdg-open {output}  # Linux")
    print(f"  start {output}  # Windows")


if __name__ == "__main__":
    main()
