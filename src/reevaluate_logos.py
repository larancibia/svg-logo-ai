#!/usr/bin/env python3
"""
Re-evalúa todos los logos existentes con las NUEVAS métricas estéticas v2.0
Actualiza logos_metadata.json con los scores corregidos
"""

import json
from pathlib import Path
from logo_validator import LogoValidator
from logo_metadata import LogoMetadata

def main():
    print("="*80)
    print("RE-EVALUACIÓN DE LOGOS CON MÉTRICAS ESTÉTICAS v2.0")
    print("="*80)

    # Paths
    output_dir = Path(__file__).parent.parent / "output"
    metadata_file = output_dir / "logos_metadata.json"

    # Initialize
    validator = LogoValidator()
    metadata = LogoMetadata(str(metadata_file))

    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Metadata file: {metadata_file}")
    print(f"📝 Total logos in metadata: {len(metadata.logos)}")

    # Re-evaluate all logos
    updated_count = 0
    not_found_count = 0

    print("\n" + "─"*80)
    print("Re-evaluating logos...")
    print("─"*80)

    for logo in metadata.logos:
        svg_path = output_dir / logo['filename']

        if not svg_path.exists():
            print(f"⚠ File not found: {logo['filename']}")
            not_found_count += 1
            continue

        # Read SVG
        with open(svg_path, 'r') as f:
            svg_code = f.read()

        # Re-evaluate with new metrics
        results = validator.validate_all(svg_code)

        # Store old score for comparison
        old_score = logo.get('score', 0)
        old_legacy = logo.get('legacy_score', old_score)  # May not exist yet

        # Update metadata
        logo['score'] = results['final_score']  # NEW score
        logo['legacy_score'] = results['legacy_score']  # OLD score for comparison

        # Add aesthetic breakdown
        logo['aesthetic_metrics'] = {
            'golden_ratio': results['level5_aesthetic']['golden_ratio'],
            'color_harmony': results['level5_aesthetic']['color_harmony'],
            'visual_interest': results['level5_aesthetic']['visual_interest'],
            'score': results['level5_aesthetic']['score']
        }

        # Calculate change
        score_change = results['final_score'] - old_legacy
        arrow = "↑" if score_change > 0 else "↓" if score_change < 0 else "→"

        print(f"{logo['filename']:<35} "
              f"Old: {old_legacy:>3} → New: {results['final_score']:>3} "
              f"{arrow} {score_change:+3d}  "
              f"(Aesthetic: {results['level5_aesthetic']['score']}/100)")

        updated_count += 1

    # Save updated metadata
    metadata._save_metadata()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Updated: {updated_count} logos")
    if not_found_count > 0:
        print(f"⚠ Not found: {not_found_count} logos")

    # Calculate v1 vs v2 averages
    v1_logos = [l for l in metadata.logos if 'v1' in l['filename']]
    v2_logos = [l for l in metadata.logos if 'v2' in l['filename']]

    if v1_logos:
        v1_new = sum(l['score'] for l in v1_logos) / len(v1_logos)
        v1_old = sum(l['legacy_score'] for l in v1_logos) / len(v1_logos)
        print(f"\nv1 logos (basic): {len(v1_logos)} logos")
        print(f"   Old system: {v1_old:.1f}/100")
        print(f"   NEW system: {v1_new:.1f}/100")
        print(f"   Change: {v1_new - v1_old:+.1f} points")

    if v2_logos:
        v2_new = sum(l['score'] for l in v2_logos) / len(v2_logos)
        v2_old = sum(l['legacy_score'] for l in v2_logos) / len(v2_logos)
        print(f"\nv2 logos (designed): {len(v2_logos)} logos")
        print(f"   Old system: {v2_old:.1f}/100")
        print(f"   NEW system: {v2_new:.1f}/100")
        print(f"   Change: {v2_new - v2_old:+.1f} points")

    if v1_logos and v2_logos:
        old_diff = v2_old - v1_old
        new_diff = v2_new - v1_new

        print(f"\n🎯 DISCRIMINATION IMPROVEMENT:")
        print(f"   Old system: v2 was {old_diff:+.1f} points better than v1")
        print(f"   NEW system: v2 is  {new_diff:+.1f} points better than v1")
        print(f"   Improvement: {(new_diff - old_diff):+.1f} points better discrimination")

        if new_diff > old_diff:
            multiplier = abs(new_diff / old_diff) if old_diff != 0 else float('inf')
            print(f"\n   ✅ NEW METRICS ARE {multiplier:.1f}× BETTER at detecting quality!")
        else:
            print(f"\n   ⚠ Unexpected: new metrics not performing as expected")

    print("\n" + "="*80)
    print(f"✅ Metadata updated: {metadata_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
