"""
LLM-Guided Mutation for MAP-Elites
===================================
Intelligently mutates logos toward specific behavioral targets using LLM reasoning.

Instead of random mutations, the LLM understands the target behavior and
modifies the logo accordingly (e.g., "make it more complex", "make it more geometric").
"""

import os
import re
import time
from typing import Dict, Tuple, Optional
import google.generativeai as genai


class LLMGuidedMutator:
    """
    LLM-based mutation operator that modifies logos toward target behaviors
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize LLM mutator

        Args:
            model_name: Gemini model to use
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def mutate_toward_target(self,
                            source_svg: str,
                            current_behavior: Tuple[int, int, int, int],
                            target_behavior: Tuple[int, int, int, int],
                            genome: Dict = None,
                            max_retries: int = 3) -> str:
        """
        Mutate logo toward target behavior

        Args:
            source_svg: Current SVG code
            current_behavior: Current behavior bins (complexity, style, symmetry, color)
            target_behavior: Target behavior bins
            genome: Optional genome information for context
            max_retries: Maximum retry attempts

        Returns:
            Modified SVG code
        """
        # Build mutation prompt based on behavior differences
        prompt = self._build_mutation_prompt(
            source_svg,
            current_behavior,
            target_behavior,
            genome
        )

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    return svg_code

            except Exception as e:
                print(f"⚠ Mutation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue

        # Fallback: return original
        print("⚠ All mutation attempts failed, returning original")
        return source_svg

    def _build_mutation_prompt(self,
                              source_svg: str,
                              current: Tuple[int, int, int, int],
                              target: Tuple[int, int, int, int],
                              genome: Optional[Dict]) -> str:
        """
        Build intelligent mutation prompt based on behavior delta

        Args:
            source_svg: Current SVG
            current: (complexity_bin, style_bin, symmetry_bin, color_bin)
            target: Target behavior bins
            genome: Optional genome context
        """
        # Calculate deltas
        complexity_delta = target[0] - current[0]
        style_delta = target[1] - current[1]
        symmetry_delta = target[2] - current[2]
        color_delta = target[3] - current[3]

        # Build mutation instructions
        instructions = []

        # Complexity mutation
        if complexity_delta > 2:
            instructions.append(f"INCREASE COMPLEXITY: Add {complexity_delta * 3}-{complexity_delta * 5} more SVG elements (circles, rectangles, paths)")
        elif complexity_delta > 0:
            instructions.append(f"SLIGHTLY INCREASE COMPLEXITY: Add {complexity_delta * 2}-{complexity_delta * 3} more SVG elements")
        elif complexity_delta < -2:
            instructions.append(f"DECREASE COMPLEXITY: Remove {abs(complexity_delta) * 3}-{abs(complexity_delta) * 5} SVG elements to simplify")
        elif complexity_delta < 0:
            instructions.append(f"SLIGHTLY DECREASE COMPLEXITY: Remove {abs(complexity_delta) * 2} SVG elements")

        # Style mutation (geometric vs organic)
        if style_delta > 2:
            instructions.append("MAKE MORE ORGANIC: Convert straight lines/rectangles to curved paths (use bezier curves, Q/C commands)")
        elif style_delta > 0:
            instructions.append("ADD SOME CURVES: Introduce subtle curves or rounded corners")
        elif style_delta < -2:
            instructions.append("MAKE MORE GEOMETRIC: Convert curves to straight lines, use circles/rectangles/polygons instead of complex paths")
        elif style_delta < 0:
            instructions.append("SIMPLIFY CURVES: Reduce complexity of curved paths, make shapes more angular")

        # Symmetry mutation
        if symmetry_delta > 2:
            instructions.append("ADD STRONG SYMMETRY: Create perfect vertical or horizontal reflection symmetry")
        elif symmetry_delta > 0:
            instructions.append("ADD PARTIAL SYMMETRY: Introduce some symmetric elements")
        elif symmetry_delta < -2:
            instructions.append("BREAK SYMMETRY: Make design asymmetric by moving/rotating elements")
        elif symmetry_delta < 0:
            instructions.append("REDUCE SYMMETRY: Introduce slight asymmetric variations")

        # Color mutation
        if color_delta > 2:
            instructions.append("ADD MORE COLORS: Introduce 2-3 new distinct colors to the palette")
        elif color_delta > 0:
            instructions.append("ADD ONE COLOR: Introduce 1 additional color")
        elif color_delta < -2:
            instructions.append("REDUCE TO MONOCHROME: Use only 1-2 colors maximum")
        elif color_delta < 0:
            instructions.append("SIMPLIFY COLORS: Remove 1 color from the palette")

        # If no changes needed, make minor variation
        if not instructions:
            instructions.append("MINOR VARIATION: Make small adjustments while preserving overall design")

        # Build full prompt
        genome_context = ""
        if genome:
            genome_context = f"""
ORIGINAL DESIGN CONTEXT:
- Company: {genome.get('company', 'N/A')}
- Industry: {genome.get('industry', 'N/A')}
- Style: {', '.join(genome.get('style_keywords', []))}
"""

        prompt = f"""You are an expert SVG logo designer. Your task is to MODIFY an existing logo to match specific behavioral targets.

{genome_context}

CURRENT LOGO:
```svg
{source_svg}
```

MUTATION INSTRUCTIONS:
{chr(10).join(f"- {inst}" for inst in instructions)}

IMPORTANT RULES:
1. START with the current logo and MODIFY it (don't create from scratch)
2. Maintain the overall design concept and aesthetic quality
3. Make changes GRADUALLY (not extreme transformations)
4. Keep viewBox and xmlns attributes
5. Ensure the output is valid, well-structured SVG
6. NO text elements (logo must be purely graphical)
7. Keep the logo professional and suitable for branding

OUTPUT FORMAT:
Return ONLY the modified SVG code in a code block:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Modified logo here -->
</svg>
```

Generate the modified logo now:"""

        return prompt

    def _extract_svg(self, text: str) -> str:
        """Extract SVG code from LLM response"""
        # Try to find SVG in code blocks
        svg_pattern = r'```(?:svg|xml)?\s*(.*?)```'
        matches = re.findall(svg_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find raw SVG
        if '<svg' in text:
            start = text.find('<svg')
            end = text.find('</svg>') + 6
            if start != -1 and end > start:
                return text[start:end].strip()

        return text.strip()

    def generate_from_genome(self, genome: Dict, max_retries: int = 3) -> str:
        """
        Generate fresh logo from genome (for initialization)

        Args:
            genome: Genome dictionary with design parameters
            max_retries: Maximum retry attempts

        Returns:
            SVG code
        """
        prompt = self._build_generation_prompt(genome)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    return svg_code

            except Exception as e:
                print(f"⚠ Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue

        # Fallback
        return self._fallback_svg()

    def _build_generation_prompt(self, genome: Dict) -> str:
        """Build prompt for generating new logo from genome"""
        company = genome.get('company', 'TechCorp')
        industry = genome.get('industry', 'technology')
        style_keywords = genome.get('style_keywords', ['modern', 'minimal'])
        color_palette = genome.get('color_palette', ['#2563eb'])
        design_principles = genome.get('design_principles', ['simplicity'])
        complexity_target = genome.get('complexity_target', 25)

        prompt = f"""You are an expert SVG logo designer. Create a professional logo with these specifications:

COMPANY: {company}
INDUSTRY: {industry}

DESIGN REQUIREMENTS:
- Style: {', '.join(style_keywords)}
- Colors: {', '.join(color_palette)} (use 1-3 colors)
- Design Principles: {', '.join(design_principles)}
- Complexity: Aim for approximately {complexity_target} SVG elements

CONSTRAINTS:
1. Logo must be scalable (use viewBox="0 0 200 200")
2. NO text/fonts (purely graphical)
3. Maximum 3 colors
4. Professional quality suitable for branding
5. Simple enough to work at small sizes (favicon)

OUTPUT FORMAT:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Your logo here -->
</svg>
```

Generate the logo now:"""

        return prompt

    def _fallback_svg(self) -> str:
        """Simple fallback SVG if generation fails"""
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <circle cx="100" cy="100" r="40" fill="#ffffff"/>
</svg>"""


def demo():
    """Demo LLM-guided mutation"""
    print("="*60)
    print("LLM-Guided Mutation Demo")
    print("="*60)

    mutator = LLMGuidedMutator()

    # Test 1: Generate from genome
    print("\nTest 1: Generate from Genome")
    print("-"*60)
    genome = {
        'company': 'FlowAI',
        'industry': 'artificial intelligence',
        'style_keywords': ['modern', 'geometric', 'minimal'],
        'color_palette': ['#2563eb', '#1e40af'],
        'design_principles': ['simplicity', 'balance'],
        'complexity_target': 20
    }

    svg = mutator.generate_from_genome(genome)
    print(f"Generated {len(svg)} characters of SVG")
    print(f"Has svg tag: {'<svg' in svg}")

    # Test 2: Mutate toward more complex
    print("\n\nTest 2: Mutate to Increase Complexity")
    print("-"*60)

    source_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

    current_behavior = (0, 0, 9, 0)  # Very simple, geometric, symmetric, monochrome
    target_behavior = (5, 0, 9, 0)   # More complex, still geometric, symmetric, monochrome

    mutated_svg = mutator.mutate_toward_target(
        source_svg,
        current_behavior,
        target_behavior,
        genome
    )

    print(f"Source SVG length: {len(source_svg)}")
    print(f"Mutated SVG length: {len(mutated_svg)}")
    print(f"Has svg tag: {'<svg' in mutated_svg}")

    # Save for inspection
    output_dir = "/home/luis/svg-logo-ai/experiments/experiment_20251127_053108"
    with open(f"{output_dir}/demo_mutation_original.svg", 'w') as f:
        f.write(source_svg)
    with open(f"{output_dir}/demo_mutation_mutated.svg", 'w') as f:
        f.write(mutated_svg)

    print(f"\nSVG files saved to {output_dir}")


if __name__ == "__main__":
    demo()
