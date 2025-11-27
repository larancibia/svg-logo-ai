"""
LLM Logo Generator
==================
Advanced LLM-based SVG logo generation with multiple variations and targeted generation.

Features:
- Generate multiple diverse variations from natural language
- Target specific behavioral characteristics
- Include design rationale with each generation
- Support for Google Gemini 2.0 Flash
"""

import os
import re
import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LogoVariation:
    """Represents a single logo variation"""
    svg_code: str
    design_rationale: str
    style_description: str
    estimated_complexity: int
    estimated_fitness: float
    metadata: Dict


class LLMLogoGenerator:
    """
    Sophisticated LLM-based SVG logo generator

    Generates logos from natural language using chain-of-thought reasoning
    and design principles from professional logo design.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", cache_enabled: bool = True):
        """
        Initialize LLM Logo Generator

        Args:
            model_name: Gemini model to use
            cache_enabled: Enable response caching for efficiency
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None

        logger.info(f"Initialized LLMLogoGenerator with model: {model_name}")

    def generate_from_prompt(self,
                            user_query: str,
                            num_variations: int = 20,
                            max_retries: int = 3) -> List[LogoVariation]:
        """
        Generate multiple logo variations from natural language.

        Args:
            user_query: Natural language description (e.g., "minimalist tech logos conveying trust")
            num_variations: Number of different designs to generate
            max_retries: Maximum retry attempts per variation

        Returns:
            List of LogoVariation objects
        """
        logger.info(f"Generating {num_variations} variations for query: '{user_query}'")

        # Parse query to understand key requirements
        parsed_query = self._parse_user_query(user_query)

        variations = []
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")

            # Build unique prompt for this variation
            prompt = self._build_variation_prompt(parsed_query, variation_num=i+1)

            # Generate with retries
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    variation = self._parse_variation_response(response.text, parsed_query)

                    if variation:
                        variations.append(variation)
                        logger.info(f"  ✓ Variation {i+1} generated (complexity: {variation.estimated_complexity})")
                        # Rate limiting: 6 seconds between calls (10 calls/min max)
                        time.sleep(6)
                        break
                    else:
                        logger.warning(f"  ⚠ Failed to parse variation {i+1}, attempt {attempt+1}")

                except Exception as e:
                    logger.error(f"  ✗ Error generating variation {i+1}, attempt {attempt+1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(6)
            else:
                logger.warning(f"  ⚠ Skipping variation {i+1} after {max_retries} failed attempts")

        logger.info(f"Successfully generated {len(variations)}/{num_variations} variations")
        return variations

    def generate_targeted(self,
                         base_prompt: str,
                         behavioral_target: Dict,
                         max_retries: int = 3) -> Optional[LogoVariation]:
        """
        Generate logo targeting specific behavioral characteristics.

        Args:
            base_prompt: Base design requirements (company, industry, style)
            behavioral_target: Target characteristics {complexity: 0.7, style: 0.3, symmetry: 0.8, ...}
            max_retries: Maximum retry attempts

        Returns:
            LogoVariation targeting the specified behavior
        """
        logger.info(f"Generating targeted logo with behavior: {behavioral_target}")

        # Build targeted prompt
        prompt = self._build_targeted_prompt(base_prompt, behavioral_target)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                variation = self._parse_variation_response(
                    response.text,
                    {"query": base_prompt, "target_behavior": behavioral_target}
                )

                if variation:
                    logger.info(f"✓ Targeted logo generated (complexity: {variation.estimated_complexity})")
                    return variation

            except Exception as e:
                logger.error(f"Error generating targeted logo, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        logger.error("Failed to generate targeted logo after all retries")
        return None

    def _parse_user_query(self, query: str) -> Dict:
        """Parse natural language query into structured requirements"""
        query_lower = query.lower()

        # Extract style keywords
        style_keywords = []
        styles = ['minimalist', 'minimal', 'modern', 'geometric', 'organic', 'abstract',
                 'bold', 'elegant', 'playful', 'professional', 'tech', 'futuristic']
        for style in styles:
            if style in query_lower:
                style_keywords.append(style)

        # Extract emotion/feeling
        emotions = {
            'trust': ['trust', 'trustworthy', 'reliable', 'secure'],
            'innovation': ['innovation', 'innovative', 'cutting-edge', 'forward'],
            'friendly': ['friendly', 'approachable', 'warm', 'welcoming'],
            'professional': ['professional', 'corporate', 'business'],
            'playful': ['playful', 'fun', 'whimsical', 'creative']
        }

        detected_emotions = []
        for emotion, keywords in emotions.items():
            if any(kw in query_lower for kw in keywords):
                detected_emotions.append(emotion)

        # Extract quantity if specified
        quantity_match = re.search(r'(\d+)\s+(?:logos?|designs?|variations?)', query_lower)
        quantity = int(quantity_match.group(1)) if quantity_match else None

        return {
            'query': query,
            'style_keywords': style_keywords or ['modern', 'professional'],
            'emotions': detected_emotions or ['professional'],
            'quantity': quantity
        }

    def _build_variation_prompt(self, parsed_query: Dict, variation_num: int) -> str:
        """Build prompt for generating a specific variation"""
        styles = ', '.join(parsed_query['style_keywords'])
        emotions = ', '.join(parsed_query['emotions'])

        # Add diversity instructions based on variation number
        diversity_prompts = [
            "Focus on geometric shapes and symmetry",
            "Emphasize organic curves and flowing lines",
            "Use negative space creatively",
            "Create a bold, minimal design",
            "Design with circular motifs",
            "Use angular, triangular elements",
            "Incorporate abstract patterns",
            "Design with layered depth",
            "Use radial symmetry",
            "Create an asymmetric, dynamic design",
        ]

        diversity_instruction = diversity_prompts[(variation_num - 1) % len(diversity_prompts)]

        prompt = f"""You are an expert logo designer. Create a professional SVG logo based on these requirements:

REQUIREMENTS:
- Query: {parsed_query['query']}
- Style: {styles}
- Emotional tone: {emotions}
- Variation approach: {diversity_instruction}

DESIGN PRINCIPLES:
1. Simplicity: Use 15-40 SVG elements (optimal for recognition)
2. Scalability: Must work from 16px (favicon) to 1024px
3. Memorability: Distinctive and easy to remember
4. Versatility: Works in color and monochrome
5. Timelessness: Avoid trendy effects

TECHNICAL REQUIREMENTS:
- Use viewBox="0 0 200 200"
- Maximum 3 colors
- NO text elements (purely graphical)
- Valid SVG syntax
- Clean, structured code

OUTPUT FORMAT:

## DESIGN RATIONALE
[Explain your design concept in 2-3 sentences. What visual metaphor are you using? Why?]

## STYLE DESCRIPTION
[Describe the style in design terms: geometric/organic, symmetric/asymmetric, color palette, mood]

## COMPLEXITY ESTIMATE
[Number of SVG elements: X]

## FITNESS ESTIMATE
[Estimated quality score 0-100 based on design principles]

## SVG CODE
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Your logo design here -->
</svg>
```

Generate the logo now:"""

        return prompt

    def _build_targeted_prompt(self, base_prompt: str, behavioral_target: Dict) -> str:
        """Build prompt for targeted behavioral generation"""

        # Translate behavioral targets to design instructions
        complexity = behavioral_target.get('complexity', 0.5)
        style = behavioral_target.get('style', 0.5)
        symmetry = behavioral_target.get('symmetry', 0.5)
        color_richness = behavioral_target.get('color_richness', 0.5)

        # Complexity instructions
        if complexity < 0.3:
            complexity_instruction = "Very simple design with 10-15 elements"
        elif complexity < 0.5:
            complexity_instruction = "Simple design with 15-25 elements"
        elif complexity < 0.7:
            complexity_instruction = "Moderate complexity with 25-35 elements"
        else:
            complexity_instruction = "Complex design with 35-45 elements"

        # Style instructions
        if style < 0.3:
            style_instruction = "Purely geometric (circles, rectangles, straight lines)"
        elif style < 0.7:
            style_instruction = "Mix of geometric and organic shapes"
        else:
            style_instruction = "Organic, curved shapes (bezier curves, ellipses)"

        # Symmetry instructions
        if symmetry < 0.3:
            symmetry_instruction = "Asymmetric, dynamic composition"
        elif symmetry < 0.7:
            symmetry_instruction = "Partially symmetric design"
        else:
            symmetry_instruction = "Perfect vertical or radial symmetry"

        # Color instructions
        if color_richness < 0.25:
            color_instruction = "Monochrome (1 color only)"
        elif color_richness < 0.5:
            color_instruction = "Duotone (2 colors)"
        elif color_richness < 0.75:
            color_instruction = "Tritone (3 colors)"
        else:
            color_instruction = "Polychromatic (4+ colors)"

        prompt = f"""You are an expert logo designer. Create a professional SVG logo with SPECIFIC behavioral characteristics.

BASE REQUIREMENTS:
{base_prompt}

BEHAVIORAL TARGETS (CRITICAL - MUST MATCH):
1. Complexity: {complexity_instruction}
2. Style: {style_instruction}
3. Symmetry: {symmetry_instruction}
4. Colors: {color_instruction}

IMPORTANT: Your design MUST match these behavioral targets as closely as possible.

TECHNICAL REQUIREMENTS:
- Use viewBox="0 0 200 200"
- NO text elements
- Valid SVG syntax
- Clean code

OUTPUT FORMAT:

## DESIGN RATIONALE
[Explain how your design meets the behavioral targets]

## STYLE DESCRIPTION
[Describe the style and characteristics]

## COMPLEXITY ESTIMATE
[Number of SVG elements: X]

## FITNESS ESTIMATE
[Estimated quality score 0-100]

## SVG CODE
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Your logo design here -->
</svg>
```

Generate the logo now:"""

        return prompt

    def _parse_variation_response(self, response_text: str, context: Dict) -> Optional[LogoVariation]:
        """Parse LLM response into LogoVariation object"""
        try:
            # Extract sections
            rationale = self._extract_section(response_text, "DESIGN RATIONALE")
            style_desc = self._extract_section(response_text, "STYLE DESCRIPTION")
            svg_code = self._extract_svg_code(response_text)

            # Extract complexity estimate
            complexity_match = re.search(r'(?:Number of SVG elements|Complexity):\s*(\d+)', response_text, re.IGNORECASE)
            complexity = int(complexity_match.group(1)) if complexity_match else 25

            # Extract fitness estimate
            fitness_match = re.search(r'(?:Estimated quality score|FITNESS ESTIMATE).*?(\d+)', response_text, re.IGNORECASE | re.DOTALL)
            fitness = float(fitness_match.group(1)) if fitness_match else 75.0

            if not svg_code or '<svg' not in svg_code:
                logger.warning("No valid SVG code found in response")
                return None

            return LogoVariation(
                svg_code=svg_code,
                design_rationale=rationale or "Design created based on requirements",
                style_description=style_desc or "Professional logo design",
                estimated_complexity=complexity,
                estimated_fitness=fitness,
                metadata=context
            )

        except Exception as e:
            logger.error(f"Error parsing variation response: {e}")
            return None

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract content from a markdown section"""
        pattern = rf'##\s*{section_name}\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_svg_code(self, text: str) -> Optional[str]:
        """Extract SVG code from response"""
        patterns = [
            r'```svg\s*(.*?)```',
            r'```xml\s*(.*?)```',
            r'```\s*(<svg[^>]*>.*?</svg>)\s*```',
            r'(<svg[^>]*>.*?</svg>)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                svg = match.group(1).strip()
                if '<svg' in svg:
                    return svg

        return None

    def save_variations(self, variations: List[LogoVariation], output_dir: str, prefix: str = "logo"):
        """
        Save logo variations to files

        Args:
            variations: List of LogoVariation objects
            output_dir: Directory to save files
            prefix: Filename prefix
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save each variation
        for i, var in enumerate(variations, 1):
            # Save SVG
            svg_path = os.path.join(output_dir, f"{prefix}_{i:03d}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(var.svg_code)

            # Save metadata
            meta_path = os.path.join(output_dir, f"{prefix}_{i:03d}_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(var), f, indent=2)

        # Save summary
        summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
        summary = {
            'total_variations': len(variations),
            'avg_complexity': sum(v.estimated_complexity for v in variations) / len(variations),
            'avg_fitness': sum(v.estimated_fitness for v in variations) / len(variations),
            'variations': [
                {
                    'id': i,
                    'complexity': v.estimated_complexity,
                    'fitness': v.estimated_fitness,
                    'rationale': v.design_rationale[:100] + '...'
                }
                for i, v in enumerate(variations, 1)
            ]
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved {len(variations)} variations to {output_dir}")
        logger.info(f"Summary: avg_complexity={summary['avg_complexity']:.1f}, avg_fitness={summary['avg_fitness']:.1f}")


def demo():
    """Demonstrate LLM logo generator capabilities"""
    print("="*80)
    print("LLM LOGO GENERATOR DEMO")
    print("="*80)

    generator = LLMLogoGenerator()

    # Demo 1: Generate variations
    print("\n" + "="*80)
    print("DEMO 1: Generate Multiple Variations")
    print("="*80)

    query = "minimalist tech logos conveying innovation and trust"
    variations = generator.generate_from_prompt(query, num_variations=5)

    print(f"\nGenerated {len(variations)} variations:")
    for i, var in enumerate(variations, 1):
        print(f"\n  Variation {i}:")
        print(f"    Complexity: {var.estimated_complexity} elements")
        print(f"    Fitness: {var.estimated_fitness}/100")
        print(f"    Rationale: {var.design_rationale[:80]}...")

    # Save variations
    output_dir = "/home/luis/svg-logo-ai/output/llm_generator_demo"
    generator.save_variations(variations, output_dir, prefix="tech_logo")
    print(f"\n✓ Saved to: {output_dir}")

    # Demo 2: Targeted generation
    print("\n" + "="*80)
    print("DEMO 2: Generate with Behavioral Targets")
    print("="*80)

    base_prompt = "Professional logo for 'DataFlow' - a data analytics company"
    behavioral_target = {
        'complexity': 0.7,   # Moderately complex
        'style': 0.3,        # Geometric
        'symmetry': 0.8,     # Highly symmetric
        'color_richness': 0.25  # Duotone
    }

    targeted = generator.generate_targeted(base_prompt, behavioral_target)

    if targeted:
        print(f"\nTargeted logo generated:")
        print(f"  Complexity: {targeted.estimated_complexity} elements")
        print(f"  Fitness: {targeted.estimated_fitness}/100")
        print(f"  Rationale: {targeted.design_rationale[:100]}...")

        # Save targeted logo
        generator.save_variations([targeted], output_dir, prefix="dataflow_targeted")
        print(f"\n✓ Saved to: {output_dir}")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo()
