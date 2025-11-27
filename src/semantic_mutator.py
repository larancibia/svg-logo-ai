"""
Semantic Mutator
================
LLM-guided mutation operators for intelligent logo evolution.

Unlike random mutations, semantic mutations understand design concepts and
modify logos meaningfully toward specific goals.
"""

import os
import re
import time
import logging
from typing import Dict, Optional, Tuple
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMutator:
    """
    LLM-guided mutation operators for logo evolution

    Mutations are semantic and goal-directed, not random.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Semantic Mutator

        Args:
            model_name: Gemini model to use
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized SemanticMutator with model: {model_name}")

    def mutate_toward_behavior(self,
                               logo_svg: str,
                               current_behavior: Dict,
                               target_behavior: Dict,
                               user_intent: str,
                               max_retries: int = 3) -> Optional[str]:
        """
        Intelligently mutate logo toward target behavioral characteristics.

        Examples:
        - Current complexity 0.3 → Target 0.7: "Add 10-15 geometric elements"
        - Current symmetry 0.2 → Target 0.9: "Add mirror symmetry"
        - Current emotion 'serious' → 'playful': "Make more whimsical and friendly"

        Args:
            logo_svg: Current SVG code
            current_behavior: Current behavioral features
            target_behavior: Target behavioral features
            user_intent: Original design intent (company, industry, etc.)
            max_retries: Maximum retry attempts

        Returns:
            Modified SVG code or None if failed
        """
        logger.info("Mutating logo toward target behavior")

        # Build mutation instructions
        instructions = self._build_mutation_instructions(current_behavior, target_behavior)

        # Build prompt
        prompt = self._build_mutation_prompt(logo_svg, instructions, user_intent)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    logger.info(f"✓ Mutation successful (attempt {attempt+1})")
                    return svg_code
                else:
                    logger.warning(f"⚠ Failed to extract SVG (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Mutation error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        logger.error("Failed to mutate logo after all retries")
        return None

    def semantic_crossover(self,
                          parent1_svg: str,
                          parent2_svg: str,
                          user_intent: str,
                          parent1_behavior: Optional[Dict] = None,
                          parent2_behavior: Optional[Dict] = None,
                          max_retries: int = 3) -> Optional[str]:
        """
        Combine two logos semantically, not just geometrically.

        LLM understands design concepts from both parents and creates
        a meaningful hybrid.

        Args:
            parent1_svg: First parent SVG
            parent2_svg: Second parent SVG
            user_intent: Design intent
            parent1_behavior: Optional behavioral features of parent1
            parent2_behavior: Optional behavioral features of parent2
            max_retries: Maximum retry attempts

        Returns:
            Child SVG code combining both parents
        """
        logger.info("Performing semantic crossover")

        # Build crossover prompt
        prompt = self._build_crossover_prompt(
            parent1_svg, parent2_svg, user_intent,
            parent1_behavior, parent2_behavior
        )

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    logger.info(f"✓ Crossover successful (attempt {attempt+1})")
                    return svg_code
                else:
                    logger.warning(f"⚠ Failed to extract SVG (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Crossover error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        logger.error("Failed to perform crossover after all retries")
        return None

    def directed_exploration(self,
                            logo_svg: str,
                            direction: str,
                            user_intent: str,
                            max_retries: int = 3) -> Optional[str]:
        """
        Mutate in semantic direction.

        Examples:
        - "more modern"
        - "more organic"
        - "more bold"
        - "simpler"
        - "more playful"

        Args:
            logo_svg: Current SVG code
            direction: Semantic direction for mutation
            user_intent: Design intent
            max_retries: Maximum retry attempts

        Returns:
            Modified SVG code
        """
        logger.info(f"Directed exploration: {direction}")

        # Build directed exploration prompt
        prompt = self._build_directed_prompt(logo_svg, direction, user_intent)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    logger.info(f"✓ Directed exploration successful (attempt {attempt+1})")
                    return svg_code
                else:
                    logger.warning(f"⚠ Failed to extract SVG (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Directed exploration error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        logger.error("Failed directed exploration after all retries")
        return None

    def _build_mutation_instructions(self,
                                    current_behavior: Dict,
                                    target_behavior: Dict) -> str:
        """Build mutation instructions based on behavioral delta"""
        instructions = []

        # Complexity
        curr_complexity = current_behavior.get('complexity', 0.5)
        targ_complexity = target_behavior.get('complexity', 0.5)
        delta_complexity = targ_complexity - curr_complexity

        if delta_complexity > 0.2:
            instructions.append(f"INCREASE COMPLEXITY: Add approximately {int(delta_complexity * 40)} more SVG elements (shapes, paths)")
        elif delta_complexity < -0.2:
            instructions.append(f"DECREASE COMPLEXITY: Remove approximately {int(abs(delta_complexity) * 40)} SVG elements to simplify")

        # Style (geometric vs organic)
        curr_style = current_behavior.get('style', 0.5)
        targ_style = target_behavior.get('style', 0.5)
        delta_style = targ_style - curr_style

        if delta_style > 0.2:
            instructions.append("MAKE MORE ORGANIC: Convert straight edges to curves, use bezier paths (C, Q commands)")
        elif delta_style < -0.2:
            instructions.append("MAKE MORE GEOMETRIC: Convert curves to straight lines, use circles/rectangles/polygons")

        # Symmetry
        curr_symmetry = current_behavior.get('symmetry', 0.5)
        targ_symmetry = target_behavior.get('symmetry', 0.5)
        delta_symmetry = targ_symmetry - curr_symmetry

        if delta_symmetry > 0.2:
            instructions.append("ADD SYMMETRY: Create vertical or radial symmetry")
        elif delta_symmetry < -0.2:
            instructions.append("BREAK SYMMETRY: Make design more asymmetric and dynamic")

        # Color richness
        curr_color = current_behavior.get('color_richness', 0.5)
        targ_color = target_behavior.get('color_richness', 0.5)
        delta_color = targ_color - curr_color

        if delta_color > 0.2:
            instructions.append("ADD COLORS: Introduce 1-2 new colors to the palette")
        elif delta_color < -0.2:
            instructions.append("SIMPLIFY COLORS: Reduce to fewer colors (max 2)")

        if not instructions:
            instructions.append("MINOR VARIATION: Make small refinements while preserving design")

        return "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))

    def _build_mutation_prompt(self,
                               logo_svg: str,
                               instructions: str,
                               user_intent: str) -> str:
        """Build prompt for behavioral mutation"""
        prompt = f"""You are an expert logo designer. Your task is to MODIFY an existing logo based on specific behavioral targets.

ORIGINAL DESIGN INTENT:
{user_intent}

CURRENT LOGO:
```svg
{logo_svg}
```

MUTATION INSTRUCTIONS:
{instructions}

IMPORTANT RULES:
1. START with the current logo and MODIFY it intelligently
2. Maintain the core design concept and brand identity
3. Make changes GRADUALLY (not extreme transformations)
4. Keep viewBox="0 0 200 200"
5. NO text elements (purely graphical)
6. Ensure valid, well-structured SVG
7. Preserve professional quality

OUTPUT FORMAT:
Return ONLY the modified SVG in a code block:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Modified logo here -->
</svg>
```

Generate the modified logo now:"""

        return prompt

    def _build_crossover_prompt(self,
                               parent1_svg: str,
                               parent2_svg: str,
                               user_intent: str,
                               parent1_behavior: Optional[Dict],
                               parent2_behavior: Optional[Dict]) -> str:
        """Build prompt for semantic crossover"""

        behavior_context = ""
        if parent1_behavior and parent2_behavior:
            behavior_context = f"""
PARENT 1 CHARACTERISTICS:
- Complexity: {parent1_behavior.get('complexity', 'unknown')}
- Style: {parent1_behavior.get('style', 'unknown')}
- Symmetry: {parent1_behavior.get('symmetry', 'unknown')}

PARENT 2 CHARACTERISTICS:
- Complexity: {parent2_behavior.get('complexity', 'unknown')}
- Style: {parent2_behavior.get('style', 'unknown')}
- Symmetry: {parent2_behavior.get('symmetry', 'unknown')}
"""

        prompt = f"""You are an expert logo designer. Your task is to create a CHILD logo by intelligently combining design elements from TWO parent logos.

DESIGN INTENT:
{user_intent}

{behavior_context}

PARENT LOGO 1:
```svg
{parent1_svg}
```

PARENT LOGO 2:
```svg
{parent2_svg}
```

CROSSOVER INSTRUCTIONS:
1. Analyze the design concepts, shapes, and styles of BOTH parents
2. Identify the strongest elements from each parent
3. Create a NEW logo that semantically combines both parents
4. The child should inherit characteristics from both parents but be DISTINCT
5. Maintain professional quality and coherence

TECHNICAL REQUIREMENTS:
- viewBox="0 0 200 200"
- NO text elements
- Valid SVG syntax
- Professional quality

OUTPUT FORMAT:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Child logo combining both parents -->
</svg>
```

Generate the child logo now:"""

        return prompt

    def _build_directed_prompt(self,
                              logo_svg: str,
                              direction: str,
                              user_intent: str) -> str:
        """Build prompt for directed exploration"""
        prompt = f"""You are an expert logo designer. Your task is to modify a logo in a specific SEMANTIC DIRECTION.

DESIGN INTENT:
{user_intent}

CURRENT LOGO:
```svg
{logo_svg}
```

DIRECTION OF CHANGE:
Make the logo {direction}

IMPORTANT GUIDELINES:
1. Understand what "{direction}" means in design terms
2. Modify the logo to move clearly in that direction
3. Maintain professional quality and brand coherence
4. Make noticeable but tasteful changes
5. Keep viewBox="0 0 200 200"
6. NO text elements
7. Valid SVG syntax

EXAMPLES OF SEMANTIC DIRECTIONS:
- "more modern" → cleaner lines, flatter design, sans-serif feel
- "more organic" → curves, natural shapes, flowing lines
- "more bold" → stronger colors, thicker lines, higher contrast
- "simpler" → fewer elements, cleaner composition
- "more playful" → rounded shapes, friendly curves, lighter feel

OUTPUT FORMAT:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <!-- Modified logo here -->
</svg>
```

Generate the modified logo now:"""

        return prompt

    def _extract_svg(self, text: str) -> Optional[str]:
        """Extract SVG code from LLM response"""
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


def demo():
    """Demonstrate semantic mutation capabilities"""
    print("="*80)
    print("SEMANTIC MUTATOR DEMO")
    print("="*80)

    mutator = SemanticMutator()

    # Demo 1: Behavioral mutation
    print("\n" + "="*80)
    print("DEMO 1: Mutate Toward Behavior")
    print("="*80)

    source_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

    current_behavior = {
        'complexity': 0.1,  # Very simple (2 elements)
        'style': 0.0,       # Pure geometric
        'symmetry': 1.0,    # Perfect symmetry
        'color_richness': 0.25  # Duotone
    }

    target_behavior = {
        'complexity': 0.6,  # More complex
        'style': 0.4,       # More organic
        'symmetry': 1.0,    # Keep symmetry
        'color_richness': 0.5  # More colors
    }

    user_intent = "Professional logo for 'TechFlow' - a technology company"

    mutated = mutator.mutate_toward_behavior(
        source_svg,
        current_behavior,
        target_behavior,
        user_intent
    )

    if mutated:
        print("\n✓ Behavioral mutation successful")
        print(f"  Original length: {len(source_svg)} chars")
        print(f"  Mutated length: {len(mutated)} chars")

        # Save results
        output_dir = "/home/luis/svg-logo-ai/output/semantic_mutator_demo"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/behavior_original.svg", 'w') as f:
            f.write(source_svg)
        with open(f"{output_dir}/behavior_mutated.svg", 'w') as f:
            f.write(mutated)

        print(f"  Saved to: {output_dir}")

    # Demo 2: Semantic crossover
    print("\n" + "="*80)
    print("DEMO 2: Semantic Crossover")
    print("="*80)

    parent1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="70" fill="#2563eb"/>
  <circle cx="100" cy="100" r="50" fill="#1e40af"/>
</svg>"""

    parent2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="50" y="50" width="100" height="100" fill="#10b981"/>
  <rect x="70" y="70" width="60" height="60" fill="#059669"/>
</svg>"""

    child = mutator.semantic_crossover(parent1, parent2, user_intent)

    if child:
        print("\n✓ Semantic crossover successful")
        print(f"  Parent1 length: {len(parent1)} chars")
        print(f"  Parent2 length: {len(parent2)} chars")
        print(f"  Child length: {len(child)} chars")

        with open(f"{output_dir}/crossover_parent1.svg", 'w') as f:
            f.write(parent1)
        with open(f"{output_dir}/crossover_parent2.svg", 'w') as f:
            f.write(parent2)
        with open(f"{output_dir}/crossover_child.svg", 'w') as f:
            f.write(child)

        print(f"  Saved to: {output_dir}")

    # Demo 3: Directed exploration
    print("\n" + "="*80)
    print("DEMO 3: Directed Exploration")
    print("="*80)

    directions = ["more modern", "more organic", "bolder"]

    for direction in directions:
        print(f"\nDirection: '{direction}'")

        modified = mutator.directed_exploration(source_svg, direction, user_intent)

        if modified:
            print(f"  ✓ Success")
            filename = direction.replace(" ", "_")
            with open(f"{output_dir}/directed_{filename}.svg", 'w') as f:
                f.write(modified)
        else:
            print(f"  ✗ Failed")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo()
