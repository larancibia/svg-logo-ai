"""
LLM Logo Evaluator
==================
Uses LLM as a fitness judge to evaluate logo quality across multiple dimensions.

The LLM acts as a design expert, providing nuanced evaluation of aesthetic
quality, brand alignment, and emotional impact.
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


class LLMLogoEvaluator:
    """
    LLM-based logo quality evaluator

    Evaluates logos across multiple dimensions using LLM as expert judge.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize LLM Logo Evaluator

        Args:
            model_name: Gemini model to use
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized LLMLogoEvaluator with model: {model_name}")

    def evaluate_fitness(self,
                        logo_svg: str,
                        user_query: str,
                        max_retries: int = 3) -> Dict[str, float]:
        """
        Evaluate logo quality across multiple dimensions.

        Args:
            logo_svg: SVG code to evaluate
            user_query: Original design requirements/query
            max_retries: Maximum retry attempts

        Returns:
            Dict with scores:
                - aesthetic: 0-100
                - match_to_query: 0-100
                - professionalism: 0-100
                - originality: 0-100
                - emotional_impact: 0-100
                - overall: weighted average
        """
        logger.info("Evaluating logo fitness")

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(logo_svg, user_query)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                scores = self._parse_evaluation_response(response.text)

                if scores:
                    logger.info(f"✓ Evaluation successful: overall={scores['overall']:.1f}")
                    return scores
                else:
                    logger.warning(f"⚠ Failed to parse scores (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Evaluation error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        # Fallback: return default scores
        logger.warning("Failed to evaluate, returning default scores")
        return {
            'aesthetic': 50.0,
            'match_to_query': 50.0,
            'professionalism': 50.0,
            'originality': 50.0,
            'emotional_impact': 50.0,
            'overall': 50.0
        }

    def extract_emotional_tone(self,
                               logo_svg: str,
                               max_retries: int = 3) -> float:
        """
        Extract emotional tone from logo.

        NEW behavioral dimension:
        0.0 = serious/professional
        1.0 = playful/friendly

        Args:
            logo_svg: SVG code
            max_retries: Maximum retry attempts

        Returns:
            Float between 0.0 and 1.0
        """
        logger.info("Extracting emotional tone")

        # Build emotion extraction prompt
        prompt = self._build_emotion_prompt(logo_svg)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                score = self._parse_emotion_response(response.text)

                if score is not None:
                    logger.info(f"✓ Emotional tone extracted: {score:.2f}")
                    return score
                else:
                    logger.warning(f"⚠ Failed to parse emotion (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Emotion extraction error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        # Fallback: neutral
        logger.warning("Failed to extract emotion, returning neutral (0.5)")
        return 0.5

    def critique_and_suggest(self,
                            logo_svg: str,
                            user_query: str,
                            max_retries: int = 3) -> Dict:
        """
        Detailed critique with improvement suggestions.

        Used for learning/debugging.

        Args:
            logo_svg: SVG code
            user_query: Original requirements
            max_retries: Maximum retry attempts

        Returns:
            Dict with:
                - strengths: List[str]
                - weaknesses: List[str]
                - suggestions: List[str]
                - overall_assessment: str
        """
        logger.info("Generating critique and suggestions")

        # Build critique prompt
        prompt = self._build_critique_prompt(logo_svg, user_query)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)
                critique = self._parse_critique_response(response.text)

                if critique:
                    logger.info(f"✓ Critique generated")
                    return critique
                else:
                    logger.warning(f"⚠ Failed to parse critique (attempt {attempt+1})")

            except Exception as e:
                logger.error(f"✗ Critique error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)

        # Fallback
        logger.warning("Failed to generate critique, returning default")
        return {
            'strengths': ['Design created'],
            'weaknesses': ['Unable to evaluate'],
            'suggestions': ['Try manual review'],
            'overall_assessment': 'Evaluation failed'
        }

    def _build_evaluation_prompt(self, logo_svg: str, user_query: str) -> str:
        """Build prompt for comprehensive evaluation"""
        prompt = f"""You are an expert logo design critic. Evaluate this SVG logo across multiple professional dimensions.

ORIGINAL REQUIREMENTS:
{user_query}

LOGO TO EVALUATE:
```svg
{logo_svg}
```

EVALUATION CRITERIA:

Rate each dimension from 0-100:

1. AESTHETIC QUALITY (0-100)
   - Visual appeal and design harmony
   - Color balance and composition
   - Professional polish

2. MATCH TO QUERY (0-100)
   - How well does it fulfill the requirements?
   - Does it convey the intended message?
   - Appropriate for the use case?

3. PROFESSIONALISM (0-100)
   - Suitable for commercial use
   - Timeless vs trendy
   - Versatility across contexts

4. ORIGINALITY (0-100)
   - Uniqueness and distinctiveness
   - Avoids clichés
   - Memorable design

5. EMOTIONAL IMPACT (0-100)
   - Evokes appropriate emotion
   - Strong brand personality
   - Engaging and memorable

OUTPUT FORMAT:

## SCORES
AESTHETIC: [0-100]
MATCH_TO_QUERY: [0-100]
PROFESSIONALISM: [0-100]
ORIGINALITY: [0-100]
EMOTIONAL_IMPACT: [0-100]

## OVERALL
SCORE: [0-100]

Provide your evaluation now:"""

        return prompt

    def _build_emotion_prompt(self, logo_svg: str) -> str:
        """Build prompt for emotion extraction"""
        prompt = f"""You are an expert in design psychology. Analyze the EMOTIONAL TONE of this logo.

LOGO:
```svg
{logo_svg}
```

TASK:
Rate the emotional tone on a scale from 0.0 to 1.0:

0.0 = SERIOUS/PROFESSIONAL
- Formal, corporate, authoritative
- Straight lines, geometric shapes
- Muted colors, minimal design
- Conservative, traditional

0.5 = BALANCED
- Mix of formal and approachable
- Moderate complexity

1.0 = PLAYFUL/FRIENDLY
- Whimsical, fun, approachable
- Rounded shapes, organic curves
- Bright colors, dynamic design
- Creative, modern

OUTPUT FORMAT:
EMOTIONAL_TONE: [0.0-1.0]

Provide your assessment now:"""

        return prompt

    def _build_critique_prompt(self, logo_svg: str, user_query: str) -> str:
        """Build prompt for detailed critique"""
        prompt = f"""You are a senior logo design consultant. Provide a detailed critique of this logo.

REQUIREMENTS:
{user_query}

LOGO:
```svg
{logo_svg}
```

TASK:
Provide a comprehensive design critique.

OUTPUT FORMAT:

## STRENGTHS
- [List 2-4 strong points]

## WEAKNESSES
- [List 2-4 areas for improvement]

## SUGGESTIONS
- [List 2-4 specific improvement recommendations]

## OVERALL ASSESSMENT
[2-3 sentence summary of the design quality]

Provide your critique now:"""

        return prompt

    def _parse_evaluation_response(self, text: str) -> Optional[Dict[str, float]]:
        """Parse evaluation scores from LLM response"""
        try:
            # Extract scores using regex
            scores = {}

            patterns = {
                'aesthetic': r'AESTHETIC[:\s]+(\d+(?:\.\d+)?)',
                'match_to_query': r'MATCH_TO_QUERY[:\s]+(\d+(?:\.\d+)?)',
                'professionalism': r'PROFESSIONALISM[:\s]+(\d+(?:\.\d+)?)',
                'originality': r'ORIGINALITY[:\s]+(\d+(?:\.\d+)?)',
                'emotional_impact': r'EMOTIONAL_IMPACT[:\s]+(\d+(?:\.\d+)?)',
                'overall': r'(?:OVERALL[:\s]+)?SCORE[:\s]+(\d+(?:\.\d+)?)'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))

            # Calculate overall if not provided
            if 'overall' not in scores and len(scores) >= 5:
                # Weighted average (aesthetic and match_to_query are most important)
                weights = {
                    'aesthetic': 0.25,
                    'match_to_query': 0.25,
                    'professionalism': 0.2,
                    'originality': 0.15,
                    'emotional_impact': 0.15
                }
                scores['overall'] = sum(scores[k] * weights[k] for k in weights if k in scores)

            # Validate we have all scores
            required_keys = ['aesthetic', 'match_to_query', 'professionalism', 'originality', 'emotional_impact', 'overall']
            if all(k in scores for k in required_keys):
                return scores
            else:
                logger.warning(f"Missing some scores: {set(required_keys) - set(scores.keys())}")
                return None

        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return None

    def _parse_emotion_response(self, text: str) -> Optional[float]:
        """Parse emotional tone score from response"""
        try:
            # Look for EMOTIONAL_TONE: X.X
            match = re.search(r'EMOTIONAL_TONE[:\s]+([\d.]+)', text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Ensure in range [0, 1]
                return max(0.0, min(1.0, score))

            # Alternative: look for any decimal between 0 and 1
            match = re.search(r'\b(0\.\d+|1\.0+)\b', text)
            if match:
                return float(match.group(1))

            return None

        except Exception as e:
            logger.error(f"Error parsing emotion response: {e}")
            return None

    def _parse_critique_response(self, text: str) -> Optional[Dict]:
        """Parse critique from LLM response"""
        try:
            critique = {
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'overall_assessment': ''
            }

            # Extract strengths
            strengths_match = re.search(r'##\s*STRENGTHS\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL | re.IGNORECASE)
            if strengths_match:
                strengths_text = strengths_match.group(1)
                critique['strengths'] = [s.strip('- ').strip() for s in strengths_text.split('\n') if s.strip().startswith('-')]

            # Extract weaknesses
            weaknesses_match = re.search(r'##\s*WEAKNESSES\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL | re.IGNORECASE)
            if weaknesses_match:
                weaknesses_text = weaknesses_match.group(1)
                critique['weaknesses'] = [s.strip('- ').strip() for s in weaknesses_text.split('\n') if s.strip().startswith('-')]

            # Extract suggestions
            suggestions_match = re.search(r'##\s*SUGGESTIONS\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL | re.IGNORECASE)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                critique['suggestions'] = [s.strip('- ').strip() for s in suggestions_text.split('\n') if s.strip().startswith('-')]

            # Extract overall assessment
            assessment_match = re.search(r'##\s*OVERALL ASSESSMENT\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL | re.IGNORECASE)
            if assessment_match:
                critique['overall_assessment'] = assessment_match.group(1).strip()

            return critique

        except Exception as e:
            logger.error(f"Error parsing critique response: {e}")
            return None


def demo():
    """Demonstrate LLM evaluator capabilities"""
    print("="*80)
    print("LLM LOGO EVALUATOR DEMO")
    print("="*80)

    evaluator = LLMLogoEvaluator()

    # Test logo
    test_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="70" fill="#2563eb"/>
  <circle cx="100" cy="100" r="50" fill="#ffffff"/>
  <circle cx="100" cy="100" r="30" fill="#1e40af"/>
</svg>"""

    user_query = "Professional logo for 'CircleFlow' - a modern tech company focusing on seamless integration"

    # Demo 1: Evaluate fitness
    print("\n" + "="*80)
    print("DEMO 1: Comprehensive Evaluation")
    print("="*80)

    scores = evaluator.evaluate_fitness(test_svg, user_query)

    print("\nEvaluation Scores:")
    print(f"  Aesthetic Quality: {scores['aesthetic']:.1f}/100")
    print(f"  Match to Query: {scores['match_to_query']:.1f}/100")
    print(f"  Professionalism: {scores['professionalism']:.1f}/100")
    print(f"  Originality: {scores['originality']:.1f}/100")
    print(f"  Emotional Impact: {scores['emotional_impact']:.1f}/100")
    print(f"  ─" * 40)
    print(f"  OVERALL: {scores['overall']:.1f}/100")

    # Demo 2: Extract emotional tone
    print("\n" + "="*80)
    print("DEMO 2: Emotional Tone Extraction")
    print("="*80)

    emotion = evaluator.extract_emotional_tone(test_svg)

    print(f"\nEmotional Tone: {emotion:.2f}")
    if emotion < 0.3:
        print("  → Serious/Professional")
    elif emotion < 0.7:
        print("  → Balanced")
    else:
        print("  → Playful/Friendly")

    # Demo 3: Critique and suggestions
    print("\n" + "="*80)
    print("DEMO 3: Detailed Critique")
    print("="*80)

    critique = evaluator.critique_and_suggest(test_svg, user_query)

    print("\nStrengths:")
    for strength in critique['strengths']:
        print(f"  + {strength}")

    print("\nWeaknesses:")
    for weakness in critique['weaknesses']:
        print(f"  - {weakness}")

    print("\nSuggestions:")
    for suggestion in critique['suggestions']:
        print(f"  → {suggestion}")

    print("\nOverall Assessment:")
    print(f"  {critique['overall_assessment']}")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo()
