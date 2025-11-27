"""
Natural Language Query Parser
==============================
Parses natural language queries into structured search parameters for logo generation.

Transforms queries like:
"100 minimalist tech logos with circular motifs conveying innovation"

Into structured parameters for the QD system.
"""

import re
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a natural language query"""
    original_query: str
    quantity: int
    style_keywords: List[str]
    emotion_target: str
    color_preferences: List[str]
    motifs: List[str]
    constraints: Dict
    behavioral_preferences: Dict
    industry: Optional[str] = None
    company_name: Optional[str] = None


class NLQueryParser:
    """
    Natural Language Query Parser for Logo Generation

    Extracts structured parameters from natural language descriptions.
    """

    # Style vocabulary
    STYLE_KEYWORDS = {
        'minimalist', 'minimal', 'simple', 'clean',
        'modern', 'contemporary', 'futuristic',
        'geometric', 'angular', 'structured',
        'organic', 'flowing', 'curved', 'natural',
        'abstract', 'conceptual',
        'bold', 'strong', 'powerful',
        'elegant', 'refined', 'sophisticated',
        'playful', 'fun', 'whimsical',
        'professional', 'corporate', 'business',
        'tech', 'technical', 'digital',
        'vintage', 'retro', 'classic',
        'flat', 'material', '3d',
    }

    # Emotion vocabulary
    EMOTIONS = {
        'trust': ['trust', 'trustworthy', 'reliable', 'dependable', 'secure', 'safe'],
        'innovation': ['innovation', 'innovative', 'cutting-edge', 'forward-thinking', 'pioneering', 'advanced'],
        'friendly': ['friendly', 'approachable', 'warm', 'welcoming', 'inviting'],
        'professional': ['professional', 'corporate', 'business', 'serious', 'formal'],
        'playful': ['playful', 'fun', 'whimsical', 'creative', 'entertaining'],
        'energetic': ['energetic', 'dynamic', 'vibrant', 'active', 'lively'],
        'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxing'],
        'luxury': ['luxury', 'premium', 'exclusive', 'high-end', 'sophisticated'],
        'eco': ['eco', 'green', 'sustainable', 'natural', 'organic', 'environmental'],
    }

    # Motifs vocabulary
    MOTIFS = {
        'circular', 'round', 'circle', 'rings',
        'triangular', 'triangle', 'angular',
        'square', 'rectangular', 'box',
        'hexagonal', 'hexagon',
        'star', 'stars',
        'arrow', 'arrows',
        'wave', 'waves', 'flowing',
        'leaf', 'leaves', 'plant',
        'tech', 'circuit', 'digital',
        'abstract', 'geometric',
    }

    # Industry keywords
    INDUSTRIES = {
        'technology', 'tech', 'software', 'ai', 'artificial intelligence', 'machine learning',
        'healthcare', 'health', 'medical', 'wellness',
        'finance', 'financial', 'banking', 'investment',
        'food', 'restaurant', 'culinary',
        'retail', 'ecommerce', 'shopping',
        'education', 'learning', 'academic',
        'real estate', 'property',
        'consulting', 'advisory',
        'manufacturing', 'industrial',
        'creative', 'design', 'agency',
    }

    # Color vocabulary
    COLORS = {
        'blue': '#2563eb',
        'red': '#dc2626',
        'green': '#10b981',
        'yellow': '#eab308',
        'purple': '#7c3aed',
        'orange': '#f97316',
        'pink': '#ec4899',
        'teal': '#14b8a6',
        'indigo': '#4f46e5',
        'black': '#000000',
        'white': '#ffffff',
        'gray': '#6b7280',
        'grey': '#6b7280',
    }

    def __init__(self):
        """Initialize NL Query Parser"""
        logger.info("Initialized NLQueryParser")

    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language into structured search parameters.

        Example:
        "100 minimalist tech logos with circular motifs conveying innovation"

        Returns:
        ParsedQuery with all extracted parameters
        """
        logger.info(f"Parsing query: '{query}'")

        query_lower = query.lower()

        # Extract quantity
        quantity = self._extract_quantity(query_lower)

        # Extract style keywords
        style_keywords = self._extract_style_keywords(query_lower)

        # Extract emotion target
        emotion_target = self._extract_emotion_target(query_lower)

        # Extract color preferences
        color_preferences = self._extract_color_preferences(query_lower)

        # Extract motifs
        motifs = self._extract_motifs(query_lower)

        # Extract industry
        industry = self._extract_industry(query_lower)

        # Extract company name (if quoted)
        company_name = self._extract_company_name(query)

        # Build constraints
        constraints = self._build_constraints(query_lower, style_keywords)

        # Build behavioral preferences
        behavioral_preferences = self._build_behavioral_preferences(
            style_keywords, motifs, emotion_target
        )

        parsed = ParsedQuery(
            original_query=query,
            quantity=quantity,
            style_keywords=style_keywords,
            emotion_target=emotion_target,
            color_preferences=color_preferences,
            motifs=motifs,
            constraints=constraints,
            behavioral_preferences=behavioral_preferences,
            industry=industry,
            company_name=company_name
        )

        logger.info(f"Parsed: quantity={quantity}, styles={len(style_keywords)}, emotion={emotion_target}")
        return parsed

    def _extract_quantity(self, query: str) -> int:
        """Extract number of logos requested"""
        # Look for patterns like "100 logos", "50 designs", "20 variations"
        patterns = [
            r'(\d+)\s+(?:logos?|designs?|variations?|examples?)',
            r'^(\d+)\s+',  # Number at start
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                quantity = int(match.group(1))
                # Reasonable limits
                return max(1, min(1000, quantity))

        # Default
        return 20

    def _extract_style_keywords(self, query: str) -> List[str]:
        """Extract style keywords from query"""
        found_styles = []

        for style in self.STYLE_KEYWORDS:
            if style in query:
                found_styles.append(style)

        # If no styles found, infer from context
        if not found_styles:
            # Tech/digital context
            if any(word in query for word in ['tech', 'digital', 'software', 'ai', 'data']):
                found_styles.extend(['modern', 'clean', 'tech'])
            # Corporate context
            elif any(word in query for word in ['corporate', 'business', 'professional']):
                found_styles.extend(['professional', 'clean'])
            # Creative context
            elif any(word in query for word in ['creative', 'artistic', 'design']):
                found_styles.extend(['modern', 'creative'])
            else:
                # Default
                found_styles.extend(['modern', 'professional'])

        return found_styles

    def _extract_emotion_target(self, query: str) -> str:
        """Extract primary emotional target"""
        found_emotions = []

        for emotion, keywords in self.EMOTIONS.items():
            if any(kw in query for kw in keywords):
                found_emotions.append(emotion)

        # Return primary emotion
        if found_emotions:
            return found_emotions[0]

        # Default based on style
        if any(word in query for word in ['tech', 'digital', 'professional']):
            return 'trust'
        elif any(word in query for word in ['creative', 'fun', 'play']):
            return 'playful'
        else:
            return 'professional'

    def _extract_color_preferences(self, query: str) -> List[str]:
        """Extract color preferences"""
        found_colors = []

        for color, hex_code in self.COLORS.items():
            if color in query:
                found_colors.append(hex_code)

        return found_colors

    def _extract_motifs(self, query: str) -> List[str]:
        """Extract design motifs"""
        found_motifs = []

        for motif in self.MOTIFS:
            if motif in query:
                found_motifs.append(motif)

        return found_motifs

    def _extract_industry(self, query: str) -> Optional[str]:
        """Extract industry/domain"""
        for industry in self.INDUSTRIES:
            if industry in query:
                return industry

        return None

    def _extract_company_name(self, query: str) -> Optional[str]:
        """Extract company name if quoted"""
        # Look for text in quotes
        match = re.search(r'["\']([^"\']+)["\']', query)
        if match:
            return match.group(1)

        # Look for "for CompanyName"
        match = re.search(r'for\s+([A-Z][a-zA-Z]+)', query)
        if match:
            return match.group(1)

        return None

    def _build_constraints(self, query: str, style_keywords: List[str]) -> Dict:
        """Build design constraints"""
        constraints = {
            'max_complexity': 50,
            'min_complexity': 10,
            'max_colors': 3,
            'require_symmetry': False,
            'avoid_text': True,
        }

        # Adjust based on style
        if 'minimalist' in style_keywords or 'minimal' in style_keywords or 'simple' in style_keywords:
            constraints['max_complexity'] = 30
            constraints['max_colors'] = 2

        if 'complex' in query or 'detailed' in query or 'intricate' in query:
            constraints['max_complexity'] = 60
            constraints['min_complexity'] = 30

        if 'symmetric' in query or 'symmetry' in query or 'balanced' in query:
            constraints['require_symmetry'] = True

        if 'colorful' in query or 'multicolor' in query or 'polychromatic' in query:
            constraints['max_colors'] = 5

        if 'monochrome' in query or 'single color' in query:
            constraints['max_colors'] = 1

        return constraints

    def _build_behavioral_preferences(self,
                                      style_keywords: List[str],
                                      motifs: List[str],
                                      emotion: str) -> Dict:
        """Build behavioral preferences for QD system"""
        preferences = {
            'complexity': 0.5,      # 0-1 scale
            'style': 0.5,           # 0=geometric, 1=organic
            'symmetry': 0.5,        # 0=asymmetric, 1=symmetric
            'color_richness': 0.5,  # 0=monochrome, 1=polychromatic
        }

        # Complexity based on style
        if any(kw in style_keywords for kw in ['minimalist', 'minimal', 'simple', 'clean']):
            preferences['complexity'] = 0.3
        elif any(kw in style_keywords for kw in ['complex', 'detailed', 'intricate']):
            preferences['complexity'] = 0.8
        else:
            preferences['complexity'] = 0.5

        # Style (geometric vs organic)
        if any(kw in style_keywords for kw in ['geometric', 'angular', 'structured']):
            preferences['style'] = 0.2  # More geometric
        elif any(kw in style_keywords for kw in ['organic', 'flowing', 'curved', 'natural']):
            preferences['style'] = 0.8  # More organic
        else:
            preferences['style'] = 0.5

        # Symmetry
        if any(kw in style_keywords for kw in ['symmetric', 'balanced', 'formal']):
            preferences['symmetry'] = 0.8
        elif any(kw in style_keywords for kw in ['asymmetric', 'dynamic', 'playful']):
            preferences['symmetry'] = 0.3
        else:
            preferences['symmetry'] = 0.5

        # Color richness
        if any(kw in style_keywords for kw in ['colorful', 'vibrant', 'multicolor']):
            preferences['color_richness'] = 0.8
        elif any(kw in style_keywords for kw in ['monochrome', 'minimal', 'simple']):
            preferences['color_richness'] = 0.2
        else:
            preferences['color_richness'] = 0.5

        # Adjust based on motifs
        if any(m in motifs for m in ['circular', 'round', 'circle']):
            preferences['style'] = max(0, preferences['style'] - 0.1)  # More geometric
            preferences['symmetry'] = min(1, preferences['symmetry'] + 0.2)

        if any(m in motifs for m in ['wave', 'flowing', 'organic']):
            preferences['style'] = min(1, preferences['style'] + 0.3)  # More organic

        # Adjust based on emotion
        if emotion == 'playful':
            preferences['style'] = min(1, preferences['style'] + 0.2)
            preferences['color_richness'] = min(1, preferences['color_richness'] + 0.2)
        elif emotion == 'professional':
            preferences['style'] = max(0, preferences['style'] - 0.2)
            preferences['symmetry'] = min(1, preferences['symmetry'] + 0.1)

        return preferences

    def to_generation_prompt(self, parsed: ParsedQuery) -> str:
        """Convert parsed query to generation prompt"""
        styles = ', '.join(parsed.style_keywords) if parsed.style_keywords else 'modern, professional'
        motifs_str = ', '.join(parsed.motifs) if parsed.motifs else 'abstract'

        if parsed.company_name:
            prompt = f"Professional logo for '{parsed.company_name}'"
            if parsed.industry:
                prompt += f" - a {parsed.industry} company"
        elif parsed.industry:
            prompt = f"Professional logo for a {parsed.industry} company"
        else:
            prompt = f"Professional logo design"

        prompt += f"\nStyle: {styles}"
        prompt += f"\nEmotion: {parsed.emotion_target}"
        prompt += f"\nMotifs: {motifs_str}"

        if parsed.color_preferences:
            colors = ', '.join(parsed.color_preferences)
            prompt += f"\nColors: {colors}"

        return prompt


def demo():
    """Demonstrate NL query parser"""
    print("="*80)
    print("NATURAL LANGUAGE QUERY PARSER DEMO")
    print("="*80)

    parser = NLQueryParser()

    # Test queries
    test_queries = [
        "100 minimalist tech logos with circular motifs conveying innovation",
        "50 organic healthcare logos in green tones conveying trust and care",
        "Create a logo for 'DataFlow' - an AI analytics company - modern and professional",
        "Playful and colorful logos for a children's education startup",
        "Luxury brand logos with elegant geometric patterns",
        "20 symmetric corporate logos in blue and gray",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {query}")
        print('='*80)

        parsed = parser.parse_query(query)

        print(f"\nParsed Results:")
        print(f"  Quantity: {parsed.quantity}")
        print(f"  Company: {parsed.company_name or 'N/A'}")
        print(f"  Industry: {parsed.industry or 'N/A'}")
        print(f"  Styles: {', '.join(parsed.style_keywords)}")
        print(f"  Emotion: {parsed.emotion_target}")
        print(f"  Motifs: {', '.join(parsed.motifs) if parsed.motifs else 'N/A'}")
        print(f"  Colors: {', '.join(parsed.color_preferences) if parsed.color_preferences else 'N/A'}")

        print(f"\nBehavioral Preferences:")
        for key, value in parsed.behavioral_preferences.items():
            print(f"  {key}: {value:.2f}")

        print(f"\nConstraints:")
        for key, value in parsed.constraints.items():
            print(f"  {key}: {value}")

        print(f"\nGeneration Prompt:")
        prompt = parser.to_generation_prompt(parsed)
        print(f"  {prompt}")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo()
