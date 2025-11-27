#!/usr/bin/env python3
"""
Test Suite for LLM Components
==============================
Comprehensive testing of all LLM-related components for logo generation.

Tests:
1. LLMLogoGenerator - generation quality and variety
2. SemanticMutator - mutation effectiveness
3. LLMLogoEvaluator - evaluation consistency
4. NLQueryParser - parsing accuracy
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_logo_generator import LLMLogoGenerator, LogoVariation
from semantic_mutator import SemanticMutator
from llm_evaluator import LLMLogoEvaluator
from nl_query_parser import NLQueryParser
from behavior_characterization import BehaviorCharacterizer


class LLMComponentTester:
    """Comprehensive test suite for LLM components"""

    def __init__(self, output_dir: str = "/home/luis/svg-logo-ai/output/llm_tests"):
        """Initialize tester"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'component_results': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        print("="*80)
        print("LLM COMPONENTS TEST SUITE")
        print("="*80)
        print(f"Output directory: {output_dir}\n")

    def run_all_tests(self):
        """Run all component tests"""
        print("\n" + "="*80)
        print("STARTING ALL TESTS")
        print("="*80)

        # Test each component
        self.test_query_parser()
        self.test_logo_generator()
        self.test_semantic_mutator()
        self.test_evaluator()

        # Generate report
        self.generate_report()

    def test_query_parser(self):
        """Test NL Query Parser"""
        print("\n" + "="*80)
        print("TEST 1: Natural Language Query Parser")
        print("="*80)

        try:
            parser = NLQueryParser()

            test_queries = [
                "100 minimalist tech logos with circular motifs conveying innovation",
                "50 organic healthcare logos in green tones",
                "Logo for 'TechFlow' - modern AI company",
            ]

            results = {
                'total_tests': len(test_queries),
                'passed': 0,
                'queries': []
            }

            for query in test_queries:
                print(f"\nQuery: {query}")

                parsed = parser.parse_query(query)

                # Validate parsing
                checks = {
                    'has_quantity': parsed.quantity > 0,
                    'has_styles': len(parsed.style_keywords) > 0,
                    'has_emotion': parsed.emotion_target is not None,
                    'has_behavioral_prefs': len(parsed.behavioral_preferences) == 4,
                }

                passed = all(checks.values())
                results['passed'] += 1 if passed else 0

                query_result = {
                    'query': query,
                    'parsed': {
                        'quantity': parsed.quantity,
                        'styles': parsed.style_keywords,
                        'emotion': parsed.emotion_target,
                        'industry': parsed.industry,
                    },
                    'checks': checks,
                    'passed': passed
                }

                results['queries'].append(query_result)

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status}")
                print(f"    Quantity: {parsed.quantity}")
                print(f"    Styles: {', '.join(parsed.style_keywords)}")
                print(f"    Emotion: {parsed.emotion_target}")

            self.results['component_results']['query_parser'] = results
            self.results['tests_run'] += results['total_tests']
            self.results['tests_passed'] += results['passed']
            self.results['tests_failed'] += results['total_tests'] - results['passed']

            print(f"\n✓ Query Parser: {results['passed']}/{results['total_tests']} tests passed")

        except Exception as e:
            print(f"\n✗ Query Parser tests failed with error: {e}")
            self.results['component_results']['query_parser'] = {'error': str(e)}

    def test_logo_generator(self):
        """Test LLM Logo Generator"""
        print("\n" + "="*80)
        print("TEST 2: LLM Logo Generator")
        print("="*80)

        try:
            generator = LLMLogoGenerator()

            results = {
                'total_tests': 3,
                'passed': 0,
                'tests': []
            }

            # Test 1: Generate variations
            print("\nTest 2.1: Generate multiple variations")
            query = "minimalist tech logos"
            variations = generator.generate_from_prompt(query, num_variations=3)

            test1_passed = len(variations) >= 2  # At least 2 of 3 should succeed
            results['passed'] += 1 if test1_passed else 0

            test1_result = {
                'test': 'generate_variations',
                'requested': 3,
                'generated': len(variations),
                'passed': test1_passed
            }
            results['tests'].append(test1_result)

            print(f"  Generated: {len(variations)}/3 variations")
            print(f"  {('✓ PASS' if test1_passed else '✗ FAIL')}")

            # Save variations
            if variations:
                var_dir = f"{self.output_dir}/generator_variations"
                generator.save_variations(variations, var_dir, prefix="test_logo")
                print(f"  Saved to: {var_dir}")

            # Test 2: Targeted generation
            print("\nTest 2.2: Generate with behavioral targets")
            base_prompt = "Logo for DataFlow - analytics company"
            target_behavior = {
                'complexity': 0.7,
                'style': 0.3,
                'symmetry': 0.8,
                'color_richness': 0.25
            }

            targeted = generator.generate_targeted(base_prompt, target_behavior)
            test2_passed = targeted is not None

            results['passed'] += 1 if test2_passed else 0

            test2_result = {
                'test': 'targeted_generation',
                'success': test2_passed,
                'passed': test2_passed
            }
            results['tests'].append(test2_result)

            print(f"  {('✓ PASS' if test2_passed else '✗ FAIL')}")

            if targeted:
                # Validate behavioral match
                characterizer = BehaviorCharacterizer()
                actual_behavior = characterizer.characterize(targeted.svg_code)

                print(f"  Target complexity: {target_behavior['complexity']:.2f}")
                print(f"  Actual complexity bin: {actual_behavior['bins'][0]}/9")

            # Test 3: SVG validity
            print("\nTest 2.3: SVG validity check")
            valid_count = sum(1 for v in variations if '<svg' in v.svg_code and '</svg>' in v.svg_code)
            test3_passed = valid_count == len(variations)

            results['passed'] += 1 if test3_passed else 0

            test3_result = {
                'test': 'svg_validity',
                'valid': valid_count,
                'total': len(variations),
                'passed': test3_passed
            }
            results['tests'].append(test3_result)

            print(f"  Valid SVGs: {valid_count}/{len(variations)}")
            print(f"  {('✓ PASS' if test3_passed else '✗ FAIL')}")

            self.results['component_results']['logo_generator'] = results
            self.results['tests_run'] += results['total_tests']
            self.results['tests_passed'] += results['passed']
            self.results['tests_failed'] += results['total_tests'] - results['passed']

            print(f"\n✓ Logo Generator: {results['passed']}/{results['total_tests']} tests passed")

        except Exception as e:
            print(f"\n✗ Logo Generator tests failed with error: {e}")
            self.results['component_results']['logo_generator'] = {'error': str(e)}

    def test_semantic_mutator(self):
        """Test Semantic Mutator"""
        print("\n" + "="*80)
        print("TEST 3: Semantic Mutator")
        print("="*80)

        try:
            mutator = SemanticMutator()

            results = {
                'total_tests': 3,
                'passed': 0,
                'tests': []
            }

            source_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

            user_intent = "Professional tech logo"

            # Test 1: Behavioral mutation
            print("\nTest 3.1: Mutate toward behavior")
            current_behavior = {'complexity': 0.1, 'style': 0.0, 'symmetry': 1.0, 'color_richness': 0.25}
            target_behavior = {'complexity': 0.6, 'style': 0.4, 'symmetry': 1.0, 'color_richness': 0.5}

            mutated = mutator.mutate_toward_behavior(
                source_svg, current_behavior, target_behavior, user_intent
            )

            test1_passed = mutated is not None and '<svg' in mutated
            results['passed'] += 1 if test1_passed else 0

            results['tests'].append({
                'test': 'behavioral_mutation',
                'success': test1_passed,
                'passed': test1_passed
            })

            print(f"  {('✓ PASS' if test1_passed else '✗ FAIL')}")

            if mutated:
                with open(f"{self.output_dir}/mutation_test.svg", 'w') as f:
                    f.write(mutated)

            # Test 2: Semantic crossover
            print("\nTest 3.2: Semantic crossover")
            parent1 = source_svg
            parent2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="50" y="50" width="100" height="100" fill="#10b981"/>
</svg>"""

            child = mutator.semantic_crossover(parent1, parent2, user_intent)

            test2_passed = child is not None and '<svg' in child
            results['passed'] += 1 if test2_passed else 0

            results['tests'].append({
                'test': 'semantic_crossover',
                'success': test2_passed,
                'passed': test2_passed
            })

            print(f"  {('✓ PASS' if test2_passed else '✗ FAIL')}")

            if child:
                with open(f"{self.output_dir}/crossover_test.svg", 'w') as f:
                    f.write(child)

            # Test 3: Directed exploration
            print("\nTest 3.3: Directed exploration")
            modified = mutator.directed_exploration(source_svg, "more modern", user_intent)

            test3_passed = modified is not None and '<svg' in modified
            results['passed'] += 1 if test3_passed else 0

            results['tests'].append({
                'test': 'directed_exploration',
                'success': test3_passed,
                'passed': test3_passed
            })

            print(f"  {('✓ PASS' if test3_passed else '✗ FAIL')}")

            if modified:
                with open(f"{self.output_dir}/directed_test.svg", 'w') as f:
                    f.write(modified)

            self.results['component_results']['semantic_mutator'] = results
            self.results['tests_run'] += results['total_tests']
            self.results['tests_passed'] += results['passed']
            self.results['tests_failed'] += results['total_tests'] - results['passed']

            print(f"\n✓ Semantic Mutator: {results['passed']}/{results['total_tests']} tests passed")

        except Exception as e:
            print(f"\n✗ Semantic Mutator tests failed with error: {e}")
            self.results['component_results']['semantic_mutator'] = {'error': str(e)}

    def test_evaluator(self):
        """Test LLM Evaluator"""
        print("\n" + "="*80)
        print("TEST 4: LLM Logo Evaluator")
        print("="*80)

        try:
            evaluator = LLMLogoEvaluator()

            results = {
                'total_tests': 3,
                'passed': 0,
                'tests': []
            }

            test_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="70" fill="#2563eb"/>
  <circle cx="100" cy="100" r="50" fill="#ffffff"/>
  <circle cx="100" cy="100" r="30" fill="#1e40af"/>
</svg>"""

            user_query = "Professional logo for CircleFlow - seamless integration company"

            # Test 1: Comprehensive evaluation
            print("\nTest 4.1: Comprehensive evaluation")
            scores = evaluator.evaluate_fitness(test_svg, user_query)

            required_keys = ['aesthetic', 'match_to_query', 'professionalism', 'originality', 'emotional_impact', 'overall']
            test1_passed = all(k in scores for k in required_keys)

            results['passed'] += 1 if test1_passed else 0
            results['tests'].append({
                'test': 'comprehensive_evaluation',
                'has_all_scores': test1_passed,
                'scores': scores,
                'passed': test1_passed
            })

            print(f"  Overall score: {scores.get('overall', 'N/A')}/100")
            print(f"  {('✓ PASS' if test1_passed else '✗ FAIL')}")

            # Test 2: Emotional tone extraction
            print("\nTest 4.2: Emotional tone extraction")
            emotion = evaluator.extract_emotional_tone(test_svg)

            test2_passed = emotion is not None and 0.0 <= emotion <= 1.0

            results['passed'] += 1 if test2_passed else 0
            results['tests'].append({
                'test': 'emotion_extraction',
                'emotion_score': emotion,
                'in_range': test2_passed,
                'passed': test2_passed
            })

            print(f"  Emotional tone: {emotion:.2f}")
            print(f"  {('✓ PASS' if test2_passed else '✗ FAIL')}")

            # Test 3: Critique generation
            print("\nTest 4.3: Critique generation")
            critique = evaluator.critique_and_suggest(test_svg, user_query)

            required_critique_keys = ['strengths', 'weaknesses', 'suggestions', 'overall_assessment']
            test3_passed = all(k in critique for k in required_critique_keys)

            results['passed'] += 1 if test3_passed else 0
            results['tests'].append({
                'test': 'critique_generation',
                'has_all_sections': test3_passed,
                'num_strengths': len(critique.get('strengths', [])),
                'num_weaknesses': len(critique.get('weaknesses', [])),
                'num_suggestions': len(critique.get('suggestions', [])),
                'passed': test3_passed
            })

            print(f"  Strengths: {len(critique.get('strengths', []))} items")
            print(f"  Weaknesses: {len(critique.get('weaknesses', []))} items")
            print(f"  Suggestions: {len(critique.get('suggestions', []))} items")
            print(f"  {('✓ PASS' if test3_passed else '✗ FAIL')}")

            self.results['component_results']['evaluator'] = results
            self.results['tests_run'] += results['total_tests']
            self.results['tests_passed'] += results['passed']
            self.results['tests_failed'] += results['total_tests'] - results['passed']

            print(f"\n✓ Evaluator: {results['passed']}/{results['total_tests']} tests passed")

        except Exception as e:
            print(f"\n✗ Evaluator tests failed with error: {e}")
            self.results['component_results']['evaluator'] = {'error': str(e)}

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        total = self.results['tests_run']
        passed = self.results['tests_passed']
        failed = self.results['tests_failed']
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        print("\n" + "-"*80)
        print("COMPONENT BREAKDOWN")
        print("-"*80)

        for component, result in self.results['component_results'].items():
            if 'error' in result:
                print(f"\n{component.upper()}: ERROR")
                print(f"  {result['error']}")
            else:
                comp_total = result.get('total_tests', 0)
                comp_passed = result.get('passed', 0)
                comp_rate = (comp_passed / comp_total * 100) if comp_total > 0 else 0
                print(f"\n{component.upper()}: {comp_passed}/{comp_total} ({comp_rate:.1f}%)")

        # Save detailed report
        report_path = f"{self.output_dir}/test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Detailed report saved: {report_path}")

        # Create markdown report
        self.create_markdown_report()

        print("\n" + "="*80)
        if pass_rate >= 80:
            print("✓ TESTS PASSED - System ready for use")
        elif pass_rate >= 60:
            print("⚠ TESTS PARTIALLY PASSED - Some issues need attention")
        else:
            print("✗ TESTS FAILED - Significant issues detected")
        print("="*80)

    def create_markdown_report(self):
        """Create markdown test report"""
        report_md = f"""# LLM Components Test Report

**Date:** {self.results['timestamp']}

## Summary

- **Total Tests:** {self.results['tests_run']}
- **Passed:** {self.results['tests_passed']}
- **Failed:** {self.results['tests_failed']}
- **Pass Rate:** {(self.results['tests_passed'] / self.results['tests_run'] * 100):.1f}%

## Component Results

"""

        for component, result in self.results['component_results'].items():
            report_md += f"### {component.replace('_', ' ').title()}\n\n"

            if 'error' in result:
                report_md += f"**Status:** ERROR\n\n"
                report_md += f"```\n{result['error']}\n```\n\n"
            else:
                total = result.get('total_tests', 0)
                passed = result.get('passed', 0)
                rate = (passed / total * 100) if total > 0 else 0

                report_md += f"- **Tests:** {passed}/{total}\n"
                report_md += f"- **Pass Rate:** {rate:.1f}%\n\n"

                if 'tests' in result:
                    report_md += "**Test Details:**\n\n"
                    for test in result['tests']:
                        status = "✓" if test.get('passed') else "✗"
                        report_md += f"- {status} {test.get('test', 'N/A')}\n"
                    report_md += "\n"

        report_md += f"""
## Files Generated

All test outputs saved to: `{self.output_dir}/`

## Conclusion

"""
        pass_rate = (self.results['tests_passed'] / self.results['tests_run'] * 100) if self.results['tests_run'] > 0 else 0

        if pass_rate >= 80:
            report_md += "✓ **All systems operational.** LLM components are ready for production use.\n"
        elif pass_rate >= 60:
            report_md += "⚠ **Partial success.** Some components need attention before production use.\n"
        else:
            report_md += "✗ **Tests failed.** Significant issues detected. Review required.\n"

        report_path = f"{self.output_dir}/TEST_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_md)

        print(f"✓ Markdown report saved: {report_path}")


def main():
    """Run all tests"""
    tester = LLMComponentTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
