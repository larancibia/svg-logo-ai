# Contributing to LLM-QD Logo System

Thank you for your interest in contributing to the LLM-QD Logo System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Expected Behavior

- Be respectful of differing viewpoints and experiences
- Give and gracefully accept constructive feedback
- Focus on what is best for the community and research advancement
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or insulting/derogatory comments
- Public or private harassment
- Publishing others' private information without permission

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/svg-logo-ai.git
   cd svg-logo-ai
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/larancibia/svg-logo-ai.git
   ```
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Google API key (Gemini)
- Git
- Virtual environment tool (venv)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (recommended)
pip install pytest pytest-cov black flake8 mypy

# Set up API key
cp .env.example .env
# Edit .env with your credentials
```

### Verify Installation

```bash
# Run tests (when available)
pytest tests/

# Run a quick demo
python src/demo_llm_qd.py
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality to the system
3. **Documentation**: Improve or expand documentation
4. **Tests**: Add test coverage
5. **Examples**: Create examples demonstrating system usage
6. **Research**: Contribute new algorithms or metrics
7. **Performance**: Optimize existing code

### Finding Work

- Check the [Issues](https://github.com/larancibia/svg-logo-ai/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Review the [Project Roadmap](docs/RESEARCH_FINDINGS.md)
- Propose new features in Discussions

## Code Style

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Maximum line length: 100 characters
# Use 4 spaces for indentation (no tabs)
# Use double quotes for strings by default

# Good example:
def generate_logo(company_name: str, style: str = "geometric") -> dict:
    """
    Generate a logo using LLM-guided Quality-Diversity.

    Args:
        company_name: Name of the company for the logo
        style: Visual style preference (geometric, organic, etc.)

    Returns:
        Dictionary containing SVG code and metadata
    """
    result = {
        "svg_code": "",
        "fitness": 0.0,
        "behavior": {}
    }
    return result
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all public functions/classes
- **Type Hints**: Always include type hints for function parameters and returns
- **Comments**: Use comments to explain "why", not "what"
- **README Updates**: Update relevant documentation when adding features

### Naming Conventions

```python
# Variables and functions: snake_case
fitness_score = calculate_fitness(logo)

# Classes: PascalCase
class LogoGenerator:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 100
DEFAULT_ARCHIVE_SIZE = (10, 10, 10, 10)

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format all Python files
black src/

# Check formatting without changes
black --check src/
```

### Linting

Use **flake8** for linting:

```bash
# Run linter
flake8 src/ --max-line-length=100

# Common issues to avoid:
# - Unused imports
# - Undefined variables
# - Line too long (>100 chars)
```

### Type Checking

Use **mypy** for static type checking:

```bash
# Run type checker
mypy src/ --ignore-missing-imports
```

## Testing Requirements

### Writing Tests

Tests should be placed in the `tests/` directory (create if needed):

```python
# tests/test_behavior_characterization.py
import pytest
from src.behavior_characterization import BehaviorCharacterization

def test_complexity_calculation():
    bc = BehaviorCharacterization()
    svg_code = '<svg><circle cx="50" cy="50" r="40"/></svg>'
    complexity = bc.calculate_complexity(svg_code)
    assert 0 <= complexity <= 1
    assert isinstance(complexity, float)

def test_style_detection():
    bc = BehaviorCharacterization()
    geometric_svg = '<svg><rect x="0" y="0" width="100" height="100"/></svg>'
    style = bc.calculate_style(geometric_svg)
    assert style < 0.5  # Should be geometric (closer to 0)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_behavior_characterization.py

# Run specific test
pytest tests/test_behavior_characterization.py::test_complexity_calculation
```

### Test Requirements

- All new features must include tests
- Bug fixes should include a test that would have caught the bug
- Aim for >80% code coverage
- Tests must pass before merging

### Integration Tests

For experiments that require API calls:

```python
import pytest
from src.llm_qd_logo_system import LLMQDLogoSystem

@pytest.mark.integration
@pytest.mark.slow
def test_full_experiment():
    """Test complete QD experiment (requires API key)."""
    system = LLMQDLogoSystem(
        company_name="TestCorp",
        num_iterations=5  # Small for testing
    )
    results = system.run_experiment()
    assert len(results['archive']) > 0
    assert results['coverage'] > 0

# Run integration tests:
# pytest tests/ -m integration
```

## Pull Request Process

### Before Submitting

1. **Update from upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   black src/
   flake8 src/ --max-line-length=100
   mypy src/ --ignore-missing-imports
   pytest tests/
   ```

3. **Update documentation:**
   - Update relevant `.md` files
   - Add docstrings to new functions
   - Update `CHANGELOG.md`

4. **Test your changes:**
   - Run experiments to verify functionality
   - Check that existing features still work
   - Verify no performance regressions

### Submitting the Pull Request

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues
   - Screenshots/examples if applicable

3. **PR Description Template:**
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation and Context
   Why is this change needed? What problem does it solve?

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Updated documentation

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex code
   - [ ] Documentation updated
   - [ ] No new warnings generated
   ```

### Review Process

1. Maintainers will review your PR within 5 business days
2. Address feedback by pushing new commits
3. Once approved, maintainers will merge your PR
4. Your contribution will be acknowledged in release notes

### Commit Message Guidelines

Use conventional commits format:

```
type(scope): brief description

Detailed explanation of changes (if needed).

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(qd): add adaptive mutation rate based on coverage

Implements dynamic mutation rate that increases when coverage
is low and decreases when archive is filling up rapidly.

Closes #42

fix(evaluator): handle empty SVG files gracefully

Previously crashed when SVG had no elements. Now returns
fitness of 0 with appropriate warning.

Fixes #67

docs(readme): update installation instructions for Python 3.11

Python 3.11 requires additional step for chromadb installation.
```

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** to ensure it's not expected behavior
3. **Update to latest version** and test if issue persists
4. **Prepare minimal reproduction** if reporting a bug

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command X
2. With parameters Y
3. See error Z

**Expected Behavior**
What should have happened

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- Package versions: [from pip freeze]

**Error Messages**
```
Paste full error traceback here
```

**Additional Context**
Any other relevant information
```

### Feature Request Template

```markdown
**Feature Description**
Clear description of proposed feature

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Research Context**
Any papers/references supporting this feature
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

### Getting Help

- **Quick Questions**: Open a Discussion
- **Bug Reports**: Open an Issue
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

### Acknowledgments

Contributors will be acknowledged in:
- `README.md` contributors section
- Release notes for their contributions
- Academic papers if contributing significant research

## Special Notes for Research Contributions

### Publishing Work Based on This System

If you use this system in your research:

1. **Citation**: Please cite using `CITATION.cff`
2. **Acknowledgment**: Mention the system in your paper
3. **Derivatives**: Let us know about your publications
4. **Datasets**: Consider contributing results back

### Contributing Algorithms

For new algorithms (QD variants, mutation operators, etc.):

1. **Provide paper reference** or theoretical justification
2. **Include benchmarks** comparing to existing methods
3. **Document parameters** and when to use the algorithm
4. **Add to documentation** in `docs/ADVANCED_OPTIMIZATION.md`

### Contributing Metrics

For new quality or behavioral metrics:

1. **Explain metric rationale** and what it measures
2. **Provide validation** (human studies if possible)
3. **Benchmark correlation** with existing metrics
4. **Document range and interpretation**

## Recognition

We value all contributions, whether they're:
- A typo fix in documentation
- A major algorithmic improvement
- Thoughtful code review
- Helping others in discussions

Thank you for contributing to advancing AI-driven design!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
