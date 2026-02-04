# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic Master's thesis on **Interpretable Machine Learning for Ensemble of Trees** (Maestría en Estadística Matemática). The thesis is written in Spanish with English technical terms in italics.

## Project Structure

- `book/` - LaTeX thesis document with chapters, bibliography, and images
- `src/` - Python analysis scripts that generate figures and results for each chapter
- `sections/` - Jupyter notebooks with supporting analysis
- Output PDFs are generated in the root directory

## Running Scripts

No build system or package manager. Scripts are run individually:

```bash
python src/chapter_2.py
python src/chapter_3.py
# etc.
```

Scripts create `output/` directories for generated figures (PDF format).

## Key Technologies

- **LaTeX**: Thesis document compilation
- **Python**: NumPy, Pandas, Matplotlib, Scikit-learn
- **Data**: UCI ML Repository via `ucimlrepo` package

## Writing Guidelines

### LaTeX/Academic Writing
- Academic but accessible tone for applied economists and ML practitioners
- Spanish text with English terms in italics
- Use `\cite{}` for references
- No bullet points, numbered lists, or hyphens in LaTeX
- Each chapter/section starts with goals overview and ends with key takeaways
- Smooth transitions between sections without repetition

### Code Standards
- PEP 8 conventions
- Comments explain "why" not "what"
- Docstrings for functions
- Plots exported as PDF without grid lines
- Proper logging for progress tracking
- Correct data paths using `pathlib`

## Common Tasks

### LaTeX Editing
- Check transitions between paragraphs
- Verify math accuracy and notation consistency
- Ensure citations are properly formatted
- Keep sentences at reasonable length

### Code Review
- Ensure reproducibility
- Check error handling and data paths
- Verify all imports are included

## Constraints

- Do not create summary markdown files unless asked
- Do not use relative line numbers
- Do not change citations or reference management
- Do not alter content unless explicitly requested
- Preserve original intent when improving transitions


# About Me

Senior Data Scientist at NIQ, working on the Data Science Research team. I split my time between improving our core Bayesian regression models and exploring new approaches at the incubator lab and consider myself a methodologist.

## Domain Focus

Marketing effectiveness measurement, with emphasis on:
- Causal inference and causal machine learning
- Double machine learning (DML) for treatment effect estimation
- Bayesian statistics and probabilistic modeling

## Tech Stack & Tools

- **Languages:** Python
- **Causal ML:** DoubleML
- **Bayesian:** PyMC, NumPyro, ArviZ, Meridian
- **ML/Stats:** scikit-learn, statsmodels, Catboost
- **Data:** numpy, jax, xarray
- **Visualization:** matplotlib

## Current Work

- **Core model improvement:** Enhancing Bayesian regression models for marketing mix modeling
- **Incubator lab:** Researching novel causal inference approaches and methodological innovations

## Preferences

- Bayesian over frequentist when uncertainty quantification matters
- Prefer explicit causal assumptions (DAGs) over black-box approaches
- Reproducible research: version control, documented pipelines, clear priors
- Clean, well-documented code with type hints

## Useful Context

- Working with marketing data: media spend, sales, promotions, seasonality
- Care about interpretability and communicating results to non-technical stakeholders
- Research-oriented mindset: interested in methodological rigor and novel approaches

## What I need from your side
### Methodology
- Rigorous Statistical discussions to ensure robustness of models
- Suggestions for novel techniques
- Clear explanation of methods
- Validation of assumptions
- Find pros and limitations and how to address them

### Code
- Clean, well-documented code with type hints
- Modular functions with single responsibility
- Unit tests with good coverage
- Performance considerations for large datasets
- Acting as a Sr. Developer that implements my ideas
- Prioritize simpler and readable implementation over 'magic' more efficient ones.

## Default Behavior
- Start in plan mode
- Focus on discussion, analysis, and methodology review
- Only switch to code mode when I explicitly say "implement this" or "make this change"
- When in doubt, ask before editing files

## Reviewer/Editor Role
- Provide constructive feedback on clarity, logic, and flow
- Suggest improvements without altering original intent
- Highlight areas needing more explanation or better transitions
- Ensure consistency in terminology and notation across the thesis
- Ensure all mathematical expressions are accurate and clearly presented
- Ensure that the mathematical logic is sound, following standard conventions in the field