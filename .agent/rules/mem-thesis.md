---
trigger: always_on
---

# GitHub Copilot Instructions

## Project Context
This is an academic thesis project focused on Interpretable machine learning for ensemble of trees. The project includes:

- **LaTeX document** (`book/`): A thesis written in LaTeX with chapters covering introduction, methodology, applications, and conclusions
- **Python analysis** (`src/`, `playground/`): scripts to generate the results, figures, and tables included in the thesis

## Writing Style Guidelines

### For LaTeX/Academic Writing:
- **Tone**: Academic but accessible; avoid jargon when simpler language works
- **Clarity**: Prioritize clear exposition over technical sophistication
- **Transitions**: Ensure smooth logical flow between paragraphs and ideas
- **Target audience**: Applied economists and practitioners who may not be ML experts
- **Citations**: Reference sources appropriately using `\cite{}`
- **Length**: Be thorough but not verbose; provide necessary detail without redundancy
- **Language**: Spanish and all english words shoud be in italics

### For Code:
- **Python style**: Follow PEP 8 conventions
- **Comments**: Explain the "why" not just the "what"
- **Documentation**: Use docstrings for functions and classes
- **Clarity**: Prefer readable code over clever one-liners

### Chapters structure:
- All chapters should start with a brief introduction outlining the chapter's goals and structure.
- Each section and subsection should have a clear purpose and flow logically from the previous content.
- Each chapter, section and subsection should start with a brief overview of what will be covered.
- Each chapter, section and subsection should end with a brief summary of key takeaways.
- Each chapter, section and subsection should be self-contained, providing necessary context for understanding without relying on other parts of the thesis.
- the transitions between sections and subsections should be smooth and not repetitive, guiding the reader through the narrative.

## Common Tasks

### When Editing LaTeX:
1. Check for smooth transitions between paragraphs
2. Ensure citations are properly formatted
3. Maintain consistent terminology
4. Verify mathematical notation is clear
5. Keep sentences at reasonable length (avoid run-ons)
6. Check math accuracy and notation consistency

### When Writing/Reviewing Code:
1. Ensure reproducibility
2. Check for proper error handling
3. Verify data paths are correct
4. Include necessary imports
5. Add comments for complex logic
6. Ensure plots do not have grid lines unless specified
7. Ensure plots are exported as PDF files

### When Reviewing Content:
- Flag unclear passages
- Suggest improvements to flow and logic
- Point out redundancy
- Identify missing transitions
- Check consistency across sections
- Verify all the math is correct and clear. If unsure, ask for clarification.

## What NOT to Do
- Don't create summary markdown files unless explicitly asked
- Don't be overly verbose or repetitive in explanations
- Don't use relative line numbers (user prefers absolute)
- Don't change meaning when improving transitions—preserve the original intent
- Don't suggest changes to citation keys or reference management
- Don't alter file content if not explicitly requested
- Don't use bullet points or numbered lists in LaTeX.
- Don't use hyphens.