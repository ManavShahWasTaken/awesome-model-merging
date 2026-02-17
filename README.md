# Model Merging — Manim Slides Presentation

A 3Blue1Brown-style interactive presentation on model merging, built with [Manim](https://www.manim.community/) and [Manim Slides](https://manim-slides.eertmans.be/).

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Manim requires a LaTeX distribution (e.g. `mactex` on macOS) for math rendering.
> Install with: `brew install --cask mactex-no-gui`

## Render & Present

```bash
# Render all slides
manim-slides render slides.py TitleSlide ConceptSlide LossBarrierSlide

# Present interactively (click/arrow keys to advance)
manim-slides TitleSlide ConceptSlide LossBarrierSlide

# Export to standalone HTML
manim-slides convert TitleSlide ConceptSlide LossBarrierSlide presentation.html --open
```

## Controls (during presentation)

- **Right arrow / Click** — next slide
- **Left arrow** — previous slide
- **Q** — quit
