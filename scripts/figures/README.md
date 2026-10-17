# Reproducible blog figures

Vector schematics for the VAE/DDPM and Flash-GRPO articles.

## Regenerate

Requires Python 3, NumPy and Matplotlib (generated with NumPy 2.3.5 and Matplotlib 3.10.8).
From the repository root:

```sh
python scripts/figures/generate_vae.py --output-dir images/vae-ddpm
python scripts/figures/generate_flash.py --output-dir images/flash-grpo
```

Each script exports SVG, PDF and PNG previews. The website uses only SVG.
Text is exported as vector glyph paths for consistent rendering without remote fonts.
No raster images are embedded in the SVGs. Fixed random seeds, SVG IDs and omitted
creation timestamps make the SVG outputs reproducible within the same dependency versions.

These are explanatory schematics, not empirical results or measured performance plots.
The diffusion cells illustrate progressive corruption; vector lengths in the gradient
figure illustrate that temporal rectification does not equalize sample gradients.
The Flash-GRPO rollout diagram distinguishes full sampling from selected policy-loss
terms; ITG shares diffusion time within a prompt group, not the initial noise or states.
