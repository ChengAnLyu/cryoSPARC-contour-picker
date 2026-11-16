# Introduction

This script is an interactive helper for contour-based particle picking alongside the cryoSPARC manual-picking workflow. It is intended to help populate the displayed area enclosed by a capsid contour with a dense, regularly spaced set of picks.

The user holds the left mouse button to trace a contour on screen. On release, the script samples a square grid inside the enclosed polygon and generates mouse clicks at the retained positions. The `click_interval` parameter controls spacing in screen pixels: smaller values give denser sampling, while larger values give fewer picks.

The supplied implementation works through the live picking interface and uses temporary 2D screen coordinates. cryoSPARC handles storing and exporting the resulting picks; direct processing of cryoSPARC output files and 3D surface reconstruction are outside this implementation.

Windows is preferred. Install a Python interpreter and the dependencies in `requirements.txt`, then run `particle_picker_ChengAn.py`; PyCharm is an optional environment for doing so. See [README.md](README.md) for installation, usage, density adjustment, and stopping instructions.
