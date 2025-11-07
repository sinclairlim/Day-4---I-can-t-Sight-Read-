# Sight Reading Helper

A computer vision tool that helps users learn to sight-read sheet music by identifying and labeling notes.

## Features

- Template-based note recognition using heatmap matching
- Staff line detection
- Note pitch identification
- Automatic note labeling on sheet music

## Setup

1. Install dependencies (for CLI + visualization use the dev set):
```bash
pip install -r requirements-dev.txt
```

2. Prepare your template images in the `templates/` directory

3. Run the tool on a sheet music image:
```bash
python sight_reader.py --input your_sheet_music.jpg
```

## Web UI (Flask)

Prefer a simple upload page? Launch the bundled Flask app, which reuses `SightReaderV2` behind the scenes:

```bash
pip install -r requirements.txt  # lightweight stack (Flask + OpenCV headless)
export FLASK_APP=app.py  # or set in .env on Windows
flask run
```

Open the printed URL, upload a treble+bass score, and the page will return an annotated image plus a table of detected notes. The UI is pure HTML/CSS—no ML services involved.

### Deploying on Vercel

1. Install the Vercel CLI (`npm i -g vercel`) and log in.
2. Ensure `vercel.json` is present (it routes all traffic to `app.py` via `@vercel/python`).
3. From the repo root run `vercel` (preview) or `vercel --prod` (production). The builder packages Flask + OpenCV into a serverless function.

## Current Assumptions / Limitations

- Input should contain paired treble and bass staves repeating in that order (treble, bass, treble, bass, ...); the pitch logic currently decides clef purely by staff index.
- Works best on clean scans with clearly separated staff lines; heavy skew, handwritten scores, or dense ledger lines may still confuse the staff detector.
- Only standard five-line staves are supported; percussion or single-clef pages will mislabel notes until clef detection is implemented.

## How It Works

1. Loads template images of different note types and a generic `find.png` notehead template.
2. Uses classical OpenCV template matching (no machine learning models) to locate circular noteheads at multiple scales.
3. Detects staff lines by scanning each horizontal row for “ink density” peaks, smoothing the signal, and grouping the five strongest peaks per staff.
4. Measures each notehead’s vertical offset from the nearest staff’s bottom line, converts the offset (in half-line units) into diatonic steps, and maps those to pitch names using the known clef ordering (treble/bass alternating).
5. Overlays the inferred pitch text directly above each detected notehead.
