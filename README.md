# AutoNav 3D

A Streamlit app that plans a car's route around pedestrians, cows, carts, cars and potholes with a time-aware A*, then drives it.

- **3D Cockpit Simulator** tab: three.js page (`vehicle_path_3d.html`) with cockpit, chase, drone, orbit and top-down cameras. Click the road to drop obstacles, even while the car is driving.
- **2D Tactical LiDAR Radar** tab: the same planner in Python, drawn with Plotly.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run vehicle_path_standalone.py
```

Then open http://localhost:8501.

Keys in the 3D view: `Space` start/pause, `R` reset, `1`-`5` cameras, `W/A/S/D` in manual mode.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Editing the 3D page

`vehicle_path_standalone.py` loads `vehicle_path_3d.html` from disk when it's there and falls back to a copy embedded in the script otherwise. After changing the HTML, refresh that copy so single-file deploys get the same page:

```bash
python embed_fix.py vehicle_path_standalone.py
```
