# 3D Model Evaluation Leaderboard — container image
# ---------------------------------------------------
# Builds a self-contained BYOM deployment: the objective pipeline + blind
# voting + fused leaderboard, served headless on 0.0.0.0:$PORT.
#
#   docker build -t 3d-eval .
#   docker run -p 8000:8000 3d-eval        # then open http://localhost:8000
#
# The mesh cache is baked at build time so the container starts instantly and
# the blind viewer works immediately. That makes the image large (it contains
# TEST CASES/ + the extracted cache); for a slimmer image, remove the "RUN
# python3 main.py --no-viewer" line and mount TEST CASES/ at runtime instead.

FROM python:3.11-slim

# pymeshlab (Layer 5 topology metrics) needs these system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxrender1 libxkbcommon0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the mesh cache (validate → pairs → extract 35 meshes → cache) so the
# first request is fast and the viewer has its GLBs. Safe to remove for a
# smaller image (the Vote step will build it on demand instead).
RUN python3 main.py --no-viewer

ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Serve only (the cache is already built above). --no-open: no browser in a container.
CMD ["python3", "main.py", "--viewer-only", "--no-open"]
