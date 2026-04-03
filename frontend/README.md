# Frontend

React + Vite viewer for the quadruped websocket streams.

## What It Does

- connects to `ws://localhost:8000/ws`
- renders arena geometry and robot geometry from backend metadata
- renders live swarm frames for continuous training
- renders single-robot frames for checkpoint replay
- shows generation metrics and reward history

## Run

```bash
npm install
npm run dev -- --port 5173
```

The frontend expects the backend to already be running. In normal use, start both together from the repo root:

```bash
python3 main.py
python3 run_single.py
```

## Notes

- The frontend no longer hardcodes the active terrain and robot dimensions as the source of truth. It uses websocket metadata emitted by the backend.
- Production builds can be verified with `npm run build`.
