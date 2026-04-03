import { useEffect, useRef, useState } from "react";
import "./App.css";

const WS_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000/ws`;

const DEFAULT_METADATA = {
  mode: "live",
  config_name: "default",
  terrain: {
    kind: "stepped_arena",
    center_half_m: 2.5,
    step_count: 5,
    step_width_m: 2.0,
    step_height_m: 0.15,
    floor_height_m: 0.0,
  },
  robot: {
    body_length_m: 0.28,
    body_width_m: 0.12,
    leg_length_m: 0.16,
  },
  training: {
    population_size: 32,
    episode_s: 30.0,
    selection_interval_s: 15.0,
  },
};

const STEP_COLORS = ["#203423", "#2f6540", "#699249", "#d6a74b", "#db7440", "#d95066"];
const BOT_COLORS = ["#7cf2a1", "#8af65d", "#f3df69", "#ff9d51", "#ff6a8a", "#ff7fd0"];

function rotMat(rot) {
  const [r, p, y] = rot;
  const cr = Math.cos(r);
  const sr = Math.sin(r);
  const cp = Math.cos(p);
  const sp = Math.sin(p);
  const cy = Math.cos(y);
  const sy = Math.sin(y);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

function applyRot(rotation, vector) {
  return [
    rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
    rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
    rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
  ];
}

function project(x, y, z, cam, width, height) {
  const { az, el, scale } = cam;
  const cosAz = Math.cos(az);
  const sinAz = Math.sin(az);
  const cosEl = Math.cos(el);
  const sinEl = Math.sin(el);
  const rx = x * cosAz - y * sinAz;
  const ry = x * sinAz + y * cosAz;
  const sx = rx * scale;
  const sy = (ry * cosEl - z * sinEl) * scale;
  return [width / 2 + sx, height / 2 + sy];
}

function drawArena(ctx, cam, width, height, terrain) {
  ctx.save();
  if (terrain.kind === "flat") {
    const size = terrain.field_half_m || 6.0;
    const corners = [[-size, -size], [size, -size], [size, size], [-size, size]];
    const pts = corners.map(([cx, cy]) => project(cx, cy, terrain.floor_height_m || 0.0, cam, width, height));
    ctx.fillStyle = "#0c1a0f";
    ctx.strokeStyle = "#1f4b2d";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    return;
  }

  for (let stepIndex = terrain.step_count; stepIndex >= 0; stepIndex -= 1) {
    const radius = terrain.center_half_m + stepIndex * terrain.step_width_m;
    const z = (terrain.floor_height_m || 0.0) + stepIndex * terrain.step_height_m;
    const corners = [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]];
    const pts = corners.map(([cx, cy]) => project(cx, cy, z, cam, width, height));

    ctx.fillStyle = stepIndex === 0 ? "#102017" : `${STEP_COLORS[Math.min(stepIndex, STEP_COLORS.length - 1)]}26`;
    ctx.strokeStyle = stepIndex === 0 ? "#285539" : `${STEP_COLORS[Math.min(stepIndex, STEP_COLORS.length - 1)]}88`;
    ctx.lineWidth = stepIndex === 0 ? 1 : 1.4;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawGoal(ctx, cam, width, height, goal) {
  if (!goal || goal.length !== 3) return;
  const [gx, gy, gz] = goal;
  const [px, py] = project(gx, gy, gz, cam, width, height);
  ctx.save();
  ctx.strokeStyle = "#f0c75d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px - 12, py);
  ctx.lineTo(px + 12, py);
  ctx.moveTo(px, py - 12);
  ctx.lineTo(px, py + 12);
  ctx.stroke();
  ctx.restore();
}

function drawSwarm(ctx, swarm, cam, width, height, robot) {
  if (!swarm || !swarm.n) return;

  const bodyHalfLength = robot.body_length_m / 2;
  const bodyHalfWidth = robot.body_width_m / 2;
  const legLength = robot.leg_length_m;
  const mountsBody = [
    [bodyHalfLength, bodyHalfWidth, 0],
    [bodyHalfLength, -bodyHalfWidth, 0],
    [-bodyHalfLength, bodyHalfWidth, 0],
    [-bodyHalfLength, -bodyHalfWidth, 0],
  ];
  const bodyCornersBody = [
    [-bodyHalfLength, -bodyHalfWidth, 0],
    [bodyHalfLength, -bodyHalfWidth, 0],
    [bodyHalfLength, bodyHalfWidth, 0],
    [-bodyHalfLength, bodyHalfWidth, 0],
  ];

  const bodyPaths = Array.from({ length: BOT_COLORS.length }, () => []);
  const legPaths = Array.from({ length: BOT_COLORS.length }, () => []);

  for (let i = 0; i < swarm.n; i += 1) {
    const bodyIndex = i * 3;
    const legIndex = i * 4;
    const bx = swarm.pos[bodyIndex];
    const by = swarm.pos[bodyIndex + 1];
    const bz = swarm.pos[bodyIndex + 2];
    const rx = swarm.rot[bodyIndex];
    const ry = swarm.rot[bodyIndex + 1];
    const rz = swarm.rot[bodyIndex + 2];
    const level = Math.min(Math.round(swarm.level[i] || 0), BOT_COLORS.length - 1);
    const rotation = rotMat([rx, ry, rz]);

    const worldCorners = bodyCornersBody.map((corner) => {
      const world = applyRot(rotation, corner);
      return project(bx + world[0], by + world[1], bz + world[2], cam, width, height);
    });
    bodyPaths[level].push(worldCorners);

    const robotLegs = [];
    for (let legOffset = 0; legOffset < 4; legOffset += 1) {
      const mountBody = mountsBody[legOffset];
      const mountWorld = applyRot(rotation, mountBody);
      const angle = swarm.leg[legIndex + legOffset];
      const footBody = [legLength * Math.sin(angle), 0, -legLength * Math.cos(angle)];
      const footWorld = applyRot(rotation, footBody);
      robotLegs.push([
        project(bx + mountWorld[0], by + mountWorld[1], bz + mountWorld[2], cam, width, height),
        project(
          bx + mountWorld[0] + footWorld[0],
          by + mountWorld[1] + footWorld[1],
          bz + mountWorld[2] + footWorld[2],
          cam,
          width,
          height,
        ),
      ]);
    }
    legPaths[level].push(robotLegs);
  }

  for (let level = 0; level < BOT_COLORS.length; level += 1) {
    if (!legPaths[level].length) continue;
    ctx.strokeStyle = `${BOT_COLORS[level]}88`;
    ctx.lineWidth = 0.85;
    ctx.beginPath();
    for (const segments of legPaths[level]) {
      for (const [[x1, y1], [x2, y2]] of segments) {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
    ctx.stroke();
  }

  for (let level = 0; level < BOT_COLORS.length; level += 1) {
    if (!bodyPaths[level].length) continue;
    ctx.strokeStyle = BOT_COLORS[level];
    ctx.lineWidth = level > 0 ? 1.2 : 0.9;
    ctx.beginPath();
    for (const corners of bodyPaths[level]) {
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < corners.length; i += 1) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath();
    }
    ctx.stroke();
  }
}

function RewardChart({ history }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const data = history.slice(-100);
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    let min = data[0];
    let max = data[0];
    for (let i = 1; i < data.length; i += 1) {
      min = Math.min(min, data[i]);
      max = Math.max(max, data[i]);
    }
    const range = max - min || 1;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#112219";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#6df1a6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((value, index) => {
      const px = (index / (data.length - 1)) * width;
      const py = height - ((value - min) / range) * (height - 8) - 4;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }, [history]);

  return <canvas ref={ref} width={320} height={96} className="reward-chart" />;
}

function StepLegend({ stepCount }) {
  return (
    <div className="legend-card">
      {BOT_COLORS.map((color, index) => (
        <span key={color} className="legend-chip">
          <span className="legend-dot" style={{ color }} />
          {index === 0 ? "center" : index === stepCount ? "escape" : `step ${index}`}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const canvasRef = useRef(null);
  const swarmRef = useRef(null);
  const camRef = useRef(null);
  const animRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [generation, setGeneration] = useState(null);

  function getCam(width, height) {
    if (!camRef.current) {
      camRef.current = {
        az: Math.PI / 4,
        el: Math.PI / 5,
        scale: Math.min(width, height) / 24,
        dragging: false,
        lastX: 0,
        lastY: 0,
      };
    }
    return camRef.current;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    function onMouseDown(event) {
      const cam = camRef.current;
      if (!cam) return;
      cam.dragging = true;
      cam.lastX = event.clientX;
      cam.lastY = event.clientY;
    }

    function onMouseMove(event) {
      const cam = camRef.current;
      if (!cam || !cam.dragging) return;
      cam.az -= (event.clientX - cam.lastX) * 0.005;
      cam.el = Math.max(0.08, Math.min(Math.PI * 0.48, cam.el + (event.clientY - cam.lastY) * 0.005));
      cam.lastX = event.clientX;
      cam.lastY = event.clientY;
    }

    function onMouseUp() {
      if (camRef.current) camRef.current.dragging = false;
    }

    function onWheel(event) {
      const cam = camRef.current;
      if (!cam) return;
      event.preventDefault();
      cam.scale *= event.deltaY > 0 ? 0.93 : 1.07;
      cam.scale = Math.max(2, Math.min(900, cam.scale));
    }

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, []);

  useEffect(() => {
    let socket;
    let reconnectTimer;

    function connect() {
      socket = new WebSocket(WS_URL);
      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        reconnectTimer = window.setTimeout(connect, 2000);
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "metadata") setMetadata((current) => ({ ...current, ...message }));
        if (message.type === "swarm") swarmRef.current = message;
        if (message.type === "generation") setGeneration(message);
      };
    }

    connect();
    return () => {
      window.clearTimeout(reconnectTimer);
      if (socket) socket.close();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    function render() {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) {
        animRef.current = requestAnimationFrame(render);
        return;
      }

      const width = window.innerWidth;
      const height = window.innerHeight;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      const ctx = canvas.getContext("2d");
      const cam = getCam(width, height);
      const terrain = metadata.terrain || DEFAULT_METADATA.terrain;
      const robot = metadata.robot || DEFAULT_METADATA.robot;
      const swarm = swarmRef.current;
      const goal = swarm?.goal || generation?.goal || metadata?.goal?.fixed_goal_xyz;

      ctx.fillStyle = "#08120d";
      ctx.fillRect(0, 0, width, height);

      drawArena(ctx, cam, width, height, terrain);
      drawGoal(ctx, cam, width, height, goal);
      drawSwarm(ctx, swarm, cam, width, height, robot);

      if (!swarm) {
        ctx.fillStyle = "#587760";
        ctx.font = "18px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(connected ? "Waiting for simulation frames..." : `Connecting to ${WS_URL}`, width / 2, height / 2);
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(render);
    }

    animRef.current = requestAnimationFrame(render);
    return () => {
      cancelled = true;
      cancelAnimationFrame(animRef.current);
    };
  }, [connected, generation, metadata]);

  const episodeSeconds = metadata.training?.episode_s ?? DEFAULT_METADATA.training.episode_s;
  const selectionInterval = metadata.training?.selection_interval_s ?? DEFAULT_METADATA.training.selection_interval_s;

  return (
    <div className="app-shell">
      <canvas ref={canvasRef} className="arena-canvas" />

      <aside className="hud-card hud-card--primary">
        <div className="hud-kicker">
          {metadata.mode === "single" ? "Checkpoint Replay" : "Live Training"}
          <span className={`status-pill ${connected ? "status-pill--live" : "status-pill--offline"}`}>
            {connected ? "live" : "offline"}
          </span>
        </div>
        <h1 className="hud-title">Quadruped Arena</h1>
        <p className="hud-subtitle">
          config <strong>{metadata.config_name}</strong>
        </p>

        {generation ? (
          <div className="metrics-grid">
            <div><span>generation</span><strong>{generation.generation}</strong></div>
            <div><span>mean reward</span><strong>{generation.mean_reward?.toFixed(2) ?? "0.00"}</strong></div>
            <div><span>best reward</span><strong>{generation.best_reward?.toFixed(2) ?? "0.00"}</strong></div>
            <div><span>top reward</span><strong>{generation.top_rewards?.[0]?.toFixed(2) ?? "n/a"}</strong></div>
          </div>
        ) : (
          <div className="hud-empty">No metrics yet.</div>
        )}

        <div className="chart-block">
          <div className="chart-label">mean reward / generation</div>
          <RewardChart history={generation?.rewards_history || []} />
        </div>
      </aside>

      <aside className="hud-card hud-card--secondary">
        <div className="spec-grid">
          <div><span>terrain</span><strong>{metadata.terrain.kind}</strong></div>
          <div><span>steps</span><strong>{metadata.terrain.step_count}</strong></div>
          <div><span>step width</span><strong>{metadata.terrain.step_width_m}m</strong></div>
          <div><span>step height</span><strong>{metadata.terrain.step_height_m}m</strong></div>
          <div><span>population</span><strong>{metadata.training.population_size}</strong></div>
          <div><span>episode</span><strong>{episodeSeconds}s</strong></div>
          <div><span>selection</span><strong>{selectionInterval}s</strong></div>
          <div><span>viewer</span><strong>{metadata.mode}</strong></div>
        </div>
        <p className="hud-note">Drag to orbit. Scroll to zoom. Goal marker updates from the active server stream.</p>
      </aside>

      <StepLegend stepCount={metadata.terrain.step_count || DEFAULT_METADATA.terrain.step_count} />
    </div>
  );
}
