import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_PORT = import.meta.env.VITE_API_PORT || "8000";
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:${API_PORT}/ws`;

const DEFAULT_METADATA = {
  mode: "viewer",
  config_name: "default",
  terrain: {
    kind: "stepped_arena",
    field_half_m: 6.0,
    center_half_m: 2.5,
    step_count: 5,
    step_width_m: 2.0,
    step_height_m: 0.15,
    floor_height_m: 0.0,
  },
  robot: {
    body_length_m: 0.28,
    body_width_m: 0.12,
    body_height_m: 0.08,
    leg_length_m: 0.16,
    leg_radius_m: 0.02,
    foot_radius_m: 0.03,
  },
  training: {
    population_size: 32,
    episode_s: 30.0,
    selection_interval_s: 15.0,
    viewer_reset_s: 30.0,
  },
  simulator: {
    backend: "jax",
  },
};

const STEP_COLORS = ["#203423", "#2f6540", "#699249", "#d6a74b", "#db7440", "#d95066"];
const MODEL_COLORS = ["#7cf2a1", "#8af65d", "#f3df69", "#ff9d51", "#ff6a8a", "#ff7fd0"];
const BODY_BOX_EDGES = [
  [0, 1],
  [0, 2],
  [0, 4],
  [1, 3],
  [1, 5],
  [2, 3],
  [2, 6],
  [3, 7],
  [4, 5],
  [4, 6],
  [5, 7],
  [6, 7],
];
const BODY_BOX_FACES = [
  [0, 1, 3, 2],
  [4, 5, 7, 6],
  [0, 1, 5, 4],
  [2, 3, 7, 6],
  [0, 2, 6, 4],
  [1, 3, 7, 5],
];

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
    const corners = [
      [-size, -size],
      [size, -size],
      [size, size],
      [-size, size],
    ];
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
    const corners = [
      [-radius, -radius],
      [radius, -radius],
      [radius, radius],
      [-radius, radius],
    ];
    const pts = corners.map(([cx, cy]) => project(cx, cy, z, cam, width, height));
    const color = STEP_COLORS[Math.min(stepIndex, STEP_COLORS.length - 1)];

    ctx.fillStyle = stepIndex === 0 ? "#102017" : `${color}26`;
    ctx.strokeStyle = stepIndex === 0 ? "#285539" : `${color}88`;
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

function drawBodyFromCorners(ctx, corners, cam, width, height, color) {
  if (!Array.isArray(corners) || corners.length < 8) return;
  const projected = corners.map((point) => project(point[0], point[1], point[2], cam, width, height));

  ctx.save();
  ctx.fillStyle = `${color}18`;
  for (const [a, b, c, d] of BODY_BOX_FACES) {
    ctx.beginPath();
    ctx.moveTo(projected[a][0], projected[a][1]);
    ctx.lineTo(projected[b][0], projected[b][1]);
    ctx.lineTo(projected[c][0], projected[c][1]);
    ctx.lineTo(projected[d][0], projected[d][1]);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  for (const [start, end] of BODY_BOX_EDGES) {
    ctx.moveTo(projected[start][0], projected[start][1]);
    ctx.lineTo(projected[end][0], projected[end][1]);
  }
  ctx.stroke();
  ctx.restore();
}

function drawLegsFromPhysics(ctx, legs, cam, width, height, color) {
  if (!Array.isArray(legs) || legs.length === 0) return;

  ctx.save();
  ctx.strokeStyle = `${color}cc`;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (const leg of legs) {
    const mount = leg?.mount;
    const foot = leg?.foot;
    if (!Array.isArray(mount) || mount.length < 3 || !Array.isArray(foot) || foot.length < 3) continue;
    const mountPoint = project(mount[0], mount[1], mount[2], cam, width, height);
    const footPoint = project(foot[0], foot[1], foot[2], cam, width, height);
    ctx.moveTo(mountPoint[0], mountPoint[1]);
    ctx.lineTo(footPoint[0], footPoint[1]);
  }
  ctx.stroke();

  for (const leg of legs) {
    const mount = leg?.mount;
    const foot = leg?.foot;
    if (!Array.isArray(mount) || mount.length < 3 || !Array.isArray(foot) || foot.length < 3) continue;
    const mountPoint = project(mount[0], mount[1], mount[2], cam, width, height);
    const footPoint = project(foot[0], foot[1], foot[2], cam, width, height);

    ctx.beginPath();
    ctx.arc(mountPoint[0], mountPoint[1], 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(footPoint[0], footPoint[1], 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawModel(ctx, frame, cam, width, height, robot) {
  if (!frame) return;

  const level = Math.min(Math.max(Math.round(frame.level?.[0] || 0), 0), MODEL_COLORS.length - 1);
  const color = MODEL_COLORS[level];
  const bodyCorners = frame?.body?.corners;
  const legs = frame?.legs;

  let renderedPhysicsGeometry = false;
  if (Array.isArray(bodyCorners) && bodyCorners.length >= 8) {
    drawBodyFromCorners(ctx, bodyCorners, cam, width, height, color);
    renderedPhysicsGeometry = true;
  }
  if (Array.isArray(legs) && legs.length > 0) {
    drawLegsFromPhysics(ctx, legs, cam, width, height, color);
    renderedPhysicsGeometry = true;
  }
  if (renderedPhysicsGeometry) return;

  if (!frame.pos || frame.pos.length < 3 || !frame.rot || frame.rot.length < 3) return;

  const bodyHalfLength = robot.body_length_m / 2;
  const bodyHalfWidth = robot.body_width_m / 2;
  const bodyHalfHeight = robot.body_height_m / 2;
  const legLength = robot.leg_length_m;
  const bodyCornersBody = [
    [-bodyHalfLength, -bodyHalfWidth, -bodyHalfHeight],
    [-bodyHalfLength, -bodyHalfWidth, bodyHalfHeight],
    [-bodyHalfLength, bodyHalfWidth, -bodyHalfHeight],
    [-bodyHalfLength, bodyHalfWidth, bodyHalfHeight],
    [bodyHalfLength, -bodyHalfWidth, -bodyHalfHeight],
    [bodyHalfLength, -bodyHalfWidth, bodyHalfHeight],
    [bodyHalfLength, bodyHalfWidth, -bodyHalfHeight],
    [bodyHalfLength, bodyHalfWidth, bodyHalfHeight],
  ];
  const mountsBody = [
    [bodyHalfLength, bodyHalfWidth, 0],
    [bodyHalfLength, -bodyHalfWidth, 0],
    [-bodyHalfLength, bodyHalfWidth, 0],
    [-bodyHalfLength, -bodyHalfWidth, 0],
  ];

  const bodyPos = frame.pos.slice(0, 3);
  const bodyRot = frame.rot.slice(0, 3);
  const legAngles = frame.leg || [];
  const rotation = rotMat(bodyRot);

  const worldBodyCorners = bodyCornersBody.map((corner) => {
    const world = applyRot(rotation, corner);
    return [bodyPos[0] + world[0], bodyPos[1] + world[1], bodyPos[2] + world[2]];
  });
  drawBodyFromCorners(ctx, worldBodyCorners, cam, width, height, color);

  ctx.save();
  ctx.strokeStyle = `${color}bb`;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let i = 0; i < 4; i += 1) {
    const mountWorld = applyRot(rotation, mountsBody[i]);
    const angle = legAngles[i] || 0;
    const footBody = [legLength * Math.sin(angle), 0, -legLength * Math.cos(angle)];
    const footWorld = applyRot(rotation, footBody);
    const mountPoint = project(
      bodyPos[0] + mountWorld[0],
      bodyPos[1] + mountWorld[1],
      bodyPos[2] + mountWorld[2],
      cam,
      width,
      height,
    );
    const footPoint = project(
      bodyPos[0] + mountWorld[0] + footWorld[0],
      bodyPos[1] + mountWorld[1] + footWorld[1],
      bodyPos[2] + mountWorld[2] + footWorld[2],
      cam,
      width,
      height,
    );
    ctx.moveTo(mountPoint[0], mountPoint[1]);
    ctx.lineTo(footPoint[0], footPoint[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function basename(value) {
  if (!value) return "uninitialized";
  const parts = String(value).split("/");
  return parts[parts.length - 1] || value;
}

export default function App() {
  const canvasRef = useRef(null);
  const camRef = useRef(null);
  const animRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState(null);

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
        if (message.type === "frame") setFrame(message);
        if (message.type === "generation") setStatus(message);
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
      const goal = frame?.goal || status?.goal || metadata?.goal?.fixed_goal_xyz;

      ctx.fillStyle = "#08120d";
      ctx.fillRect(0, 0, width, height);

      drawArena(ctx, cam, width, height, terrain);
      drawGoal(ctx, cam, width, height, goal);
      drawModel(ctx, frame, cam, width, height, robot);

      if (!frame) {
        ctx.fillStyle = "#587760";
        ctx.font = "18px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(connected ? "Waiting for model frames..." : `Connecting to ${WS_URL}`, width / 2, height / 2);
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(render);
    }

    animRef.current = requestAnimationFrame(render);
    return () => {
      cancelled = true;
      cancelAnimationFrame(animRef.current);
    };
  }, [connected, frame, metadata, status]);

  const viewerResetSeconds = metadata.training?.viewer_reset_s ?? DEFAULT_METADATA.training.viewer_reset_s;
  const remainingReset = Math.max(0, viewerResetSeconds - (frame?.time_s ?? 0));
  const checkpointLoaded = basename(status?.checkpoint_loaded);
  const simulatorBackend = status?.simulator_backend || metadata.simulator?.backend || "jax";

  return (
    <div className="app-shell">
      <canvas ref={canvasRef} className="arena-canvas" />

      <aside className="hud-card hud-card--primary">
        <div className="hud-kicker">
          Current Model Replay
          <span className={`status-pill ${connected ? "status-pill--live" : "status-pill--offline"}`}>
            {connected ? "connected" : "offline"}
          </span>
        </div>
        <h1 className="hud-title">Quadruped Viewer</h1>
        <p className="hud-subtitle">
          config <strong>{metadata.config_name}</strong> on <strong>{simulatorBackend}</strong>
        </p>

        <div className="metrics-grid">
          <div><span>checkpoint</span><strong>{checkpointLoaded}</strong></div>
          <div><span>reset in</span><strong>{remainingReset.toFixed(1)}s</strong></div>
          <div><span>generation</span><strong>{status?.generation ?? 0}</strong></div>
          <div><span>mean reward</span><strong>{status?.mean_reward?.toFixed(2) ?? "0.00"}</strong></div>
          <div><span>best reward</span><strong>{status?.best_reward?.toFixed(2) ?? "0.00"}</strong></div>
          <div><span>frame time</span><strong>{frame?.time_s?.toFixed(1) ?? "0.0"}s</strong></div>
        </div>

        <p className="hud-note">
          This viewer replays the current model only. It does not train or mutate weights, and it resets every {viewerResetSeconds.toFixed(0)} seconds.
        </p>
      </aside>

      <aside className="hud-card hud-card--secondary">
        <div className="spec-grid">
          <div><span>terrain</span><strong>{metadata.terrain.kind}</strong></div>
          <div><span>steps</span><strong>{metadata.terrain.step_count}</strong></div>
          <div><span>step width</span><strong>{metadata.terrain.step_width_m}m</strong></div>
          <div><span>step height</span><strong>{metadata.terrain.step_height_m}m</strong></div>
          <div><span>population</span><strong>{metadata.training.population_size}</strong></div>
          <div><span>episode</span><strong>{metadata.training.episode_s}s</strong></div>
          <div><span>backend</span><strong>{simulatorBackend}</strong></div>
          <div><span>mode</span><strong>{metadata.mode}</strong></div>
        </div>
        <p className="hud-note">Drag to orbit. Scroll to zoom. The goal marker and reset timer come from the active replay stream.</p>
      </aside>
    </div>
  );
}
