import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";
const FIELD = 2.0;

// Isometric projection: world (x, y, z) → canvas (px, py)
function project(x, y, z, W, H) {
  const scale = Math.min(W, H) * 0.18;
  const cx = W * 0.5;
  const cy = H * 0.42;
  const px = cx + (x - y) * scale;
  const py = cy - z * scale * 1.6 + (x + y) * scale * 0.38;
  return [px, py];
}

function drawGrid(ctx, W, H) {
  ctx.save();
  ctx.strokeStyle = "#1a2a1a";
  ctx.lineWidth = 0.5;
  const step = 0.5;
  for (let a = -FIELD; a <= FIELD + 0.01; a += step) {
    const [x0, y0] = project(a, -FIELD, 0, W, H);
    const [x1, y1] = project(a, FIELD, 0, W, H);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const [x2, y2] = project(-FIELD, a, 0, W, H);
    const [x3, y3] = project(FIELD, a, 0, W, H);
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
  }
  ctx.strokeStyle = "#2a4a2a";
  ctx.lineWidth = 2;
  const corners = [[-FIELD,-FIELD],[FIELD,-FIELD],[FIELD,FIELD],[-FIELD,FIELD],[-FIELD,-FIELD]];
  ctx.beginPath();
  corners.forEach(([a, b], i) => {
    const [px, py] = project(a, b, 0, W, H);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

function drawGoal(ctx, goal, W, H) {
  const [gx, gy] = project(goal[0], goal[1], 0, W, H);
  const [tx, ty] = project(goal[0], goal[1], goal[2], W, H);
  ctx.save();
  ctx.strokeStyle = "#ffcc00";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffcc00";
  ctx.beginPath(); ctx.arc(tx, ty, 8, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#ffcc0055";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(gx, gy, 16, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

function drawBody(ctx, body, W, H) {
  if (!body) return;
  const [x, y, z] = body.pos;
  const [roll, pitch] = body.rot;
  const L = 0.14, Wd = 0.06, Hd = 0.04;
  const offsets = [
    [ L,  Wd,  Hd], [ L, -Wd,  Hd], [-L,  Wd,  Hd], [-L, -Wd,  Hd],
    [ L,  Wd, -Hd], [ L, -Wd, -Hd], [-L,  Wd, -Hd], [-L, -Wd, -Hd],
  ];
  const pts = offsets.map(([dx, dy, dz]) => {
    const ry = dy * Math.cos(roll) - dz * Math.sin(roll);
    const rz1 = dy * Math.sin(roll) + dz * Math.cos(roll);
    const rx2 = dx * Math.cos(pitch) + rz1 * Math.sin(pitch);
    const rz2 = -dx * Math.sin(pitch) + rz1 * Math.cos(pitch);
    return project(x + rx2, y + ry, z + rz2, W, H);
  });
  const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  ctx.save();
  ctx.strokeStyle = "#8be08b";
  ctx.lineWidth = 2.5;
  edges.forEach(([a, b]) => {
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  });
  ctx.restore();
}

function drawLegs(ctx, legs, W, H) {
  if (!legs) return;
  legs.forEach(leg => {
    const color = leg.contact_mode === "static" ? "#44ff88"
                : leg.contact_mode === "kinetic" ? "#ff8844"
                : "#446688";
    const [mx, my] = project(...leg.mount, W, H);
    const [fx, fy] = project(...leg.foot, W, H);
    const [lcx, lcy] = project(...leg.com, W, H);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(fx, fy); ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(fx, fy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#ffffff55";
    ctx.beginPath(); ctx.arc(lcx, lcy, 3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  });
}

function drawCOM(ctx, com, W, H) {
  if (!com) return;
  const [px, py] = project(...com, W, H);
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#cccccc";
  ctx.font = "11px monospace";
  ctx.fillText("COM", px + 8, py - 6);
  ctx.restore();
}

function RewardChart({ history }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || history.length < 2) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0a120a";
    ctx.fillRect(0, 0, w, h);
    const min = Math.min(...history), max = Math.max(...history);
    const range = max - min || 1;
    ctx.strokeStyle = "#33ff66";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    history.forEach((v, i) => {
      const px = (i / (history.length - 1)) * w;
      const py = h - ((v - min) / range) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    if (min < 0 && max > 0) {
      const zy = h - ((0 - min) / range) * (h - 8) - 4;
      ctx.strokeStyle = "#334433";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [history]);
  return <canvas ref={ref} width={320} height={90} style={{ width: "100%", height: 90 }} />;
}

export default function App() {
  const canvasRef = useRef(null);
  const stateRef = useRef({ step: null });
  const animRef = useRef(null);
  const [genInfo, setGenInfo] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws;
    function connect() {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === "step") stateRef.current.step = msg;
        else if (msg.type === "generation") setGenInfo(msg);
      };
    }
    connect();
    return () => ws && ws.close();
  }, []);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) { animRef.current = requestAnimationFrame(render); return; }
    const W = window.innerWidth, H = window.innerHeight;
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#060e06";
    ctx.fillRect(0, 0, W, H);
    drawGrid(ctx, W, H);

    const { step } = stateRef.current;
    if (step) {
      drawGoal(ctx, step.goal, W, H);
      drawBody(ctx, step.body, W, H);
      drawLegs(ctx, step.legs, W, H);
      drawCOM(ctx, step.com, W, H);

      const pct = step.step / step.total_steps;
      ctx.fillStyle = "#0d1f0d";
      ctx.fillRect(20, H - 32, W - 40, 10);
      ctx.fillStyle = "#33cc66";
      ctx.fillRect(20, H - 32, (W - 40) * pct, 10);
      ctx.fillStyle = "#aaa";
      ctx.font = "11px monospace";
      ctx.fillText(`episode ${(pct * 100).toFixed(0)}%   reward ${step.reward.toFixed(3)}   t=${step.time_s.toFixed(2)}s`, 24, H - 38);
    } else {
      drawGoal(ctx, [1, 0, 0.16], W, H);
      ctx.fillStyle = "#335533";
      ctx.font = "18px monospace";
      ctx.textAlign = "center";
      ctx.fillText(connected ? "Waiting for first episode…" : "Connecting to ws://localhost:8000…", W / 2, H / 2);
      ctx.textAlign = "left";
    }

    animRef.current = requestAnimationFrame(render);
  }, [connected]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [render]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#060e06", overflow: "hidden", fontFamily: "monospace" }}>
      <canvas ref={canvasRef} style={{ position: "absolute", inset: 0 }} />

      {/* HUD */}
      <div style={{
        position: "absolute", top: 16, right: 16,
        background: "#0a180add", border: "1px solid #223322",
        borderRadius: 8, padding: "14px 18px", width: 340, color: "#88cc88",
        fontSize: 12, lineHeight: 1.8, pointerEvents: "none",
      }}>
        <div style={{ fontSize: 15, fontWeight: "bold", color: "#44ff88", marginBottom: 8 }}>
          SNN Quadruped Training
          <span style={{ marginLeft: 10, fontSize: 11, color: connected ? "#44ff44" : "#ff4444" }}>
            ● {connected ? "live" : "offline"}
          </span>
        </div>
        {genInfo ? (
          <>
            <div>generation      <b style={{ color: "#aaffaa" }}>{genInfo.generation}</b></div>
            <div>mean reward     <b style={{ color: "#aaffaa" }}>{genInfo.mean_reward.toFixed(3)}</b></div>
            <div>best reward     <b style={{ color: "#ffff88" }}>{genInfo.best_reward.toFixed(3)}</b></div>
            <div>goal xyz        <b style={{ color: "#ffcc88" }}>[{genInfo.goal.map(v => v.toFixed(2)).join(", ")}]</b></div>
            <div style={{ marginTop: 10, marginBottom: 4, color: "#556655", fontSize: 10 }}>mean reward / generation</div>
            <RewardChart history={genInfo.rewards_history} />
          </>
        ) : (
          <div style={{ color: "#446644" }}>—</div>
        )}
        <div style={{ marginTop: 12, color: "#334433", fontSize: 10, lineHeight: 1.6 }}>
          inputs 48 = goal·3 + COM·3 + body·3 + feet·12 + leg-COM·12 + body-IMU·3 + leg-IMU·12<br />
          hidden 64 LIF neurons (τ=20ms)   outputs 4 motor vel<br />
          ES: pop=16  σ=0.05  lr=0.02  episode=6s
        </div>
      </div>

      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 50, right: 16,
        background: "#0a180a99", border: "1px solid #1a2a1a",
        borderRadius: 6, padding: "8px 14px", color: "#88cc88", fontSize: 11,
        pointerEvents: "none",
      }}>
        <span style={{ color: "#44ff88" }}>●</span> static&ensp;
        <span style={{ color: "#ff8844" }}>●</span> kinetic&ensp;
        <span style={{ color: "#446688" }}>●</span> airborne&ensp;
        <span style={{ color: "#ffcc00" }}>★</span> goal
      </div>
    </div>
  );
}
