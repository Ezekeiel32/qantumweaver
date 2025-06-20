import React, { useRef, useEffect, useState } from "react";

interface Trace {
  data: number[];
  color?: string; // e.g. "#ffe066", "#ff00ff", "#00ffe7"
  label?: string;
}
interface Marker {
  x: number; // index in data
  y: number; // value
  label?: string;
  color?: string;
}
interface Overlay {
  text: string;
  color?: string;
  x?: number;
  y?: number;
  fontSize?: number;
}
interface NeonAnalyzerChartProps {
  traces: Trace[];
  width?: number;
  height?: number;
  overlays?: Overlay[];
  markers?: Marker[];
  gridColor?: string;
  bgColor?: string;
  showLegend?: boolean;
  yMin?: number;
  yMax?: number;
  jobName?: string;
}

const defaultColors = ["#ffe066", "#ff00ff", "#00ffe7", "#39ff14", "#ffffff"];

// --- Trace color presets ---
const colorPresets = [
  ["#ffe066", "#ff00ff", "#00ffe7", "#39ff14", "#ffffff"], // Neon
  ["#ffb300", "#1e88e5", "#43a047", "#e53935", "#8e24aa"], // Pro VNA
  ["#00e5ff", "#ff4081", "#ffd600", "#00c853", "#d500f9"], // Modern
];

export default function NeonAnalyzerChart({
  traces,
  width = 680,
  height = 340,
  overlays = [],
  markers = [],
  gridColor = "#00ffe7",
  bgColor = "#10131a",
  showLegend = true,
  yMin,
  yMax,
  jobName,
}: NeonAnalyzerChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(8); // Default: zoomed in all the way
  const [pan, setPan] = useState(1); // 1 = right (latest epoch)
  const [isPanning, setIsPanning] = useState(false);
  const [panStartX, setPanStartX] = useState<number | null>(null);
  const [panVelocity, setPanVelocity] = useState(0);
  const [lastPanTime, setLastPanTime] = useState<number | null>(null);

  // --- Measurement marker state ---
  const [measureMarkers, setMeasureMarkers] = useState<{ x: number; y: number }[]>([]); // up to 2
  const [draggedMarker, setDraggedMarker] = useState<number | null>(null);
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null);
  const [showMarkerTable, setShowMarkerTable] = useState(true);
  const [showDeltaTable, setShowDeltaTable] = useState(true);
  const [referenceLines, setReferenceLines] = useState<number[]>([]); // y-values
  const [showMiniMap, setShowMiniMap] = useState(true);

  // --- Afterglow buffer ---
  const afterglowRef = useRef<HTMLCanvasElement>(null);

  // --- Track if user has manually panned ---
  const [userPanned, setUserPanned] = useState(false);

  // --- Trace math operations ---
  const [traceMath, setTraceMath] = useState<{ op: string; t1: number; t2: number } | null>(null);
  const [colorPresetIdx, setColorPresetIdx] = useState(0);
  const traceColors = colorPresets[colorPresetIdx];

  // --- Trace math operations ---
  function getMathTrace() {
    if (!traceMath) return null;
    const { op, t1, t2 } = traceMath;
    if (!traces[t1] || !traces[t2]) return null;
    const d1 = traces[t1].data;
    const d2 = traces[t2].data;
    let data: number[] = [];
    if (op === "-") data = d1.map((v, i) => v - (d2[i] ?? 0));
    if (op === "/") data = d1.map((v, i) => (d2[i] !== 0 ? v / d2[i] : 0));
    if (op === "+") data = d1.map((v, i) => v + (d2[i] ?? 0));
    if (op === "*") data = d1.map((v, i) => v * (d2[i] ?? 0));
    return { data, color: "#fff176", label: `${traces[t1].label || `T${t1+1}`} ${op} ${traces[t2].label || `T${t2+1}`}` };
  }

  // --- Reference line controls ---
  function addReferenceLine(y: number) {
    setReferenceLines(prev => [...prev, y]);
  }
  function removeReferenceLine(idx: number) {
    setReferenceLines(prev => prev.filter((_, i) => i !== idx));
  }

  // --- Mouse/touch pan handlers ---
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    setPanStartX(e.clientX);
    setLastPanTime(Date.now());
    setUserPanned(true);
    // Check for marker drag
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    measureMarkers.forEach((m, idx) => {
      const n = traces[0]?.data.length || 0;
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      const markerX = 50 + ((m.x - startIdx) / (windowSize - 1)) * (width - 70);
      const minY = yMin ?? Math.min(...traces[0].data);
      const maxY = yMax ?? Math.max(...traces[0].data);
      const markerY = height - 30 - ((m.y - minY) / (maxY - minY)) * (height - 40);
      if (Math.abs(x - markerX) < 14 && Math.abs(y - markerY) < 14) {
        setDraggedMarker(idx);
      }
    });
  };
  const handleMouseUp = () => {
    setIsPanning(false);
    setPanStartX(null);
    setPanVelocity(0);
    setDraggedMarker(null);
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggedMarker !== null) {
      // Drag marker horizontally
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const x = e.clientX - rect.left;
      const n = traces[0]?.data.length || 0;
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      let epoch = Math.round(((x - 50) / (width - 70)) * (windowSize - 1)) + startIdx;
      epoch = Math.max(startIdx, Math.min(startIdx + windowSize - 1, epoch));
      const y = traces[0]?.data[epoch] ?? 0;
      setMeasureMarkers(prev => prev.map((m, i) => i === draggedMarker ? { x: epoch, y } : m));
      setUserPanned(true);
      return;
    }
    if (!isPanning || panStartX === null) return;
    const dx = e.clientX - panStartX;
    setPan(prev => {
      let next = Math.max(0, Math.min(1, prev - dx / width / zoom));
      setPanVelocity(dx / ((Date.now() - (lastPanTime || Date.now())) || 1));
      setLastPanTime(Date.now());
      setUserPanned(true);
      return next;
    });
    setPanStartX(e.clientX);
  };
  // --- Inertial scrolling ---
  useEffect(() => {
    if (!isPanning && Math.abs(panVelocity) > 0.001) {
      const raf = requestAnimationFrame(() => {
        setPan(prev => {
          let next = Math.max(0, Math.min(1, prev - panVelocity * 0.05));
          if (next === 0 || next === 1) setPanVelocity(0);
          else setPanVelocity(panVelocity * 0.92);
          return next;
        });
      });
      return () => cancelAnimationFrame(raf);
    }
  }, [isPanning, panVelocity]);

  // --- Keyboard shortcuts for pan/zoom ---
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") setUserPanned(true);
      if (e.key === "+" || e.key === "=") setZoom(z => Math.min(16, z * 2));
      if (e.key === "-") setZoom(z => Math.max(1, z / 2));
      if (e.key === "m") setShowMiniMap(v => !v);
      if (e.key === "t") setShowMarkerTable(v => !v);
      if (e.key === "d") setShowDeltaTable(v => !v);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  // --- Measurement marker placing/removal ---
  const handleCanvasClick = (e: React.MouseEvent) => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Remove marker if clicking on it
    let removed = false;
    measureMarkers.forEach((m, idx) => {
      const n = traces[0]?.data.length || 0;
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      const markerX = 50 + ((m.x - startIdx) / (windowSize - 1)) * (width - 70);
      const minY = yMin ?? Math.min(...traces[0].data);
      const maxY = yMax ?? Math.max(...traces[0].data);
      const markerY = height - 30 - ((m.y - minY) / (maxY - minY)) * (height - 40);
      if (Math.abs(x - markerX) < 14 && Math.abs(y - markerY) < 14) {
        setMeasureMarkers(prev => prev.filter((_, i) => i !== idx));
        removed = true;
      }
    });
    if (removed) return;
    // Place marker
    const n = traces[0]?.data.length || 0;
    const windowSize = Math.floor(n / zoom);
    const startIdx = Math.floor((n - windowSize) * pan);
    const endIdx = Math.min(n, startIdx + windowSize);
    const epoch = Math.round(((x - 50) / (width - 70)) * (windowSize - 1)) + startIdx;
    if (epoch >= startIdx && epoch < endIdx) {
      const yVal = traces[0]?.data[epoch] ?? 0;
      setMeasureMarkers(prev => prev.length === 2 ? [{ x: epoch, y: yVal }] : [...prev, { x: epoch, y: yVal }]);
    }
  };
  // --- Hover for tooltips ---
  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const n = traces[0]?.data.length || 0;
    const windowSize = Math.floor(n / zoom);
    const startIdx = Math.floor((n - windowSize) * pan);
    const endIdx = Math.min(n, startIdx + windowSize);
    const epoch = Math.round(((x - 50) / (width - 70)) * (windowSize - 1)) + startIdx;
    if (epoch >= startIdx && epoch < endIdx) setHoveredEpoch(epoch);
    else setHoveredEpoch(null);
  };

  // --- Catmull-Rom spline helper ---
  function catmullRomSpline(points: [number, number][], samples = 8) {
    const result: [number, number][] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i - 1] || points[i];
      const p1 = points[i];
      const p2 = points[i + 1] || points[i];
      const p3 = points[i + 2] || points[i + 1] || points[i];
      for (let t = 0; t < samples; t++) {
        const tt = t / samples;
        const tt2 = tt * tt;
        const tt3 = tt2 * tt;
        const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * tt + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * tt2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * tt3);
        const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * tt + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * tt2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * tt3);
        result.push([x, y]);
      }
    }
    result.push(points[points.length - 1]);
    return result;
  }

  // --- Mini-map/overview rendering ---
  function renderMiniMapExternal() {
    if (!showMiniMap || !traces[0]) return null;
    const n = traces[0].data.length;
    const windowSize = Math.floor(n / zoom);
    const startIdx = Math.floor((n - windowSize) * pan);
    const endIdx = Math.min(n, startIdx + windowSize);
    // VNA-style mini-map: dark, neon, grid, window rectangle
    return (
      <div style={{
        width: 180, height: 54, background: "#181a22F2", border: "2px solid #00ffe7", borderRadius: 10, zIndex: 3, boxShadow: "0 0 16px #00ffe7, 0 2px 16px #0008", display: "flex", alignItems: "center", justifyContent: "center", padding: 4, marginBottom: 18
      }}>
        <svg width={160} height={36} style={{ display: 'block' }}>
          {/* Neon grid */}
          <g>
            {[0, 40, 80, 120, 160].map(x => (
              <line key={x} x1={x} y1={0} x2={x} y2={36} stroke="#00ffe7" strokeWidth={x % 80 === 0 ? 1.2 : 0.5} opacity={x % 80 === 0 ? 0.18 : 0.09} />
            ))}
            {[0, 18, 36].map(y => (
              <line key={y} x1={0} y1={y} x2={160} y2={y} stroke="#00ffe7" strokeWidth={y === 18 ? 1.2 : 0.5} opacity={y === 18 ? 0.18 : 0.09} />
            ))}
          </g>
          {traces.map((trace, tIdx) => {
            const d = trace.data;
            const minY = yMin ?? Math.min(...d);
            const maxY = yMax ?? Math.max(...d);
            const points = d.map((v, i) => {
              const x = (i / (d.length - 1)) * 160;
              const y = 36 - ((v - minY) / (maxY - minY)) * 34;
              return `${x},${y}`;
            }).join(" ");
            return <polyline key={tIdx} points={points} fill="none" stroke={trace.color || defaultColors[tIdx % defaultColors.length]} strokeWidth={1.2} opacity={0.8} />;
          })}
          {/* Window rectangle */}
          {(n > 0 && windowSize > 0) ? (
            <rect
              x={isFinite(startIdx / n) ? (startIdx / n) * 160 : 0}
              y={2}
              width={isFinite(windowSize / n) ? (windowSize / n) * 160 : 0}
              height={32}
              fill="#00ffe7"
              opacity={0.10}
              stroke="#00ffe7"
              strokeWidth={1.2}
            />
          ) : null}
        </svg>
      </div>
    );
  }

  // --- Marker/Delta Table rendering ---
  function renderMarkerDeltaPanel() {
    if (!showMarkerTable && !showDeltaTable) return null;
    return (
      <div style={{
        background: "#181a22F2",
        border: "2px solid orange",
        borderRadius: 12,
        padding: "10px 18px 10px 18px",
        minWidth: 120,
        color: "orange",
        fontFamily: "'Share Tech Mono', 'VT323', monospace",
        fontSize: 13,
        boxShadow: "0 0 16px orange, 0 2px 16px #0008",
        textAlign: "left",
        marginLeft: 0,
        marginTop: 0,
        maxWidth: 180,
        marginBottom: 18
      }}>
        {showMarkerTable && measureMarkers.length > 0 && (
          <div style={{ marginBottom: showDeltaTable && measureMarkers.length === 2 ? 8 : 0 }}>
            <b style={{ fontSize: 13, color: "#ffe066" }}>Markers</b>
            {measureMarkers.map((m, i) => <div key={i}>M{i + 1}: E{m.x + 1} = {m.y.toFixed(4)}</div>)}
          </div>
        )}
        {showDeltaTable && measureMarkers.length === 2 && (
          <div>
            <b style={{ fontSize: 13, color: "#39ff14" }}>Delta</b>
            <div style={{ color: "#39ff14" }}>ΔE: {Math.abs(measureMarkers[1].x - measureMarkers[0].x)}</div>
            <div style={{ color: "#39ff14" }}>ΔY: {(measureMarkers[1].y - measureMarkers[0].y).toFixed(4)}</div>
          </div>
        )}
      </div>
    );
  }
  function renderReferenceLines(ctx: CanvasRenderingContext2D, minY: number, maxY: number) {
    referenceLines.forEach((yVal, i) => {
      const y = height - 30 - ((yVal - minY) / (maxY - minY)) * (height - 40);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(width - 10, y);
      ctx.strokeStyle = "#ff00ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.shadowColor = "#ff00ff";
      ctx.shadowBlur = 10;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "12px 'Share Tech Mono', 'VT323', monospace";
      ctx.fillStyle = "#ff00ff";
      ctx.fillText(`Ref ${i + 1}: ${yVal.toFixed(4)}`, width - 100, y - 6);
      ctx.restore();
    });
  }

  // --- Marker table rendering ---
  function renderMarkerTablePanel() {
    if (!measureMarkers.length) return null;
    return (
      <div style={{
        background: "#181a22F2",
        border: "2px solid #ffe066",
        borderRadius: 12,
        padding: "10px 18px 10px 18px",
        minWidth: 160,
        color: "#ffe066",
        fontFamily: "'Share Tech Mono', 'VT323', monospace",
        fontSize: 13,
        boxShadow: "0 0 16px #ffe066, 0 2px 16px #0008",
        textAlign: "left",
        marginBottom: 18,
        marginTop: 0,
        maxWidth: 220
      }}>
        <b style={{ fontSize: 14, color: "#ffe066" }}>Marker Table</b>
        <table style={{ width: "100%", marginTop: 6, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#39ff14", fontWeight: 600 }}>
              <td style={{ padding: 2 }}>#</td>
              <td style={{ padding: 2 }}>Epoch</td>
              <td style={{ padding: 2 }}>Value</td>
            </tr>
          </thead>
          <tbody>
            {measureMarkers.map((m, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #222" }}>
                <td style={{ padding: 2 }}>M{i + 1}</td>
                <td style={{ padding: 2 }}>E{m.x + 1}</td>
                <td style={{ padding: 2 }}>{m.y.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // --- Trace color preset controls ---
  function renderColorPresetPanel() {
    return (
      <div style={{ marginBottom: 18, background: "#181a22F2", border: "2px solid #00ffe7", borderRadius: 10, boxShadow: "0 0 8px #00ffe7", padding: "6px 12px", fontFamily: "'Share Tech Mono', 'VT323', monospace", color: "#00ffe7", fontSize: 13 }}>
        <b>Trace Colors:</b>
        {colorPresets.map((preset, idx) => (
          <button key={idx} onClick={() => setColorPresetIdx(idx)} style={{ marginLeft: 8, background: idx === colorPresetIdx ? "#00ffe7" : "#10131a", color: idx === colorPresetIdx ? "#10131a" : "#00ffe7", border: "1.5px solid #00ffe7", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontWeight: 600 }}>{`Preset ${idx + 1}`}</button>
        ))}
      </div>
    );
  }

  // --- Trace math controls ---
  function renderTraceMathPanel() {
    if (traces.length < 2) return null;
    return (
      <div style={{ marginBottom: 18, background: "#181a22F2", border: "2px solid #fff176", borderRadius: 10, boxShadow: "0 0 8px #fff176", padding: "6px 12px", fontFamily: "'Share Tech Mono', 'VT323', monospace", color: "#fff176", fontSize: 13 }}>
        <b>Trace Math:</b>
        <select value={traceMath ? traceMath.t1 : 0} onChange={e => setTraceMath(traceMath ? { ...traceMath, t1: +e.target.value } : { op: '-', t1: +e.target.value, t2: 1 })} style={{ marginLeft: 8 }}>
          {traces.map((t, i) => <option key={i} value={i}>{t.label || `Trace ${i + 1}`}</option>)}
        </select>
        <select value={traceMath ? traceMath.op : '-'} onChange={e => setTraceMath(traceMath ? { ...traceMath, op: e.target.value } : { op: e.target.value, t1: 0, t2: 1 })} style={{ marginLeft: 4 }}>
          <option value="-">-</option>
          <option value="/">/</option>
          <option value="+">+</option>
          <option value="*">*</option>
        </select>
        <select value={traceMath ? traceMath.t2 : 1} onChange={e => setTraceMath(traceMath ? { ...traceMath, t2: +e.target.value } : { op: '-', t1: 0, t2: +e.target.value })} style={{ marginLeft: 4 }}>
          {traces.map((t, i) => <option key={i} value={i}>{t.label || `Trace ${i + 1}`}</option>)}
        </select>
        <button onClick={() => setTraceMath(null)} style={{ marginLeft: 8, background: "#fff176", color: "#10131a", border: "1.5px solid #fff176", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontWeight: 600 }}>Clear</button>
      </div>
    );
  }

  // --- Reference line controls panel ---
  function renderReferenceLinePanel() {
    const [input, setInput] = useState("");
    return (
      <div style={{ marginBottom: 18, background: "#181a22F2", border: "2px solid #ff00ff", borderRadius: 10, boxShadow: "0 0 8px #ff00ff", padding: "6px 12px", fontFamily: "'Share Tech Mono', 'VT323', monospace", color: "#ff00ff", fontSize: 13 }}>
        <b>Reference Lines:</b>
        <input type="number" value={input} onChange={e => setInput(e.target.value)} placeholder="Y value" style={{ marginLeft: 8, width: 60, background: "#10131a", color: "#ff00ff", border: "1px solid #ff00ff", borderRadius: 4, padding: "2px 4px" }} />
        <button onClick={() => { if (input !== "") { addReferenceLine(Number(input)); setInput(""); } }} style={{ marginLeft: 4, background: "#ff00ff", color: "#10131a", border: "1.5px solid #ff00ff", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontWeight: 600 }}>Add</button>
        {referenceLines.length > 0 && (
          <div style={{ marginTop: 6 }}>
            {referenceLines.map((y, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: 2 }}>
                <span style={{ marginRight: 8 }}>Y={y.toFixed(4)}</span>
                <button onClick={() => removeReferenceLine(i)} style={{ background: "#10131a", color: "#ff00ff", border: "1px solid #ff00ff", borderRadius: 4, padding: "0 6px", cursor: "pointer", fontSize: 12 }}>×</button>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // --- Scrollbar state ---
  const n = traces[0]?.data.length || 0;
  const windowSize = Math.floor(n / zoom);
  const panMin = 0;
  const panMax = 1;
  const panValue = pan;
  function handleScrollbarChange(e: React.ChangeEvent<HTMLInputElement>) {
    setPan(Number(e.target.value));
    setUserPanned(true);
  }
  // --- Zoom/pan buttons ---
  function handleZoomToLatest() {
    setZoom(8); // max zoom in
    setPan(1); // latest
    setUserPanned(false);
  }
  function handleZoomOutAll() {
    setZoom(1); // min zoom (show all)
    setPan(0); // start
    setUserPanned(false);
  }

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.1;
    for (let i = 0; i <= 10; i++) {
      const y = height - 30 - (i / 10) * (height - 40);
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
    }
    for (let i = 0; i <= 10; i++) {
      const x = 50 + (i / 10) * (width - 70);
      ctx.beginPath();
      ctx.moveTo(x, 10);
      ctx.lineTo(x, height - 30);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Calculate visible window
    const n = Math.max(...traces.map(t => t.data.length));
    if (n === 0) return;
    const windowSize = Math.floor(n / zoom);
    const startIdx = Math.floor((n - windowSize) * pan);
    const endIdx = Math.min(n, startIdx + windowSize);

    // Auto-pan to latest data if user hasn't manually panned
    if (!userPanned && !isPanning && traces[0]?.data.length > 0) {
      setPan(1);
    }

    // Calculate y-axis range
    let minY = yMin ?? Math.min(...traces.map(t => Math.min(...t.data.slice(startIdx, endIdx))));
    let maxY = yMax ?? Math.max(...traces.map(t => Math.max(...t.data.slice(startIdx, endIdx))));
    const padding = (maxY - minY) * 0.1;
    minY -= padding;
    maxY += padding;

    // Draw traces
    traces.forEach((trace, i) => {
      if (!trace.data.length) return;
      const color = trace.color || defaultColors[i % defaultColors.length];
      
      // Draw line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let j = startIdx; j < endIdx; j++) {
        const x = 50 + ((j - startIdx) / (windowSize - 1)) * (width - 70);
        const y = height - 30 - ((trace.data[j] - minY) / (maxY - minY)) * (height - 40);
        if (j === startIdx) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw points
      ctx.fillStyle = color;
      for (let j = startIdx; j < endIdx; j++) {
        const x = 50 + ((j - startIdx) / (windowSize - 1)) * (width - 70);
        const y = height - 30 - ((trace.data[j] - minY) / (maxY - minY)) * (height - 40);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw axes
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, 10);
    ctx.lineTo(50, height - 30);
    ctx.lineTo(width - 20, height - 30);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = gridColor;
    ctx.font = "12px 'Share Tech Mono'";
    ctx.textAlign = "right";
    for (let i = 0; i <= 10; i++) {
      const y = height - 30 - (i / 10) * (height - 40);
      const value = minY + (i / 10) * (maxY - minY);
      ctx.fillText(value.toFixed(1), 45, y + 4);
    }
    ctx.textAlign = "center";
    for (let i = 0; i <= 10; i++) {
      const x = 50 + (i / 10) * (width - 70);
      const epoch = Math.round(startIdx + (i / 10) * (endIdx - startIdx));
      ctx.fillText(epoch.toString(), x, height - 10);
    }

    // Draw legend if enabled
    if (showLegend && traces.length > 0) {
      const legendX = width - 150;
      const legendY = 30;
      traces.forEach((trace, i) => {
        if (!trace.label) return;
        const color = trace.color || defaultColors[i % defaultColors.length];
        const y = legendY + i * 20;
        
        ctx.fillStyle = color;
        ctx.fillRect(legendX, y, 15, 2);
        
        ctx.fillStyle = gridColor;
        ctx.textAlign = "left";
        ctx.fillText(trace.label, legendX + 20, y + 5);
      });
    }

    // Draw hover tooltip
    if (hoveredEpoch !== null && hoveredEpoch >= startIdx && hoveredEpoch < endIdx) {
      const x = 50 + ((hoveredEpoch - startIdx) / (windowSize - 1)) * (width - 70);
      
      // Vertical line
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, 10);
      ctx.lineTo(x, height - 30);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tooltip box
      ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
      ctx.fillRect(x + 5, 10, 120, 15 + traces.length * 15);
      
      // Tooltip text
      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.fillText(`Epoch ${hoveredEpoch}`, x + 10, 22);
      traces.forEach((trace, i) => {
        const value = trace.data[hoveredEpoch];
        const color = trace.color || defaultColors[i % defaultColors.length];
        ctx.fillStyle = color;
        ctx.fillText(`${trace.label}: ${value.toFixed(3)}`, x + 10, 37 + i * 15);
      });
    }
  }, [traces, width, height, gridColor, bgColor, zoom, pan, isPanning, hoveredEpoch, showLegend, userPanned]);

  // --- Auto-scroll to latest epoch only if not user-panned ---
  useEffect(() => {
    if (!userPanned) setPan(1);
  }, [traces.length > 0 ? traces[0].data.length : 0]);

  // --- Scrollbars for overflow ---
  const scrollContainerStyle = {
    width: width + 32, // extra for scrollbars
    height: height + 32,
    overflow: "auto" as const,
    position: "relative" as const,
    background: "none"
  };

  // --- Zoom controls (restore) ---
  const handleZoomIn = () => setZoom(z => Math.min(16, z * 2));
  const handleZoomOut = () => setZoom(z => Math.max(1, z / 2));

  const hasEnoughData = traces.some(trace => trace.data && trace.data.length >= 2);

  return (
    <div style={{ ...scrollContainerStyle, display: "flex", flexDirection: "row", alignItems: "flex-start" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {/* Job name header */}
        {jobName && (
          <div style={{
            width,
            textAlign: "center",
            fontFamily: "'Share Tech Mono', 'VT323', monospace",
            fontSize: 22,
            color: "#ffe066",
            letterSpacing: 1.5,
            marginBottom: 8,
            textShadow: "0 0 12px #ffe066, 0 0 2px #10131a"
          }}>{jobName}</div>
        )}
        <div style={{ position: "relative", width, height, boxShadow: "0 0 32px #00ffe7, 0 2px 32px #0008", borderRadius: 16 }}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseMove={e => { handleMouseMove(e); handleCanvasMouseMove(e); }}
          onMouseLeave={handleMouseUp}
          onClick={handleCanvasClick}
          tabIndex={0}
        >
          {hasEnoughData ? (
            <>
              <canvas ref={afterglowRef} width={width} height={height} style={{ display: "none" }} />
              <canvas ref={canvasRef} width={width} height={height} style={{ display: "block", width, height, background: bgColor, borderRadius: 12, boxShadow: "0 0 24px #00ffe7", cursor: zoom > 1 ? "grab" : "default" }} />
            </>
          ) : (
            <div style={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: bgColor,
              borderRadius: 12,
              color: "#00ffe7",
              fontFamily: "'Share Tech Mono', 'VT323', monospace",
              fontSize: 24,
              textShadow: "0 0 12px #00ffe7, 0 0 2px #10131a"
            }}>
              No training data yet
            </div>
          )}
          <div style={{ position: "absolute", top: 10, right: 16, zIndex: 2, display: "flex", flexDirection: "column", gap: 4 }}>
            <button onClick={handleZoomIn} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, marginBottom: 2, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>+</button>
            <button onClick={handleZoomOut} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>-</button>
          </div>
        </div>
        {/* Zoom/pan buttons and scrollbar below chart */}
        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", width, marginTop: 8, gap: 12 }}>
          <button onClick={handleZoomToLatest} style={{ background: "#10131a", color: "#00ffe7", border: "1.5px solid #00ffe7", borderRadius: 8, padding: "4px 14px", fontSize: 15, fontFamily: "'Share Tech Mono', 'VT323', monospace", fontWeight: 600, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>Current</button>
          <button onClick={handleZoomOutAll} style={{ background: "#10131a", color: "#ffe066", border: "1.5px solid #ffe066", borderRadius: 8, padding: "4px 14px", fontSize: 15, fontFamily: "'Share Tech Mono', 'VT323', monospace", fontWeight: 600, cursor: "pointer", boxShadow: "0 0 8px #ffe066" }}>All</button>
          <input type="range" min={panMin} max={panMax} step={0.001} value={panValue} onChange={handleScrollbarChange} style={{ flex: 1, marginLeft: 18, marginRight: 8, accentColor: "#00ffe7", height: 4, borderRadius: 4, background: "#10131a", boxShadow: "0 0 8px #00ffe7", outline: "none" }} />
        </div>
      </div>
      {/* External overlays: mini-map and VNA panels, stacked vertically */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", marginLeft: 24, marginTop: 8 }}>
        {renderMiniMapExternal()}
        {renderColorPresetPanel()}
        {renderTraceMathPanel()}
        {renderReferenceLinePanel()}
        {renderMarkerDeltaPanel()}
        {renderMarkerTablePanel()}
      </div>
    </div>
  );
} 