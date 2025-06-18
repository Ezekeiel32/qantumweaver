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
          <rect x={(startIdx / n) * 160} y={2} width={(windowSize / n) * 160} height={32} fill="#00ffe7" opacity={0.10} stroke="#00ffe7" strokeWidth={1.2} />
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

  useEffect(() => {
    const canvas = canvasRef.current;
    const afterglow = afterglowRef.current;
    if (!canvas || !afterglow) return;
    const ctx = canvas.getContext("2d");
    const glowCtx = afterglow.getContext("2d");
    if (!ctx || !glowCtx) return;
    // --- Afterglow: fade previous frame ---
    glowCtx.globalAlpha = 0.7;
    glowCtx.drawImage(afterglow, 0, 0, width, height);
    glowCtx.globalAlpha = 1.0;
    // --- Draw new frame on afterglow buffer ---
    glowCtx.clearRect(0, 0, width, height);
    // Background
    glowCtx.fillStyle = bgColor;
    glowCtx.fillRect(0, 0, width, height);
    // Enhanced Neon grid (major/minor lines)
    glowCtx.save();
    for (let x = 0; x < width; x += 60) {
      glowCtx.globalAlpha = x % 120 === 0 ? 0.28 : 0.12;
      glowCtx.strokeStyle = gridColor;
      glowCtx.lineWidth = x % 120 === 0 ? 2 : 1;
      glowCtx.shadowColor = gridColor;
      glowCtx.shadowBlur = x % 120 === 0 ? 12 : 6;
      glowCtx.beginPath();
      glowCtx.moveTo(x, 0);
      glowCtx.lineTo(x, height);
      glowCtx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      glowCtx.globalAlpha = y % 80 === 0 ? 0.28 : 0.12;
      glowCtx.strokeStyle = gridColor;
      glowCtx.lineWidth = y % 80 === 0 ? 2 : 1;
      glowCtx.shadowColor = gridColor;
      glowCtx.shadowBlur = y % 80 === 0 ? 12 : 6;
      glowCtx.beginPath();
      glowCtx.moveTo(0, y);
      glowCtx.lineTo(width, y);
      glowCtx.stroke();
    }
    glowCtx.restore();
    // Axes
    glowCtx.save();
    glowCtx.strokeStyle = gridColor;
    glowCtx.lineWidth = 2.5;
    glowCtx.shadowColor = gridColor;
    glowCtx.shadowBlur = 16;
    glowCtx.beginPath();
    glowCtx.moveTo(50, 10);
    glowCtx.lineTo(50, height - 30);
    glowCtx.lineTo(width - 10, height - 30);
    glowCtx.stroke();
    glowCtx.restore();
    // Axis labels (segment font)
    glowCtx.save();
    glowCtx.font = "16px 'Share Tech Mono', 'VT323', monospace";
    glowCtx.fillStyle = gridColor;
    glowCtx.globalAlpha = 0.8;
    for (let y = 0; y <= 1; y += 0.2) {
      const yy = height - 30 - y * (height - 40);
      glowCtx.fillText(((yMax ?? 1) - (yMax ?? 1 - y * ((yMax ?? 1) - (yMin ?? 0)))).toFixed(2), 8, yy + 4);
    }
    glowCtx.restore();
    // --- Traces (with zoom/pan and afterglow, using Catmull-Rom) ---
    traces.forEach((trace, tIdx) => {
      glowCtx.save();
      glowCtx.shadowColor = trace.color || defaultColors[tIdx % defaultColors.length];
      glowCtx.shadowBlur = 12;
      glowCtx.strokeStyle = trace.color || defaultColors[tIdx % defaultColors.length];
      // Fixed, crisp line width for VNA look
      const lineWidth = 2;
      glowCtx.lineWidth = lineWidth;
      glowCtx.imageSmoothingEnabled = true;
      const d = trace.data;
      const minY = yMin ?? Math.min(...d);
      const maxY = yMax ?? Math.max(...d);
      const n = d.length;
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      const endIdx = Math.min(n, startIdx + windowSize);
      // Prepare points for spline
      const points: [number, number][] = [];
      for (let i = startIdx; i < endIdx; i++) {
        const x = 50 + ((i - startIdx) / (windowSize - 1)) * (width - 70);
        const y = height - 30 - ((d[i] - minY) / (maxY - minY)) * (height - 40);
        points.push([x, y]);
      }
      // Only draw if we have at least 2 points
      if (points.length >= 2) {
        const spline = catmullRomSpline(points, 8);
        glowCtx.beginPath();
        spline.forEach(([x, y], i) => {
          if (i === 0) glowCtx.moveTo(x, y);
          else glowCtx.lineTo(x, y);
        });
        glowCtx.stroke();
      }
      glowCtx.restore();
      // --- Per-epoch markers (fixed, visible size) ---
      glowCtx.save();
      glowCtx.globalAlpha = 0.8;
      const markerRadius = 5;
      for (let i = startIdx; i < endIdx; i++) {
        const x = 50 + ((i - startIdx) / (windowSize - 1)) * (width - 70);
        const y = height - 30 - ((d[i] - minY) / (maxY - minY)) * (height - 40);
        if (i % Math.ceil(windowSize / 12) === 0) {
          glowCtx.beginPath();
          glowCtx.arc(x, y, markerRadius, 0, 2 * Math.PI);
          glowCtx.fillStyle = trace.color || defaultColors[tIdx % defaultColors.length];
          glowCtx.shadowColor = trace.color || defaultColors[tIdx % defaultColors.length];
          glowCtx.shadowBlur = 8;
          glowCtx.fill();
          // Epoch label
          glowCtx.save();
          glowCtx.font = "12px 'Share Tech Mono', 'VT323', monospace";
          glowCtx.fillStyle = '#39ff14';
          glowCtx.shadowColor = '#39ff14';
          glowCtx.shadowBlur = 6;
          glowCtx.globalAlpha = 0.7;
          glowCtx.fillText(`E${i + 1}`, x - 10, height - 10);
          glowCtx.restore();
        }
      }
      glowCtx.restore();
    });
    // Copy afterglow buffer to main canvas
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(afterglow, 0, 0, width, height);
    // --- Measurement markers and delta ---
    if (measureMarkers.length > 0) {
      measureMarkers.forEach((m, idx) => {
        ctx.save();
        const x = 50 + ((m.x - Math.floor((traces[0].data.length - Math.floor(traces[0].data.length / zoom)) * pan)) / (Math.floor(traces[0].data.length / zoom) - 1)) * (width - 70);
        const minY = yMin ?? Math.min(...traces[0].data);
        const maxY = yMax ?? Math.max(...traces[0].data);
        const y = height - 30 - ((m.y - minY) / (maxY - minY)) * (height - 40);
        // Neon box
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, 2 * Math.PI);
        ctx.fillStyle = idx === 0 ? "#ffe066" : "#ff00ff";
        ctx.shadowColor = idx === 0 ? "#ffe066" : "#ff00ff";
        ctx.shadowBlur = 18;
        ctx.fill();
        // Box with value
        ctx.font = "bold 14px 'Share Tech Mono', 'VT323', monospace";
        ctx.fillStyle = "#10131a";
        ctx.shadowBlur = 0;
        ctx.fillRect(x + 12, y - 18, 70, 28);
        ctx.strokeStyle = idx === 0 ? "#ffe066" : "#ff00ff";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 12, y - 18, 70, 28);
        ctx.fillStyle = idx === 0 ? "#ffe066" : "#ff00ff";
        ctx.fillText(`M${idx + 1}: E${m.x + 1}`, x + 18, y - 2);
        ctx.fillText(m.y.toFixed(4), x + 18, y + 12);
        ctx.restore();
      });
      // Delta marker if two points
      if (measureMarkers.length === 2) {
        const [m1, m2] = measureMarkers;
        ctx.save();
        const n = traces[0].data.length;
        const windowSize = Math.floor(n / zoom);
        const startIdx = Math.floor((n - windowSize) * pan);
        const x1 = 50 + ((m1.x - startIdx) / (windowSize - 1)) * (width - 70);
        const x2 = 50 + ((m2.x - startIdx) / (windowSize - 1)) * (width - 70);
        const minY = yMin ?? Math.min(...traces[0].data);
        const maxY = yMax ?? Math.max(...traces[0].data);
        const y1 = height - 30 - ((m1.y - minY) / (maxY - minY)) * (height - 40);
        const y2 = height - 30 - ((m2.y - minY) / (maxY - minY)) * (height - 40);
        // Neon line between markers
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "#39ff14";
        ctx.lineWidth = 3;
        ctx.shadowColor = "#39ff14";
        ctx.shadowBlur = 16;
        ctx.stroke();
        // Delta box
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        ctx.beginPath();
        ctx.fillStyle = "#10131a";
        ctx.shadowBlur = 0;
        ctx.fillRect(midX - 40, midY - 32, 80, 32);
        ctx.strokeStyle = "#39ff14";
        ctx.lineWidth = 2;
        ctx.strokeRect(midX - 40, midY - 32, 80, 32);
        ctx.font = "bold 14px 'Share Tech Mono', 'VT323', monospace";
        ctx.fillStyle = "#39ff14";
        ctx.fillText(`ΔE: ${Math.abs(m2.x - m1.x)}`, midX - 32, midY - 16);
        ctx.fillText(`ΔY: ${(m2.y - m1.y).toFixed(4)}`, midX - 32, midY - 2);
        ctx.restore();
      }
    }
    // Markers (existing)
    markers.forEach((m, i) => {
      ctx.save();
      ctx.beginPath();
      const x = 50 + (m.x / (traces[0].data.length - 1)) * (width - 70);
      const minY = yMin ?? Math.min(...traces[0].data);
      const maxY = yMax ?? Math.max(...traces[0].data);
      const y = height - 30 - ((m.y - minY) / (maxY - minY)) * (height - 40);
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = m.color || "#ffe066";
      ctx.shadowColor = m.color || "#ffe066";
      ctx.shadowBlur = 16;
      ctx.fill();
      ctx.font = "14px 'Share Tech Mono', 'VT323', monospace";
      ctx.fillStyle = m.color || "#ffe066";
      ctx.shadowBlur = 0;
      if (m.label) ctx.fillText(m.label, x + 10, y - 10);
      ctx.restore();
    });
    // Overlays (existing)
    overlays.forEach(o => {
      ctx.save();
      ctx.font = `${o.fontSize || 24}px 'Share Tech Mono', 'VT323', monospace`;
      ctx.fillStyle = o.color || "#39ff14";
      ctx.shadowColor = o.color || "#39ff14";
      ctx.shadowBlur = 16;
      ctx.globalAlpha = 0.9;
      ctx.fillText(o.text, o.x ?? 80, o.y ?? 40);
      ctx.restore();
    });
    // Legend (existing)
    if (showLegend && traces.some(t => t.label)) {
      ctx.save();
      traces.forEach((t, i) => {
        ctx.font = "16px 'Share Tech Mono', 'VT323', monospace";
        ctx.fillStyle = t.color || defaultColors[i % defaultColors.length];
        ctx.shadowColor = t.color || defaultColors[i % defaultColors.length];
        ctx.shadowBlur = 8;
        ctx.fillText(t.label || `Trace ${i + 1}`, width - 120, 30 + i * 22);
      });
      ctx.restore();
    }
    // Tooltip for hovered epoch
    if (hoveredEpoch !== null && traces[0]) {
      const d = traces[0].data;
      const n = d.length;
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      const x = 50 + ((hoveredEpoch - startIdx) / (windowSize - 1)) * (width - 70);
      const minY = yMin ?? Math.min(...d);
      const maxY = yMax ?? Math.max(...d);
      const y = height - 30 - ((d[hoveredEpoch] - minY) / (maxY - minY)) * (height - 40);
      if (d[hoveredEpoch] !== undefined) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = "#39ff14";
        ctx.shadowColor = "#39ff14";
        ctx.shadowBlur = 16;
        ctx.fill();
        ctx.font = "bold 13px 'Share Tech Mono', 'VT323', monospace";
        ctx.fillStyle = "#10131a";
        ctx.shadowBlur = 0;
        ctx.fillRect(x + 12, y - 18, 70, 28);
        ctx.strokeStyle = "#39ff14";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 12, y - 18, 70, 28);
        ctx.fillStyle = "#39ff14";
        ctx.fillText(`E${hoveredEpoch + 1}`, x + 18, y - 2);
        ctx.fillText(d[hoveredEpoch].toFixed(4), x + 18, y + 12);
        ctx.restore();
      }
    }
    // --- Reference lines ---
    if (traces[0]) {
      const d = traces[0].data;
      const minY = yMin ?? Math.min(...d);
      const maxY = yMax ?? Math.max(...d);
      renderReferenceLines(ctx, minY, maxY);
    }
  }, [traces, width, height, overlays, markers, gridColor, bgColor, showLegend, yMin, yMax, zoom, pan, measureMarkers, hoveredEpoch, referenceLines]);

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
          <canvas ref={afterglowRef} width={width} height={height} style={{ display: "none" }} />
          <canvas ref={canvasRef} width={width} height={height} style={{ display: "block", width, height, background: bgColor, borderRadius: 12, boxShadow: "0 0 24px #00ffe7", cursor: zoom > 1 ? "grab" : "default" }} />
          <div style={{ position: "absolute", top: 10, right: 16, zIndex: 2, display: "flex", flexDirection: "column", gap: 4 }}>
            <button onClick={handleZoomIn} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, marginBottom: 2, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>+</button>
            <button onClick={handleZoomOut} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>-</button>
          </div>
        </div>
      </div>
      {/* External overlays: mini-map and marker/delta panel, stacked vertically */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", marginLeft: 24, marginTop: 8 }}>
        {renderMiniMapExternal()}
        {renderMarkerDeltaPanel()}
      </div>
    </div>
  );
} 