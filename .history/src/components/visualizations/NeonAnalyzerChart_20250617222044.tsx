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
}

const defaultColors = ["#ffe066", "#ff00ff", "#00ffe7", "#39ff14", "#ffffff"];

export default function NeonAnalyzerChart({
  traces,
  width = 600,
  height = 300,
  overlays = [],
  markers = [],
  gridColor = "#00ffe7",
  bgColor = "#10131a",
  showLegend = true,
  yMin,
  yMax,
}: NeonAnalyzerChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState(0); // 0 = left, 1 = right
  const [isPanning, setIsPanning] = useState(false);
  const [panStartX, setPanStartX] = useState<number | null>(null);

  // --- Afterglow buffer ---
  const afterglowRef = useRef<HTMLCanvasElement>(null);

  // --- Mouse/touch pan handlers ---
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    setPanStartX(e.clientX);
  };
  const handleMouseUp = () => setIsPanning(false);
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning || panStartX === null) return;
    const dx = e.clientX - panStartX;
    setPan(prev => Math.max(0, Math.min(1, prev - dx / width / zoom)));
    setPanStartX(e.clientX);
  };
  // --- Zoom controls ---
  const handleZoomIn = () => setZoom(z => Math.min(8, z * 2));
  const handleZoomOut = () => setZoom(z => Math.max(1, z / 2));

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
    // Neon grid
    glowCtx.save();
    glowCtx.globalAlpha = 0.18;
    glowCtx.strokeStyle = gridColor;
    glowCtx.lineWidth = 1.2;
    glowCtx.shadowColor = gridColor;
    glowCtx.shadowBlur = 8;
    for (let x = 0; x < width; x += 60) {
      glowCtx.beginPath();
      glowCtx.moveTo(x, 0);
      glowCtx.lineTo(x, height);
      glowCtx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      glowCtx.beginPath();
      glowCtx.moveTo(0, y);
      glowCtx.lineTo(width, y);
      glowCtx.stroke();
    }
    glowCtx.restore();
    // Axes
    glowCtx.save();
    glowCtx.strokeStyle = gridColor;
    glowCtx.lineWidth = 2;
    glowCtx.shadowColor = gridColor;
    glowCtx.shadowBlur = 12;
    glowCtx.beginPath();
    glowCtx.moveTo(50, 10);
    glowCtx.lineTo(50, height - 30);
    glowCtx.lineTo(width - 10, height - 30);
    glowCtx.stroke();
    glowCtx.restore();
    // Axis labels
    glowCtx.save();
    glowCtx.font = "16px 'Share Tech Mono', 'VT323', monospace";
    glowCtx.fillStyle = gridColor;
    glowCtx.globalAlpha = 0.8;
    for (let y = 0; y <= 1; y += 0.2) {
      const yy = height - 30 - y * (height - 40);
      glowCtx.fillText(((yMax ?? 1) - (yMax ?? 1 - y * ((yMax ?? 1) - (yMin ?? 0)))).toFixed(2), 8, yy + 4);
    }
    glowCtx.restore();
    // --- Traces (with zoom/pan and afterglow) ---
    traces.forEach((trace, tIdx) => {
      glowCtx.save();
      glowCtx.shadowColor = trace.color || defaultColors[tIdx % defaultColors.length];
      glowCtx.shadowBlur = 24;
      glowCtx.strokeStyle = trace.color || defaultColors[tIdx % defaultColors.length];
      glowCtx.lineWidth = 3.5;
      glowCtx.beginPath();
      const d = trace.data;
      const minY = yMin ?? Math.min(...d);
      const maxY = yMax ?? Math.max(...d);
      const n = d.length;
      // Zoom/pan window
      const windowSize = Math.floor(n / zoom);
      const startIdx = Math.floor((n - windowSize) * pan);
      const endIdx = Math.min(n, startIdx + windowSize);
      // Cubic bezier for smooth RF look
      let prevX = 0, prevY = 0;
      for (let i = startIdx; i < endIdx; i++) {
        const x = 50 + ((i - startIdx) / (windowSize - 1)) * (width - 70);
        const y = height - 30 - ((d[i] - minY) / (maxY - minY)) * (height - 40);
        if (i === startIdx) {
          glowCtx.moveTo(x, y);
        } else {
          // Cubic bezier: control points are halfway between previous and current
          const cx1 = prevX + (x - prevX) / 2;
          const cy1 = prevY;
          const cx2 = x - (x - prevX) / 2;
          const cy2 = y;
          glowCtx.bezierCurveTo(cx1, cy1, cx2, cy2, x, y);
        }
        prevX = x;
        prevY = y;
      }
      glowCtx.stroke();
      glowCtx.restore();
    });
    // Copy afterglow buffer to main canvas
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(afterglow, 0, 0, width, height);
    // Markers
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
    // Overlays
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
    // Legend
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
  }, [traces, width, height, overlays, markers, gridColor, bgColor, showLegend, yMin, yMax, zoom, pan]);

  return (
    <div style={{ position: "relative", width, height }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
    >
      <canvas ref={afterglowRef} width={width} height={height} style={{ display: "none" }} />
      <canvas ref={canvasRef} width={width} height={height} style={{ display: "block", width, height, background: bgColor, borderRadius: 12, boxShadow: "0 0 24px #00ffe7", cursor: zoom > 1 ? "grab" : "default" }} />
      <div style={{ position: "absolute", top: 10, right: 16, zIndex: 2, display: "flex", flexDirection: "column", gap: 4 }}>
        <button onClick={handleZoomIn} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, marginBottom: 2, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>+</button>
        <button onClick={handleZoomOut} style={{ background: "#10131a", color: "#00ffe7", border: "1px solid #00ffe7", borderRadius: 6, width: 28, height: 28, fontSize: 20, cursor: "pointer", boxShadow: "0 0 8px #00ffe7" }}>-</button>
      </div>
    </div>
  );
} 