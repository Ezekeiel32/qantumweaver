import React, { useRef, useEffect } from "react";

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
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    // Background
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);
    // Neon grid
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1.2;
    ctx.shadowColor = gridColor;
    ctx.shadowBlur = 8;
    for (let x = 0; x < width; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.restore();
    // Axes
    ctx.save();
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 2;
    ctx.shadowColor = gridColor;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(50, 10);
    ctx.lineTo(50, height - 30);
    ctx.lineTo(width - 10, height - 30);
    ctx.stroke();
    ctx.restore();
    // Axis labels (segment font)
    ctx.save();
    ctx.font = "16px 'Share Tech Mono', 'VT323', monospace";
    ctx.fillStyle = gridColor;
    ctx.globalAlpha = 0.8;
    for (let y = 0; y <= 1; y += 0.2) {
      const yy = height - 30 - y * (height - 40);
      ctx.fillText(((yMax ?? 1) - (yMax ?? 1 - y * ((yMax ?? 1) - (yMin ?? 0)))).toFixed(2), 8, yy + 4);
    }
    ctx.restore();
    // Traces
    traces.forEach((trace, tIdx) => {
      ctx.save();
      ctx.shadowColor = trace.color || defaultColors[tIdx % defaultColors.length];
      ctx.shadowBlur = 16;
      ctx.strokeStyle = trace.color || defaultColors[tIdx % defaultColors.length];
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const d = trace.data;
      const minY = yMin ?? Math.min(...d);
      const maxY = yMax ?? Math.max(...d);
      d.forEach((v, i) => {
        const x = 50 + (i / (d.length - 1)) * (width - 70);
        const y = height - 30 - ((v - minY) / (maxY - minY)) * (height - 40);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    });
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
    // Overlays (PASS, Center, Span, etc.)
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
  }, [traces, width, height, overlays, markers, gridColor, bgColor, showLegend, yMin, yMax]);
  return (
    <div style={{ position: "relative", width, height }}>
      <canvas ref={canvasRef} width={width} height={height} style={{ display: "block", width, height, background: bgColor, borderRadius: 12, boxShadow: "0 0 24px #00ffe7" }} />
    </div>
  );
} 