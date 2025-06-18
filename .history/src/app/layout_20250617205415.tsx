import type { Metadata } from "next";
import { Inter, Source_Code_Pro } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import AppLayout from "@/components/layout/AppLayout"; // Import the main layout
import { ClientBackground } from "./ClientBackground";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: 'swap',
});

const sourceCodePro = Source_Code_Pro({
  subsets: ["latin"],
  variable: "--font-source-code-pro",
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Quantum Weaver",
  description: "Quantum ZPE Network Analysis & Training Platform",
};

export function ClientBackground() {
  useEffect(() => {
    const canvas = document.getElementById('holo-bg-canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let animationFrameId: number;
    const w = canvas.width;
    const h = canvas.height;
    function drawGrid(time: number) {
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      // Parallax shimmer
      ctx.globalAlpha = 0.18 + 0.04 * Math.sin(time/1000);
      // Animated gridlines
      ctx.strokeStyle = '#00ffe7';
      ctx.lineWidth = 1.2;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x + (time/40)%40, 0);
        ctx.lineTo(x + (time/40)%40, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y + (time/60)%40);
        ctx.lineTo(w, y + (time/60)%40);
        ctx.stroke();
      }
      // Glowing nodes
      for (let x = 0; x < w; x += 160) {
        for (let y = 0; y < h; y += 160) {
          ctx.beginPath();
          ctx.arc(x + (time/40)%40, y + (time/60)%40, 4 + 2*Math.sin(time/500 + x + y), 0, 2*Math.PI);
          ctx.fillStyle = 'rgba(0,255,255,0.18)';
          ctx.shadowColor = '#00ffe7';
          ctx.shadowBlur = 16;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }
      ctx.restore();
      animationFrameId = requestAnimationFrame(drawGrid);
    }
    animationFrameId = requestAnimationFrame(drawGrid);
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  return (
    <div className="animated-holo-bg">
      <canvas id="holo-bg-canvas" width="1920" height="1080" style={{ width: '100vw', height: '100vh', display: 'block' }} />
    </div>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${sourceCodePro.variable} dark`} suppressHydrationWarning>
      <head />
      <body className="font-body antialiased retro-scanlines">
        <ClientBackground />
        <AppLayout>
          {children}
        </AppLayout>
        <Toaster />
      </body>
    </html>
  );
}