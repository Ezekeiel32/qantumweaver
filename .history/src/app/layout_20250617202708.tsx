import type { Metadata } from "next";
import { Inter, Source_Code_Pro } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import AppLayout from "@/components/layout/AppLayout"; // Import the main layout

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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${sourceCodePro.variable} dark`} suppressHydrationWarning>
      <head>
        {/* Using next/font, so direct Google Font links are not needed here. They are managed by the font objects. */}
      </head>
      <body className="font-body antialiased retro-scanlines">
        {/* Animated holographic SVG/canvas background */}
        <div className="animated-holo-bg">
          {/* Example SVG: Replace with animated SVG/canvas for holographic effect */}
          <svg width="100%" height="100%" viewBox="0 0 1920 1080" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="holoGradient" x1="0" y1="0" x2="1920" y2="1080" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00ffe7"/>
                <stop offset="0.5" stop-color="#ff00e6"/>
                <stop offset="1" stop-color="#00aaff"/>
              </linearGradient>
            </defs>
            <g opacity="0.5">
              <rect x="0" y="0" width="1920" height="1080" fill="url(#holoGradient)"/>
              <g stroke="#00ffe7" stroke-width="2" opacity="0.2">
                <rect x="100" y="100" width="1720" height="880" rx="40"/>
                <rect x="300" y="300" width="1320" height="480" rx="40"/>
              </g>
              <g stroke="#ff00e6" stroke-width="1.5" opacity="0.12">
                <circle cx="960" cy="540" r="400"/>
                <circle cx="960" cy="540" r="600"/>
              </g>
            </g>
          </svg>
        </div>
        <AppLayout>
          {children}
        </AppLayout>
        <Toaster />
      </body>
    </html>
  );
}
