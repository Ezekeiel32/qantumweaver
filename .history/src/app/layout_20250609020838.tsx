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
      <body className="font-body antialiased">
        <AppLayout>
          {children}
        </AppLayout>
        <Toaster />
      </body>
    </html>
  );
}
