
"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  PanelLeft, Home, Cpu, Zap, Atom, BarChart3, Settings, PlayCircle, Lightbulb, 
  Replace, Cog, Scaling, Box, Share2, Wrench, Moon, Sun, BrainCircuit, Globe, 
  ScatterChart, IterationCw, MessageSquare, Signal, SlidersHorizontal, Monitor, TrendingUp, Wand2, Rocket, ArrowRight, Settings2, ListChecks // Added ListChecks
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";

const mainNavItems = [
  { href: "/introduction", label: "Introduction", icon: ArrowRight },
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/train", label: "Train Model", icon: PlayCircle },
  { href: "/my-zpe-ai-models", label: "My ZPE-AI Models", icon: ListChecks }, // Updated Link
  { href: "/performance", label: "Performance Analysis", icon: TrendingUp },
  { href: "/architecture", label: "Architecture", icon: Cpu },
  { href: "/gpu-monitor", label: "GPU Monitor", icon: Monitor },
];

const advancedToolsNavItems = [
  { href: "/zpe-flow-analysis", label: "ZPE Flow Analysis", icon: SlidersHorizontal },
  { href: "/zpe-flow", label: "HS-QNN Advisor", icon: BrainCircuit }, 
  { href: "/quantum-noise", label: "Quantum Noise", icon: Atom },
  { href: "/rf-generator", label: "RF Generator", icon: Signal },
  { href: "/ai-analysis", label: "AI Assistant", icon: MessageSquare },
  { href: "/cloud-gpu", label: "Cloud GPU Connect", icon: Settings2 },
];

const visNavItems = [
  { href: "/vis/bloch-sphere", label: "Bloch Sphere", icon: Globe },
  { href: "/vis/dynamic-formation", label: "Dynamic Formation", icon: IterationCw }, 
  { href: "/vis/zpe-particle-simulation", label: "ZPE Particle Sim", icon: ScatterChart },
];

const aiFlowsNavItems = [
  { href: "/ai", label: "AI Flows Hub", icon: Rocket },
  { href: "/ai/implement-zpe", label: "Simulate ZPE", icon: Lightbulb },
  { href: "/ai/approximate-zpe", label: "Approximate Flow", icon: Replace },
  { href: "/ai/adapt-zpe", label: "Adapt ZPE", icon: Cog },
  { href: "/ai/show-scaled-output", label: "Scaled Output", icon: Scaling },
  { href: "/ai/quantize-model", label: "Quantize Model", icon: Box },
  { href: "/ai/extract-components", label: "Extract Components", icon: Share2 },
  { href: "/ai/configure-model", label: "Configure Model", icon: Wrench },
  { href: "/ai/invoke-llm", label: "Generic LLM", icon: Wand2 },
];


export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true); 

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
      setIsDarkMode(storedTheme === "dark");
    } else {
      setIsDarkMode(window.matchMedia("(prefers-color-scheme: dark)").matches);
    }
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);
  
  const NavLinkContent = ({ item }: { item: { label: string, icon: React.ElementType } }) => (
    <>
      <item.icon className="h-5 w-5 mr-3 flex-shrink-0" />
      {item.label}
    </>
  );

  const renderNavSection = (title: string, items: Array<{ href: string; label: string; icon: React.ElementType }>) => (
    <div className="mb-2">
      <h3 className="px-3 py-2 text-xs font-semibold uppercase text-sidebar-foreground/70 tracking-wider">{title}</h3>
      {items.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className={cn(
            "flex items-center px-3 py-2.5 rounded-md text-sm font-medium transition-colors duration-150 ease-in-out",
            (pathname === item.href || (pathname.startsWith(item.href + '/') && item.href !== "/dashboard" && item.href !== "/" && item.href !== "/ai")) || 
             (pathname === "/" && item.href === "/introduction") || 
             (pathname === "/ai" && item.href === "/ai")
              ? "bg-sidebar-primary text-sidebar-primary-foreground"
              : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
          )}
          onClick={() => setIsMobileNavOpen(false)}
        >
          <NavLinkContent item={item} />
        </Link>
      ))}
    </div>
  );

  const SidebarContent = () => (
    <div className="flex flex-col h-full bg-sidebar text-sidebar-foreground">
      <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
        <Link href="/dashboard" className="flex items-center gap-2 text-xl font-bold text-primary" onClick={() => setIsMobileNavOpen(false)}>
          <Zap className="h-7 w-7" />
          <span>Quantum Weaver</span>
        </Link>
      </div>
      <ScrollArea className="flex-1">
        <nav className="px-2 py-4 space-y-0">
          {renderNavSection("Main", mainNavItems)}
          {renderNavSection("Advanced Tools", advancedToolsNavItems)}
          {renderNavSection("Visualizations", visNavItems)}
          {renderNavSection("AI Flows", aiFlowsNavItems)}
        </nav>
      </ScrollArea>
      <div className="p-4 border-t border-sidebar-border mt-auto">
        <Button variant="outline" className="w-full justify-start bg-sidebar-accent text-sidebar-accent-foreground hover:bg-sidebar-accent/80 border-sidebar-border" onClick={toggleDarkMode}>
          {isDarkMode ? <Sun className="mr-2 h-4 w-4" /> : <Moon className="mr-2 h-4 w-4" />}
          {isDarkMode ? "Light Mode" : "Dark Mode"}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen w-full bg-background">
      <aside className="hidden md:flex md:flex-col md:w-72 border-r border-border fixed inset-y-0 left-0 z-40">
        <SidebarContent />
      </aside>
      <div className="flex flex-col flex-1 w-full md:pl-72">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-card px-4 md:px-6 md:hidden">
           <Sheet open={isMobileNavOpen} onOpenChange={setIsMobileNavOpen}>
            <SheetTrigger asChild>
              <Button variant="outline" size="icon" className="shrink-0">
                <PanelLeft className="h-5 w-5" />
                <span className="sr-only">Toggle navigation menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="flex flex-col p-0 w-72 bg-sidebar text-sidebar-foreground">
              <SidebarContent />
            </SheetContent>
          </Sheet>
          <Link href="/dashboard" className="flex items-center gap-2 text-lg font-semibold text-primary md:hidden" onClick={() => setIsMobileNavOpen(false)}>
            <Zap className="h-6 w-6" />
            <span className="sr-only">Quantum Weaver</span>
          </Link>
          <div className="ml-auto">
             <Button variant="ghost" size="icon" onClick={toggleDarkMode} className="md:hidden">
              {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}

    
      