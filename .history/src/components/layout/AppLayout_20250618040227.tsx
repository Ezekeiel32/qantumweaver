"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
 PanelLeft, Home, Cpu, Zap, Atom, BarChart3, PlayCircle, Lightbulb, 
  Replace, Cog, Scaling, Box, Share2, Wrench, Moon, Sun, BrainCircuit, Globe, 
  ScatterChart, IterationCw, MessageSquare, Signal, SlidersHorizontal, Monitor, TrendingUp, Wand2, Rocket, ArrowRight, Settings2, ListChecks, Database
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/toaster";
import { cn } from "@/lib/utils";
import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarTrigger,
  SidebarInset,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  useSidebar, 
} from "@/components/ui/sidebar";
import FloatingMiniAdvisor from "@/components/FloatingMiniAdvisor";
import { AdvisorProvider } from "@/components/mini-hs-qnn-advisor";

const mainNavItems = [
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/workflow-builder", label: "Workflow Builder", icon: Wand2, priority: true },
  { href: "/data-portal", label: "Data Portal", icon: Database, priority: true },
  { href: "/train", label: "Train Model", icon: PlayCircle },
  { href: "/my-zpe-ai-models", label: "My ZPE-AI Models", icon: ListChecks },
  { href: "/performance", label: "Performance Analysis", icon: TrendingUp },
  { href: "/gpu-monitor", label: "GPU Monitor", icon: Monitor },
];

const advancedToolsNavItems = [
  { href: "/zpe-flow", label: "HS-QNN Advisor", icon: BrainCircuit },
  { href: "/zpe-flow-analysis", label: "ZPE Flow Analysis", icon: SlidersHorizontal },
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

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  priority?: boolean;
}

const NavLinkWrapper = ({
  item,
  pathname,
  onClick,
}: {
  item: NavItem;
  pathname: string;
  onClick: () => void;
}) => {
  const Icon = item.icon;
  const isActive =
    (pathname === item.href ||
      (pathname.startsWith(item.href + "/") &&
        item.href !== "/dashboard" &&
        item.href !== "/" &&
        item.href !== "/ai")) ||
    (pathname === "/" && item.href === "/introduction") ||
    (pathname === "/ai" && item.href === "/ai");

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        isActive={isActive}
        tooltip={{ children: item.label, side: "right", align: "center" }}
        className={cn(
          "justify-start transition-all duration-200",
          item.priority && "bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/20"
        )}
        onClick={onClick}
      >
        <Link href={item.href}>
          <Icon className="h-5 w-5" />
          <span className="group-data-[state=collapsed]:hidden">{item.label}</span>
          {item.priority && (
            <div className="ml-auto group-data-[state=collapsed]:hidden">
              <div className="w-2 h-2 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full animate-pulse" />
            </div>
          )}
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
};

const AppSidebarContent = () => {
  const pathname = usePathname();
  const { isMobile, setOpenMobile } = useSidebar(); 
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
      setIsDarkMode(storedTheme === "dark");
    } else {
      setIsDarkMode(window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
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

  const toggleDarkMode = () => setIsDarkMode(prev => !prev);

  const handleLinkClick = () => {
    if (isMobile) {
      setOpenMobile(false);
    }
  };
  
  const renderNavSection = (title: string, items: NavItem[]) => (
    <SidebarGroup>
      <SidebarGroupLabel className="text-white/70 font-semibold">{title}</SidebarGroupLabel>
      <SidebarMenu>
        {items.map((item) => (
          <NavLinkWrapper
            key={item.href}
            item={item}
            pathname={pathname}
            onClick={handleLinkClick}
          />
        ))}
      </SidebarMenu>
    </SidebarGroup>
  );

  return (
    <>
      <SidebarHeader className="border-b border-white/10 pb-4">
        <Link href="/dashboard" className="flex items-center gap-3 text-xl font-bold text-white group" onClick={handleLinkClick}>
          <div className="p-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg shadow-lg">
            <Zap className="h-6 w-6 text-white" />
          </div>
          <span className="group-data-[state=collapsed]:hidden bg-gradient-to-r from-white to-cyan-300 bg-clip-text text-transparent">
            Quantum Weaver
          </span>
        </Link>
      </SidebarHeader>
      <SidebarContent className="flex-1 overflow-y-auto p-0">
        <ScrollArea className="h-full px-2 py-0">
          {renderNavSection("Main", mainNavItems)}
          {renderNavSection("Advanced Tools", advancedToolsNavItems)}
          {renderNavSection("Visualizations", visNavItems)}
          {renderNavSection("AI Flows", aiFlowsNavItems)}
        </ScrollArea>
      </SidebarContent>
      <SidebarFooter className="border-t border-white/10 pt-4">
        <SidebarMenu>
          <SidebarMenuItem>
          <SidebarMenuButton variant="outline" onClick={toggleDarkMode} className="w-full justify-start text-sm border-white/20 text-white hover:bg-white/10">
              {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              <span className="group-data-[state=collapsed]:hidden">
                {isDarkMode ? "Light Mode" : "Dark Mode"}
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </>
  );
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
      setIsDarkMode(storedTheme === "dark");
    } else {
      setIsDarkMode(window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
    }
  }, []);

  const toggleDarkModeGlobal = () => {
    setIsDarkMode(prev => !prev);
  };

  return (
    <AdvisorProvider>
      <SidebarProvider>
        <div className="flex h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
          <Sidebar className="bg-white/5 backdrop-blur-xl border-r border-white/10">
            <AppSidebarContent />
          </Sidebar>
          <SidebarInset className="flex-1">
            <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
              <div className="flex items-center gap-2 px-4">
                <SidebarTrigger className="-ml-1" />
              </div>
              <div className="flex-1" />
            </header>
            <div className="flex flex-1 flex-col gap-4 p-4 pt-0 min-h-0">
              <div className="flex-1 min-h-0">
                {children}
              </div>
            </div>
          </SidebarInset>
        </div>
        <FloatingMiniAdvisor />
        <Toaster />
      </SidebarProvider>
    </AdvisorProvider>
  );
}
