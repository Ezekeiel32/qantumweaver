
"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  PanelLeft, Home, Cpu, Zap, Atom, BarChart3, PlayCircle, Lightbulb, 
  Replace, Cog, Scaling, Box, Share2, Wrench, Moon, Sun, BrainCircuit, Globe, 
  ScatterChart, IterationCw, MessageSquare, Signal, SlidersHorizontal, Monitor, TrendingUp, Wand2, Rocket, ArrowRight, Settings2, ListChecks
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/toaster";
import { cn } from "@/lib/utils";
import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent, // Corrected import name
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

const mainNavItems = [
  { href: "/introduction", label: "Introduction", icon: ArrowRight },
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/train", label: "Train Model", icon: PlayCircle },
  { href: "/my-zpe-ai-models", label: "My ZPE-AI Models", icon: ListChecks },
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

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
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
        className="justify-start"
        onClick={onClick}
      >
        <Link href={item.href}>
          <span className="flex items-center gap-2">
            <item.icon className="h-5 w-5" />
            <span className="group-data-[state=collapsed]:hidden">{item.label}</span>
          </span>
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
      <SidebarGroupLabel>{title}</SidebarGroupLabel>
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
      <SidebarHeader>
        <Link href="/dashboard" className="flex items-center gap-2 text-xl font-bold text-primary" onClick={handleLinkClick}>
          <Zap className="h-7 w-7" />
          <span className="group-data-[state=collapsed]:hidden">Quantum Weaver</span>
        </Link>
      </SidebarHeader>
      <SidebarContent className="flex-1 overflow-y-auto p-0"> {/* Corrected usage */}
        <ScrollArea className="h-full px-2 py-0">
          {renderNavSection("Main", mainNavItems)}
          {renderNavSection("Advanced Tools", advancedToolsNavItems)}
          {renderNavSection("Visualizations", visNavItems)}
          {renderNavSection("AI Flows", aiFlowsNavItems)}
        </ScrollArea>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton variant="ghost" onClick={toggleDarkMode} className="w-full justify-start text-sm">
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
  const { isMobile, setOpenMobile } = useSidebar(); // Get from context now
  const [isDarkMode, setIsDarkMode] = useState(true); // Independent theme state for header toggle

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
      setIsDarkMode(storedTheme === "dark");
    } else {
      setIsDarkMode(window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
    }
  }, []);
  
  const toggleDarkModeGlobal = () => {
     setIsDarkMode(prev => {
        const newMode = !prev;
        if (newMode) {
            document.documentElement.classList.add("dark");
            localStorage.setItem("theme", "dark");
        } else {
            document.documentElement.classList.remove("dark");
            localStorage.setItem("theme", "light");
        }
        return newMode;
     });
  };


  return (
    // SidebarProvider is now at the root of this component's return
    // We don't need to wrap it again here if this AppLayout is always under a SidebarProvider
    // However, if AppLayout IS the provider, then it's correct.
    // Assuming AppLayout IS the provider based on previous setup.
    <div className="flex min-h-screen w-full bg-background">
      <Sidebar
        side="left"
        collapsible="offcanvas" 
        variant="sidebar" 
        className="bg-sidebar text-sidebar-foreground" 
      >
        <AppSidebarContent />
      </Sidebar>

      <div className="flex flex-col flex-1 w-full"> 
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-card px-4 md:px-6">
          {/* SidebarTrigger for mobile is implicitly handled by Sidebar component when collapsible="offcanvas" */}
          {/* The custom SidebarTrigger here might be redundant if the default one from ui/sidebar is preferred */}
          {/* Let's use the SidebarTrigger from ui/sidebar, which gets its toggle function from context */}
           <SidebarTrigger className="" />

          <div className="flex-1 md:hidden"> 
            <Link href="/dashboard" className="flex items-center gap-2 text-lg font-semibold text-primary">
              <Zap className="h-6 w-6" />
              <span className="sr-only">Quantum Weaver</span>
            </Link>
          </div>
          <div className="ml-auto flex items-center gap-2">
             {/* Mobile theme toggle could be here if needed, but it's in the sidebar now for consistency */}
             {/* Adding a theme toggle for header on larger screens for convenience */}
             <Button variant="ghost" size="icon" onClick={toggleDarkModeGlobal} className="hidden md:inline-flex">
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                <span className="sr-only">Toggle theme</span>
            </Button>
            {/* Theme toggle for very small screens if sidebar is completely hidden */}
             <Button variant="ghost" size="icon" onClick={toggleDarkModeGlobal} className="md:hidden">
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </header>
        <SidebarInset> 
          {children}
        </SidebarInset>
      </div>
    </div>
  );
}
