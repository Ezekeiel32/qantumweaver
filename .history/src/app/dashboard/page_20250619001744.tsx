"use client";
import React, { useState, useEffect, useCallback } from "react";
import Link from 'next/link';
import {
  Home, Cpu, Zap, Atom, BarChart3, Settings, PlayCircle, Lightbulb,
  Replace, Cog, Scaling, Box, Share2, Wrench, BrainCircuit, Globe, 
  ScatterChart, IterationCw, Database, MessageSquare, Signal, SlidersHorizontal, Monitor, TrendingUp, Wand2, Rocket, ArrowRight, Settings2
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const appFeatures = [
  { href: "/train", label: "Train Model", icon: PlayCircle, description: "Configure and initiate ZPE model training jobs." },
  { href: "/model-configs", label: "Model Configs", icon: Database, description: "Manage and compare different model configurations." },
  { href: "/configurations", label: "Job History", icon: BarChart3, description: "Review parameters and outcomes of past training." }, 
  { href: "/performance", label: "Performance Analysis", icon: TrendingUp, description: "Analyze training metrics and model health." },
  { href: "/architecture", label: "Model Architecture", icon: Cpu, description: "Explore the ZPE Quantum Neural Network structure." },
  { href: "/gpu-monitor", label: "GPU Monitor", icon: Monitor, description: "Live statistics for your primary GPU resources." },
  { href: "/zpe-flow-analysis", label: "ZPE Flow Analysis", icon: SlidersHorizontal, description: "Interactively analyze ZPE flow parameters." },
  { href: "/zpe-flow", label: "HS-QNN Advisor", icon: BrainCircuit, description: "AI advice for Hilbert Space QNN parameters." },
  { href: "/quantum-noise", label: "Quantum Noise", icon: Atom, description: "Generate and analyze quantum-derived noise patterns." },
  { href: "/rf-generator", label: "RF Generator", icon: Signal, description: "Configure RF waves for conceptual network influence." },
  { href: "/ai-analysis", label: "AI Assistant", icon: MessageSquare, description: "Chat with an AI for insights and optimization." },
  { href: "/ai", label: "AI Flows Hub", icon: Rocket, description: "Access all specialized GenAI-powered tools." },
  { href: "/vis/bloch-sphere", label: "Bloch Sphere", icon: Globe, description: "Visualize qubit states in Hilbert space." },
  { href: "/vis/dynamic-formation", label: "Dynamic Formation", icon: IterationCw, description: "Conceptual visualization of dynamic particle formations." },
  { href: "/vis/zpe-particle-simulation", label: "ZPE Particle Sim", icon: ScatterChart, description: "Interactive 3D ZPE particle system." },
  { href: "/cloud-gpu", label: "Cloud GPU Connect", icon: Settings2, description: "Connect to third-party cloud GPU providers." },
];

// Minimal type for job
interface Job {
  job_id: string;
  model_name?: string;
  status?: string;
  accuracy?: number;
  zpe_effects?: number[];
  parameters?: {
    modelName?: string;
    quantumMode?: boolean;
    keyFeature?: string;
  };
}

export default function DashboardPage() {
  const [latestJob, setLatestJob] = useState<Job | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchLatestJob = useCallback(async () => {
    setIsLoading(true);
    try {
      const jobsRes = await fetch("/api/jobs?limit=10");
      const jobsData = await jobsRes.json();
      const jobs: Job[] = (jobsData.jobs || []).filter((job: Job) => ["completed", "running", "stopped"].includes(job.status || ""));
      if (jobs.length > 0) {
        const latest = jobs[0];
        const detailRes = await fetch(`/api/status/${latest.job_id}`);
        const detail: Job = await detailRes.json();
        setLatestJob({ ...latest, ...detail });
      } else {
        setLatestJob(null);
      }
    } catch (e) {
      setLatestJob(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLatestJob();
  }, [fetchLatestJob]);

  // Fallback if no job
  const latestModelData = latestJob ? {
    name: latestJob.parameters?.modelName || latestJob.model_name || "-",
    accuracy: latestJob.accuracy || 0,
    zpeLayers: Array.isArray(latestJob.zpe_effects) ? latestJob.zpe_effects.length : 6,
    avgZPEEffect: Array.isArray(latestJob.zpe_effects) && latestJob.zpe_effects.length > 0 ? (latestJob.zpe_effects.reduce((a: number, b: number) => a + b, 0) / latestJob.zpe_effects.length) : 0,
    quantumMode: latestJob.parameters?.quantumMode ?? false,
    keyFeature: latestJob.parameters?.keyFeature || "-"
  } : {
    name: "-",
    accuracy: 0,
    zpeLayers: 6, 
    avgZPEEffect: 0,
    quantumMode: false,
    keyFeature: "-"
  };

  return (
    <div className="p-1 md:p-2 space-y-6">
      <div className="matrix-panel">
      <Card className="shadow-lg border-primary/20 bg-card/80 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <Zap className="h-10 w-10 text-primary animate-pulse" />
            <div>
              <CardTitle className="text-3xl font-headline tracking-tight text-primary">
                Quantum Weaver Control Panel
              </CardTitle>
              <CardDescription className="text-lg text-muted-foreground">
                Harness Zero-Point Energy for advanced AI model development.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted/50 p-6 rounded-lg border border-border/50">
            <h3 className="text-xl font-semibold mb-3 flex items-center gap-2 text-accent-foreground"><Atom className="h-6 w-6 text-accent"/>Welcome to the TetraZPE Platform!</h3>
            <p className="text-sm text-foreground/80 leading-relaxed">
              Navigate through specialized tools to configure, train, analyze, and visualize your ZPE-enhanced neural networks.
              Utilize AI-powered assistants and explore quantum-inspired simulations.
            </p>
          </div>

          <Card className="bg-gradient-to-br from-card to-primary/5 border-primary/20">
            <CardHeader>
                <CardTitle className="text-xl font-headline flex items-center gap-2"><TrendingUp className="h-5 w-5 text-accent"/>Latest Model Overview</CardTitle>
                <CardDescription>Key statistics from the most recent real job run.</CardDescription>
            </CardHeader>
            <CardContent className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
              <div className="bg-background/70 p-3 rounded-md flex justify-between items-center ring-1 ring-inset ring-border">
                <span className="text-muted-foreground">Model Name:</span>
                <Badge variant="secondary" className="font-code">{latestModelData.name}</Badge>
              </div>
              <div className="bg-background/70 p-3 rounded-md flex justify-between items-center ring-1 ring-inset ring-border">
                  <span className="text-muted-foreground">Accuracy:</span>
                <Badge className="bg-green-500/20 text-green-400 border-green-600/50 font-semibold">{latestModelData.accuracy}%</Badge>
              </div>
              <div className="bg-background/70 p-3 rounded-md flex justify-between items-center ring-1 ring-inset ring-border">
                <span className="text-muted-foreground">Avg. ZPE Effect:</span>
                <span className="font-semibold font-code">{latestModelData.avgZPEEffect.toFixed(3)}</span>
              </div>
              <div className="bg-background/70 p-3 rounded-md flex justify-between items-center ring-1 ring-inset ring-border">
                <span className="text-muted-foreground">Quantum Mode:</span>
                {latestModelData.quantumMode ? (
                  <Badge className="bg-purple-500/20 text-purple-400 border-purple-600/50">Enabled</Badge>
                ) : (
                  <Badge variant="outline">Disabled</Badge>
                )}
              </div>
              <div className="bg-background/70 p-3 rounded-md md:col-span-2 lg:col-span-2 ring-1 ring-inset ring-border">
                <span className="text-muted-foreground">Key Feature: </span>
                <span className="font-medium">{latestModelData.keyFeature}</span>
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
      </div>

      <div>
        <h2 className="text-2xl font-headline mb-6 tracking-tight flex items-center gap-2">
          <Wrench className="h-6 w-6 text-primary"/>Application Tools & Features
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {appFeatures.map((feature) => (
            <Card key={feature.href} className="flex flex-col bg-card/90 hover:shadow-xl hover:border-primary/50 transition-all duration-200 ease-in-out transform hover:-translate-y-1 backdrop-blur-sm">
              <CardHeader className="flex-row items-start gap-4 space-y-0 pb-3">
                <div className="p-3 bg-primary/10 rounded-lg">
                  <feature.icon className="h-7 w-7 text-primary" />
                </div>
                <div className="flex-1">
                  <CardTitle className="text-lg font-headline mb-1">{feature.label}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex-1 mb-2">
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </CardContent>
              <CardFooter>
                <Button asChild variant="outline" className="w-full border-primary/30 hover:bg-primary/10 hover:text-primary group">
                  <Link href={feature.href}>
                    Open Tool
                    <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform"/>
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
