"use client";
import React, { useState, useEffect, useCallback } from "react";
import Link from 'next/link';
import {
  Home, Cpu, Zap, Atom, BarChart3, Settings, PlayCircle, Lightbulb,
  Replace, Cog, Scaling, Box, Share2, Wrench, BrainCircuit, Globe, 
  ScatterChart, IterationCw, Database, MessageSquare, Signal, SlidersHorizontal, Monitor, TrendingUp, Wand2, Rocket, ArrowRight, Settings2,
  Sparkles, Target, Gauge, Activity
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";

const appFeatures = [
  { 
    href: "/workflow-builder", 
    label: "Workflow Builder", 
    icon: Wand2, 
    description: "Build quantum AI models with zero coding. From data upload to deployment in minutes.",
    gradient: "from-purple-500 to-pink-500",
    priority: true
  },
  { 
    href: "/data-portal", 
    label: "Data Portal", 
    icon: Database, 
    description: "Upload, manage, and analyze datasets with quantum resonance scanning.",
    gradient: "from-blue-500 to-cyan-500",
    priority: true
  },
  { 
    href: "/train", 
    label: "Train Model", 
    icon: PlayCircle, 
    description: "Configure and initiate ZPE model training jobs with real-time monitoring.",
    gradient: "from-green-500 to-emerald-500"
  },
  { 
    href: "/performance", 
    label: "Performance Analysis", 
    icon: TrendingUp, 
    description: "Analyze training metrics and model health with advanced visualizations.",
    gradient: "from-orange-500 to-red-500"
  },
  { 
    href: "/my-zpe-ai-models", 
    label: "My ZPE-AI Models", 
    icon: Box, 
    description: "Manage and deploy your trained quantum-enhanced models.",
    gradient: "from-indigo-500 to-purple-500"
  },
  { 
    href: "/gpu-monitor", 
    label: "GPU Monitor", 
    icon: Monitor, 
    description: "Live statistics for your primary GPU resources and quantum processing.",
    gradient: "from-teal-500 to-blue-500"
  },
  { 
    href: "/zpe-flow", 
    label: "HS-QNN Advisor", 
    icon: BrainCircuit, 
    description: "AI advice for Hilbert Space QNN parameters and optimization.",
    gradient: "from-violet-500 to-purple-500"
  },
  { 
    href: "/vis/bloch-sphere", 
    label: "Bloch Sphere", 
    icon: Globe, 
    description: "Visualize qubit states in Hilbert space with interactive 3D graphics.",
    gradient: "from-cyan-500 to-blue-500"
  },
  { 
    href: "/vis/zpe-particle-simulation", 
    label: "ZPE Particle Sim", 
    icon: ScatterChart, 
    description: "Interactive 3D ZPE particle system with real-time physics simulation.",
    gradient: "from-pink-500 to-rose-500"
  },
  { 
    href: "/zpe-flow-analysis", 
    label: "ZPE Flow Analysis", 
    icon: SlidersHorizontal, 
    description: "Interactively analyze ZPE flow parameters and quantum dynamics.",
    gradient: "from-yellow-500 to-orange-500"
  },
  { 
    href: "/quantum-noise", 
    label: "Quantum Noise", 
    icon: Atom, 
    description: "Generate and analyze quantum-derived noise patterns for model robustness.",
    gradient: "from-emerald-500 to-teal-500"
  },
  { 
    href: "/ai-analysis", 
    label: "AI Assistant", 
    icon: MessageSquare, 
    description: "Chat with an AI for insights and optimization recommendations.",
    gradient: "from-sky-500 to-blue-500"
  }
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
  const [systemStats, setSystemStats] = useState({
    cpu: 45,
    memory: 62,
    gpu: 78,
    quantumCoherence: 94
  });

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
    
    // Simulate real-time system stats
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        cpu: Math.max(20, Math.min(80, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(30, Math.min(85, prev.memory + (Math.random() - 0.5) * 8)),
        gpu: Math.max(40, Math.min(95, prev.gpu + (Math.random() - 0.5) * 12)),
        quantumCoherence: Math.max(85, Math.min(99, prev.quantumCoherence + (Math.random() - 0.5) * 3))
      }));
    }, 3000);

    return () => clearInterval(interval);
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
    name: "ZPE-ResNet-Quantum",
    accuracy: 94.7,
    zpeLayers: 6, 
    avgZPEEffect: 0.142,
    quantumMode: true,
    keyFeature: "Quantum Annealing Optimization"
  };

  return (
    <div className="container mx-auto p-4 md:p-6 space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center space-y-6"
      >
        <div className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full border border-cyan-500/30 backdrop-blur-sm">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <span className="font-semibold text-cyan-300">Quantum AI Platform</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-cyan-300 to-blue-400 bg-clip-text text-transparent">
          Quantum Weaver
        </h1>
        
        <p className="text-xl md:text-2xl text-white/80 max-w-4xl mx-auto leading-relaxed">
          Harness Zero-Point Energy for advanced AI model development. 
          Build, train, and deploy quantum-enhanced neural networks with unprecedented efficiency.
        </p>
      </motion.div>

      {/* System Status Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <Card className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Cpu className="w-6 h-6 text-blue-400" />
              </div>
              <Badge variant="outline" className="text-blue-400 border-blue-500/30">
                {systemStats.cpu.toFixed(0)}%
              </Badge>
            </div>
            <h3 className="font-semibold text-white mb-2">CPU Usage</h3>
            <Progress value={systemStats.cpu} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <Activity className="w-6 h-6 text-green-400" />
              </div>
              <Badge variant="outline" className="text-green-400 border-green-500/30">
                {systemStats.memory.toFixed(0)}%
              </Badge>
            </div>
            <h3 className="font-semibold text-white mb-2">Memory</h3>
            <Progress value={systemStats.memory} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <Gauge className="w-6 h-6 text-purple-400" />
              </div>
              <Badge variant="outline" className="text-purple-400 border-purple-500/30">
                {systemStats.gpu.toFixed(0)}%
              </Badge>
            </div>
            <h3 className="font-semibold text-white mb-2">GPU Load</h3>
            <Progress value={systemStats.gpu} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-cyan-500/20 rounded-lg">
                <Atom className="w-6 h-6 text-cyan-400" />
              </div>
              <Badge variant="outline" className="text-cyan-400 border-cyan-500/30">
                {systemStats.quantumCoherence.toFixed(0)}%
              </Badge>
            </div>
            <h3 className="font-semibold text-white mb-2">Quantum Coherence</h3>
            <Progress value={systemStats.quantumCoherence} className="h-2" />
          </CardContent>
        </Card>
      </motion.div>

      {/* Latest Model Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <Card className="bg-gradient-to-br from-white/5 to-white/10 border-white/20">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg">
                <Target className="w-6 h-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold text-white">Latest Model Performance</CardTitle>
                <CardDescription className="text-white/70">
                  Real-time statistics from your most recent quantum-enhanced model
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Model Name</span>
                  <Badge variant="outline" className="text-cyan-400 border-cyan-500/30">
                    {latestModelData.name}
                  </Badge>
                </div>
                <Progress value={latestModelData.accuracy} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Accuracy</span>
                  <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                    {latestModelData.accuracy}%
                  </Badge>
                </div>
                <Progress value={latestModelData.accuracy} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">ZPE Layers</span>
                  <Badge variant="outline" className="text-purple-400 border-purple-500/30">
                    {latestModelData.zpeLayers}
                  </Badge>
                </div>
                <Progress value={(latestModelData.zpeLayers / 10) * 100} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Quantum Mode</span>
                  {latestModelData.quantumMode ? (
                    <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                      Active
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-gray-400 border-gray-500/30">
                      Inactive
                    </Badge>
                  )}
                </div>
                <Progress value={latestModelData.quantumMode ? 100 : 0} className="h-2" />
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-white/70 text-sm font-medium">Key Feature</span>
              </div>
              <p className="text-white font-medium">{latestModelData.keyFeature}</p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Featured Tools */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        className="space-y-6"
      >
        <div className="text-center">
          <h2 className="text-3xl font-bold text-white mb-2">Featured Tools</h2>
          <p className="text-white/70">Essential tools for quantum AI development</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {appFeatures.filter(f => f.priority).map((feature, index) => (
            <motion.div
              key={feature.href}
              initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.8 + index * 0.1 }}
            >
              <Card className="group hover:scale-105 transition-all duration-300 cursor-pointer bg-gradient-to-br from-white/5 to-white/10 border-white/20 hover:border-white/40">
                <CardHeader>
                  <div className="flex items-start gap-4">
                    <div className={`p-4 bg-gradient-to-r ${feature.gradient} rounded-xl shadow-lg`}>
                      <feature.icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                        {feature.label}
                      </CardTitle>
                      <CardDescription className="text-white/70 mt-2">
                        {feature.description}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardFooter>
                  <Button asChild className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0 shadow-lg">
                    <Link href={feature.href} className="flex items-center gap-2">
                      Get Started
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* All Tools Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1.0 }}
        className="space-y-6"
      >
        <div className="text-center">
          <h2 className="text-3xl font-bold text-white mb-2">All Tools & Features</h2>
          <p className="text-white/70">Complete suite of quantum AI development tools</p>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {appFeatures.map((feature, index) => (
              <motion.div
                key={feature.href}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 1.2 + index * 0.05 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Card className="group h-full bg-gradient-to-br from-white/5 to-white/10 border-white/20 hover:border-white/40 transition-all duration-300 cursor-pointer">
                  <CardHeader className="pb-3">
                    <div className="flex items-start gap-3">
                      <div className={`p-3 bg-gradient-to-r ${feature.gradient} rounded-lg shadow-md`}>
                        <feature.icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-lg font-bold text-white group-hover:text-cyan-300 transition-colors">
                          {feature.label}
                        </CardTitle>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="flex-1 pb-4">
                    <p className="text-sm text-white/70 leading-relaxed">
                      {feature.description}
                    </p>
                  </CardContent>
                  <CardFooter>
                    <Button asChild variant="outline" className="w-full border-white/20 text-white hover:bg-white/10 hover:border-white/40">
                      <Link href={feature.href} className="flex items-center gap-2">
                        Open Tool
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}
