"use client";
import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { 
  Wand2, 
  Database, 
  Zap, 
  Atom, 
  Brain, 
  Target, 
  ArrowRight, 
  Sparkles, 
  Play,
  CheckCircle,
  Star,
  Users,
  TrendingUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const features = [
  {
    icon: Wand2,
    title: "No-Code Workflow Builder",
    description: "Build quantum AI models with zero coding. From data upload to deployment in minutes.",
    gradient: "from-purple-500 to-pink-500",
    href: "/workflow-builder"
  },
  {
    icon: Database,
    title: "Quantum Data Portal",
    description: "Upload, manage, and analyze datasets with quantum resonance scanning and ZPE field detection.",
    gradient: "from-blue-500 to-cyan-500",
    href: "/data-portal"
  },
  {
    icon: Brain,
    title: "HS-QNN Advisor",
    description: "AI-powered advice for Hilbert Space Quantum Neural Network parameters and optimization.",
    gradient: "from-green-500 to-emerald-500",
    href: "/zpe-flow"
  },
  {
    icon: Atom,
    title: "Quantum Visualizations",
    description: "Interactive 3D visualizations of quantum states, Bloch spheres, and ZPE particle dynamics.",
    gradient: "from-orange-500 to-red-500",
    href: "/vis/bloch-sphere"
  }
];

const stats = [
  { label: "Models Trained", value: "1,247", icon: TrendingUp },
  { label: "Active Users", value: "2,891", icon: Users },
  { label: "Accuracy Boost", value: "+23%", icon: Star },
  { label: "Processing Speed", value: "12x", icon: Zap }
];

const benefits = [
  "Zero coding required for quantum AI development",
  "Real-time quantum resonance scanning",
  "Advanced ZPE field optimization",
  "Interactive 3D quantum visualizations",
  "AI-powered parameter suggestions",
  "One-click model deployment",
  "Quantum state compression",
  "Live system monitoring"
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10" />
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="1"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-30" />
        
        <div className="relative container mx-auto px-4 py-20 md:py-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center space-y-8"
          >
            <div className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full border border-cyan-500/30 backdrop-blur-sm">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <span className="font-semibold text-cyan-300">Quantum AI Platform</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-white via-cyan-300 to-blue-400 bg-clip-text text-transparent leading-tight">
              Quantum Weaver
            </h1>
            
            <p className="text-xl md:text-2xl text-white/80 max-w-4xl mx-auto leading-relaxed">
              Harness Zero-Point Energy for advanced AI model development. 
              Build, train, and deploy quantum-enhanced neural networks with unprecedented efficiency.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button asChild size="lg" className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0 shadow-lg px-8 py-4 text-lg">
                <Link href="/workflow-builder" className="flex items-center gap-2">
                  <Play className="w-5 h-5" />
                  Get Started
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="border-white/20 text-white hover:bg-white/10 px-8 py-4 text-lg">
                <Link href="/dashboard" className="flex items-center gap-2">
                  Explore Dashboard
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white/5 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="flex justify-center mb-4">
                  <div className="p-3 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg">
                    <stat.icon className="w-8 h-8 text-cyan-400" />
                  </div>
                </div>
                <div className="text-3xl md:text-4xl font-bold text-white mb-2">{stat.value}</div>
                <div className="text-white/70">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Revolutionary Features
            </h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Experience the future of AI development with our cutting-edge quantum-enhanced tools
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="group hover:scale-105 transition-all duration-300 cursor-pointer bg-gradient-to-br from-white/5 to-white/10 border-white/20 hover:border-white/40 h-full">
                  <CardHeader>
                    <div className="flex items-start gap-4">
                      <div className={`p-4 bg-gradient-to-r ${feature.gradient} rounded-xl shadow-lg`}>
                        <feature.icon className="w-8 h-8 text-white" />
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-2xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                          {feature.title}
                        </CardTitle>
                        <CardDescription className="text-white/70 mt-2 text-lg">
                          {feature.description}
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <Button asChild className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0 shadow-lg">
                      <Link href={feature.href} className="flex items-center gap-2">
                        Explore Feature
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </Link>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 bg-white/5 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Why Choose Quantum Weaver?
            </h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Unlock the power of quantum computing for your AI development workflow
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {benefits.map((benefit, index) => (
              <motion.div
                key={benefit}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex items-start gap-3 p-4 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-all duration-300"
              >
                <CheckCircle className="w-6 h-6 text-green-400 mt-1 flex-shrink-0" />
                <span className="text-white/90">{benefit}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <Card className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/20 max-w-4xl mx-auto">
              <CardHeader className="text-center">
                <div className="flex justify-center mb-4">
                  <div className="p-4 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full">
                    <Zap className="w-8 h-8 text-white" />
                  </div>
                </div>
                <CardTitle className="text-3xl md:text-4xl font-bold text-white">
                  Ready to Transform Your AI Development?
                </CardTitle>
                <CardDescription className="text-xl text-white/70">
                  Start building quantum-enhanced AI models today with our revolutionary platform
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <Button asChild size="lg" className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0 shadow-lg px-8 py-4 text-lg">
                  <Link href="/workflow-builder" className="flex items-center gap-2">
                    <Wand2 className="w-5 h-5" />
                    Start Building
                    <ArrowRight className="w-5 h-5" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="border-white/20 text-white hover:bg-white/10 px-8 py-4 text-lg">
                  <Link href="/data-portal" className="flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Upload Data
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-white/10">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="p-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-white to-cyan-300 bg-clip-text text-transparent">
              Quantum Weaver
            </span>
          </div>
          <p className="text-white/60 mb-4">
            Harnessing Zero-Point Energy for the future of AI development
          </p>
          <div className="flex justify-center gap-6">
            <Link href="/dashboard" className="text-white/60 hover:text-white transition-colors">
              Dashboard
            </Link>
            <Link href="/workflow-builder" className="text-white/60 hover:text-white transition-colors">
              Workflow Builder
            </Link>
            <Link href="/data-portal" className="text-white/60 hover:text-white transition-colors">
              Data Portal
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
