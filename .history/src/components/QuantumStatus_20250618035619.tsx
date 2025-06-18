import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Zap, Cpu, Gpu, Memory, HardDrive, Activity, RefreshCw, TrendingUp, AlertTriangle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface SystemStats {
  cpu: {
    percent: number;
    cores: number;
    temperature: number;
  };
  gpu: {
    total: number;
    allocated: number;
    percent: number;
    temperature: number;
    memory_used: number;
    memory_total: number;
  };
  memory: {
    total: number;
    used: number;
    percent: number;
  };
  disk: {
    total: number;
    used: number;
    percent: number;
  };
  network: {
    bytes_sent: number;
    bytes_recv: number;
  };
  quantum_metrics: {
    coherence: number;
    entanglement: number;
    tunneling_rate: number;
    zpe_stability: number;
  };
}

interface QuantumStatusProps {
  stats?: SystemStats | null;
  isLive?: boolean;
  onRefresh?: () => void;
}

export default function QuantumStatus({ stats, isLive = true, onRefresh }: QuantumStatusProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [showDetails, setShowDetails] = useState(false);

  // Generate mock historical data
  useEffect(() => {
    const generateHistoricalData = () => {
      return Array.from({ length: 20 }, (_, i) => ({
        time: i * 5,
        cpu: 30 + Math.random() * 40,
        gpu: 20 + Math.random() * 60,
        memory: 40 + Math.random() * 30,
        coherence: 80 + Math.random() * 15,
        entanglement: 70 + Math.random() * 20
      }));
    };
    setHistoricalData(generateHistoricalData());
  }, []);

  const handleRefresh = async () => {
    setIsLoading(true);
    onRefresh?.();
    setTimeout(() => setIsLoading(false), 1000);
  };

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return "text-red-600";
    if (value >= thresholds.warning) return "text-orange-600";
    return "text-green-600";
  };

  const getStatusBadge = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return { text: "Critical", color: "bg-red-100 text-red-800" };
    if (value >= thresholds.warning) return { text: "Warning", color: "bg-orange-100 text-orange-800" };
    return { text: "Normal", color: "bg-green-100 text-green-800" };
  };

  if (!stats) {
    return (
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl font-bold text-slate-800">
            <Zap className="w-6 h-6 text-teal-600" />
            System & Quantum Status
          </CardTitle>
          <CardDescription>Awaiting backend connection...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <Activity className="w-12 h-12 mx-auto mb-4 text-slate-300 animate-pulse" />
              <p className="text-slate-500">Could not fetch system stats.</p>
              <Button 
                onClick={handleRefresh} 
                variant="outline" 
                size="sm" 
                className="mt-2"
                disabled={isLoading}
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Retry
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { cpu, gpu, memory, disk, quantum_metrics } = stats;
  const gpu_usage = (gpu.allocated / gpu.total) * 100;
  const memory_usage = (memory.used / memory.total) * 100;
  const disk_usage = (disk.used / disk.total) * 100;

  return (
    <Card className="bg-white/80 backdrop-blur-sm border-slate-200/50 hover:shadow-xl transition-all duration-500">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-teal-100 to-blue-100 rounded-lg">
              <Zap className="w-6 h-6 text-teal-600" />
            </div>
            <div>
              <CardTitle className="text-xl font-bold text-slate-800">
                System Status
              </CardTitle>
              <CardDescription>Real-time performance from backend</CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={isLive ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"}>
              {isLive ? "Live" : "Offline"}
            </Badge>
            <Button
              size="sm"
              variant="outline"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? "Hide" : "Details"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-slate-600">CPU</span>
              </div>
              <Badge className={getStatusBadge(cpu.percent, { warning: 70, critical: 90 }).color}>
                {getStatusBadge(cpu.percent, { warning: 70, critical: 90 }).text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span className={`font-bold ${getStatusColor(cpu.percent, { warning: 70, critical: 90 })}`}>
                  {cpu.percent.toFixed(1)}%
                </span>
              </div>
              <Progress value={cpu.percent} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>{cpu.cores} cores</span>
                <span>{cpu.temperature}°C</span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Gpu className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-slate-600">GPU</span>
              </div>
              <Badge className={getStatusBadge(gpu_usage, { warning: 80, critical: 95 }).color}>
                {getStatusBadge(gpu_usage, { warning: 80, critical: 95 }).text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span className={`font-bold ${getStatusColor(gpu_usage, { warning: 80, critical: 95 })}`}>
                  {gpu_usage.toFixed(1)}%
                </span>
              </div>
              <Progress value={gpu_usage} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>{gpu.memory_used}GB</span>
                <span>{gpu.temperature}°C</span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Memory className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-slate-600">Memory</span>
              </div>
              <Badge className={getStatusBadge(memory_usage, { warning: 80, critical: 90 }).color}>
                {getStatusBadge(memory_usage, { warning: 80, critical: 90 }).text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span className={`font-bold ${getStatusColor(memory_usage, { warning: 80, critical: 90 })}`}>
                  {memory_usage.toFixed(1)}%
                </span>
              </div>
              <Progress value={memory_usage} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>{memory.used}GB</span>
                <span>{memory.total}GB</span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-orange-600" />
                <span className="text-sm font-medium text-slate-600">Disk</span>
              </div>
              <Badge className={getStatusBadge(disk_usage, { warning: 85, critical: 95 }).color}>
                {getStatusBadge(disk_usage, { warning: 85, critical: 95 }).text}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>Usage</span>
                <span className={`font-bold ${getStatusColor(disk_usage, { warning: 85, critical: 95 })}`}>
                  {disk_usage.toFixed(1)}%
                </span>
              </div>
              <Progress value={disk_usage} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500">
                <span>{disk.used}GB</span>
                <span>{disk.total}GB</span>
              </div>
            </div>
          </div>
        </div>

        {/* Quantum Metrics */}
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg border border-purple-200">
          <h3 className="font-semibold text-purple-800 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Quantum Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{quantum_metrics.coherence.toFixed(1)}%</div>
              <div className="text-sm text-purple-700">Coherence</div>
              <Progress value={quantum_metrics.coherence} className="mt-1 h-1" />
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{quantum_metrics.entanglement.toFixed(1)}%</div>
              <div className="text-sm text-blue-700">Entanglement</div>
              <Progress value={quantum_metrics.entanglement} className="mt-1 h-1" />
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{quantum_metrics.tunneling_rate.toFixed(1)}%</div>
              <div className="text-sm text-green-700">Tunneling</div>
              <Progress value={quantum_metrics.tunneling_rate} className="mt-1 h-1" />
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{quantum_metrics.zpe_stability.toFixed(1)}%</div>
              <div className="text-sm text-orange-700">ZPE Stability</div>
              <Progress value={quantum_metrics.zpe_stability} className="mt-1 h-1" />
            </div>
          </div>
        </div>

        {/* Historical Data Chart */}
        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="space-y-3">
                <h4 className="font-semibold text-slate-800 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Performance History (Last 5 minutes)
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="cpu" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                      <Area type="monotone" dataKey="gpu" stackId="2" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                      <Area type="monotone" dataKey="coherence" stackId="3" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex gap-2 text-xs text-slate-500">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-500 rounded"></div>
                    CPU Usage
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-purple-500 rounded"></div>
                    GPU Usage
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-green-500 rounded"></div>
                    Quantum Coherence
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Alerts */}
        {(cpu.percent > 90 || gpu_usage > 95 || memory_usage > 90) && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-600" />
              <span className="font-medium text-red-800">System Alert</span>
            </div>
            <p className="text-sm text-red-700 mt-1">
              High resource usage detected. Consider optimizing your workload.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 