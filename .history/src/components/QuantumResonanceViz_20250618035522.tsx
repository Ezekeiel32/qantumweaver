import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Atom, Zap, Activity, Target, Gauge, Waveform } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, AreaChart, Area } from 'recharts';

interface QuantumResonanceVizProps {
  data?: any;
  isActive?: boolean;
  intensity?: number;
  frequency?: number;
  onIntensityChange?: (intensity: number) => void;
  onFrequencyChange?: (frequency: number) => void;
}

export default function QuantumResonanceViz({ 
  data = null, 
  isActive = false, 
  intensity = 50,
  frequency = 2.4,
  onIntensityChange,
  onFrequencyChange
}: QuantumResonanceVizProps) {
  const [resonanceData, setResonanceData] = useState([]);
  const [currentIntensity, setCurrentIntensity] = useState(intensity);
  const [currentFrequency, setCurrentFrequency] = useState(frequency);
  const [isPaused, setIsPaused] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState<'scatter' | 'line' | 'area'>('scatter');

  useEffect(() => {
    if (!data && !isActive) return;

    // Generate quantum resonance field visualization
    const generateResonanceField = () => {
      const field = [];
      for (let x = 0; x < 20; x++) {
        for (let y = 0; y < 20; y++) {
          const distance = Math.sqrt(Math.pow(x - 10, 2) + Math.pow(y - 10, 2));
          const intensity = Math.exp(-distance / 5) * (0.5 + 0.5 * Math.sin(Date.now() * 0.001 + distance * 0.5));
          field.push({
            x: x * 5,
            y: y * 5,
            z: intensity * 100,
            color: intensity > 0.7 ? '#ff4444' : intensity > 0.4 ? '#ffaa44' : '#4488ff'
          });
        }
      }
      return field;
    };

    const generateTimeSeriesData = () => {
      return Array.from({length: 100}, (_, i) => ({
        time: i * 0.1,
        amplitude: Math.sin(i * 0.1 * currentFrequency) * (currentIntensity / 100) + Math.random() * 0.1,
        phase: Math.cos(i * 0.1 * currentFrequency * 0.5) * (currentIntensity / 100),
        resonance: Math.exp(-Math.pow(i * 0.1 - 5, 2) / 10) * (currentIntensity / 100)
      }));
    };

    const interval = setInterval(() => {
      if (isActive && !isPaused) {
        setResonanceData(generateResonanceField());
        setCurrentIntensity(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 2)));
      }
    }, 100);

    return () => clearInterval(interval);
  }, [data, isActive, isPaused, currentFrequency, currentIntensity]);

  const getIntensityStatus = (intensity: number) => {
    if (intensity > 80) return { status: "Critical", color: "bg-red-500", text: "text-red-700" };
    if (intensity > 60) return { status: "High", color: "bg-orange-500", text: "text-orange-700" };
    if (intensity > 30) return { status: "Moderate", color: "bg-yellow-500", text: "text-yellow-700" };
    return { status: "Low", color: "bg-green-500", text: "text-green-700" };
  };

  const intensityStatus = getIntensityStatus(currentIntensity);

  const handleIntensityChange = (value: number[]) => {
    const newIntensity = value[0];
    setCurrentIntensity(newIntensity);
    onIntensityChange?.(newIntensity);
  };

  const handleFrequencyChange = (value: number[]) => {
    const newFrequency = value[0];
    setCurrentFrequency(newFrequency);
    onFrequencyChange?.(newFrequency);
  };

  const generateTimeSeriesData = () => {
    return Array.from({length: 100}, (_, i) => ({
      time: i * 0.1,
      amplitude: Math.sin(i * 0.1 * currentFrequency) * (currentIntensity / 100) + Math.random() * 0.1,
      phase: Math.cos(i * 0.1 * currentFrequency * 0.5) * (currentIntensity / 100),
      resonance: Math.exp(-Math.pow(i * 0.1 - 5, 2) / 10) * (currentIntensity / 100)
    }));
  };

  if (!data && !isActive) {
    return (
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-blue-600" />
            Quantum Resonance Field
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-slate-500">
            <Atom className="w-16 h-16 mx-auto mb-4 text-slate-300" />
            <p>Upload dataset to visualize quantum resonance patterns</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-white/80 backdrop-blur-sm border-slate-200/50 hover:shadow-xl transition-all duration-500">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg">
              <Atom className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <CardTitle className="text-xl font-bold text-slate-800">
                Quantum Resonance Field
              </CardTitle>
              <p className="text-sm text-slate-600">Real-time ZPE fluctuations</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={`${intensityStatus.color} text-white px-3 py-1`}>
              {intensityStatus.status}
            </Badge>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? <Activity className="w-4 h-4" /> : <Target className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium text-slate-600">Intensity</Label>
            <Slider
              value={[currentIntensity]}
              onValueChange={handleIntensityChange}
              max={100}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-500">
              <span>0%</span>
              <span>{currentIntensity.toFixed(0)}%</span>
              <span>100%</span>
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium text-slate-600">Frequency (THz)</Label>
            <Slider
              value={[currentFrequency]}
              onValueChange={handleFrequencyChange}
              min={0.1}
              max={10}
              step={0.1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-500">
              <span>0.1 THz</span>
              <span>{currentFrequency.toFixed(1)} THz</span>
              <span>10 THz</span>
            </div>
          </div>
        </div>

        {/* Visualization Mode Selector */}
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={visualizationMode === 'scatter' ? 'default' : 'outline'}
            onClick={() => setVisualizationMode('scatter')}
          >
            <Target className="w-4 h-4 mr-1" />
            Field
          </Button>
          <Button
            size="sm"
            variant={visualizationMode === 'line' ? 'default' : 'outline'}
            onClick={() => setVisualizationMode('line')}
          >
            <Waveform className="w-4 h-4 mr-1" />
            Waveform
          </Button>
          <Button
            size="sm"
            variant={visualizationMode === 'area' ? 'default' : 'outline'}
            onClick={() => setVisualizationMode('area')}
          >
            <Gauge className="w-4 h-4 mr-1" />
            Resonance
          </Button>
        </div>

        {/* Current Resonance Level */}
        <div className="flex items-center justify-between p-4 bg-gradient-to-r from-slate-50 to-blue-50/50 rounded-xl">
          <div>
            <p className="text-sm font-medium text-slate-600">Current Resonance</p>
            <p className="text-3xl font-bold text-slate-800">{currentIntensity.toFixed(1)}</p>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-slate-600">Peak Frequency</p>
            <div className="flex items-center gap-1 text-blue-600">
              <Zap className="w-4 h-4" />
              <span className="font-bold">{currentFrequency.toFixed(2)} THz</span>
            </div>
          </div>
        </div>

        {/* Visualization */}
        <div className="relative h-64 bg-gradient-to-br from-slate-900 to-blue-900 rounded-xl p-4 overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            {isActive ? (
              <ResponsiveContainer width="100%" height="100%">
                {visualizationMode === 'scatter' ? (
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
                    <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
                    <Tooltip 
                      formatter={(value, name) => [value.toFixed(2), "Intensity"]}
                      labelFormatter={(label) => `Position: ${label}`}
                    />
                    <Scatter data={resonanceData} fill="#8884d8">
                      {resonanceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                ) : visualizationMode === 'line' ? (
                  <LineChart data={generateTimeSeriesData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="time" stroke="#888" />
                    <YAxis stroke="#888" />
                    <Tooltip />
                    <Line type="monotone" dataKey="amplitude" stroke="#00ff88" strokeWidth={2} />
                    <Line type="monotone" dataKey="phase" stroke="#ff0088" strokeWidth={2} />
                  </LineChart>
                ) : (
                  <AreaChart data={generateTimeSeriesData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="time" stroke="#888" />
                    <YAxis stroke="#888" />
                    <Tooltip />
                    <Area type="monotone" dataKey="resonance" stroke="#4488ff" fill="#4488ff" fillOpacity={0.6} />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            ) : (
              <div className="text-center text-white/60">
                <Activity className="w-12 h-12 mx-auto mb-2 animate-pulse" />
                <p>Quantum field inactive</p>
              </div>
            )}
          </div>
          
          {/* Overlay indicators */}
          <div className="absolute top-2 left-4 flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-xs font-medium text-white/80">
              {isActive ? 'Active Field' : 'Field Inactive'}
            </span>
          </div>
        </div>

        {/* Resonance Parameters */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-blue-50/50 rounded-lg">
            <p className="text-xs font-medium text-blue-600 uppercase tracking-wide">Amplitude</p>
            <p className="text-lg font-bold text-blue-800">{(currentIntensity / 100).toFixed(2)}</p>
          </div>
          <div className="text-center p-3 bg-purple-50/50 rounded-lg">
            <p className="text-xs font-medium text-purple-600 uppercase tracking-wide">Phase</p>
            <p className="text-lg font-bold text-purple-800">{(Math.random() * Math.PI * 2).toFixed(2)}</p>
          </div>
          <div className="text-center p-3 bg-green-50/50 rounded-lg">
            <p className="text-xs font-medium text-green-600 uppercase tracking-wide">Coherence</p>
            <p className="text-lg font-bold text-green-800">{(85 + Math.random() * 10).toFixed(1)}%</p>
          </div>
        </div>

        {/* Detected Resonance Peaks */}
        {data?.resonancePeaks && (
          <div className="space-y-3">
            <h4 className="font-semibold text-slate-800">Detected Resonance Peaks</h4>
            <div className="grid gap-2">
              {data.resonancePeaks.map((peak: number, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
                    <span className="font-medium text-slate-800">Peak {index + 1}</span>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-blue-700">{peak.toFixed(1)}</p>
                    <p className="text-xs text-blue-600">Intensity</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 