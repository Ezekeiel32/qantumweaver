"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Wand2, 
  Upload, 
  Brain, 
  Play, 
  Download, 
  Settings, 
  Zap, 
  Target, 
  TrendingUp, 
  Activity,
  Database,
  Layers,
  Atom,
  BarChart3,
  FileText,
  Image,
  Sparkles,
  ArrowRight,
  CheckCircle,
  RefreshCw,
  Eye,
  Rocket,
  Cpu,
  Gauge
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter } from 'recharts';
import { motion, AnimatePresence } from "framer-motion";
import { useToast } from "@/hooks/use-toast";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

export default function WorkflowBuilder() {
  const [currentStep, setCurrentStep] = useState(0);
  const [workflowData, setWorkflowData] = useState({
    dataset: null,
    taskType: null,
    architecture: null,
    training: null,
    deployment: null
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [zpeMetrics, setZPEMetrics] = useState({
    resonance: 0,
    coherence: 0,
    entropy: 0,
    tunneling: 0
  });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const { toast } = useToast();

  const workflowSteps = [
    {
      id: 0,
      title: "Dataset Upload",
      description: "Drag & drop with quantum resonance scanning",
      icon: <Upload className="w-5 h-5" />,
      zpeFeature: "Quantum Data Resonance Scanning"
    },
    {
      id: 1,
      title: "Task Detection",
      description: "Entropic task type auto-detection",
      icon: <Target className="w-5 h-5" />,
      zpeFeature: "Entropic Task Detection"
    },
    {
      id: 2,
      title: "Architecture Design",
      description: "AI-powered ZPE parameter suggestion",
      icon: <Brain className="w-5 h-5" />,
      zpeFeature: "Quantum Annealing Optimization"
    },
    {
      id: 3,
      title: "ZPE Training",
      description: "Real-time adaptive learning",
      icon: <Zap className="w-5 h-5" />,
      zpeFeature: "ZPE Dynamics & Entanglement"
    },
    {
      id: 4,
      title: "Deployment",
      description: "One-click quantum compression",
      icon: <Download className="w-5 h-5" />,
      zpeFeature: "Quantum State Compression"
    }
  ];

  // Mock ZPE data generation
  useEffect(() => {
    const interval = setInterval(() => {
      setZPEMetrics(prev => ({
        resonance: Math.max(0, Math.min(100, prev.resonance + (Math.random() - 0.5) * 10)),
        coherence: Math.max(0, Math.min(100, prev.coherence + (Math.random() - 0.5) * 5)),
        entropy: Math.max(0, Math.min(10, prev.entropy + (Math.random() - 0.5) * 0.5)),
        tunneling: Math.max(0, Math.min(100, prev.tunneling + (Math.random() - 0.5) * 8))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    setIsProcessing(true);

    // Simulate quantum resonance scanning
    const uploadInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(uploadInterval);
          
          // Simulate ZPE resonance detection
          setTimeout(() => {
            const detectedType = file.name.includes('image') || file.type.startsWith('image/') ? 'image' : 'tabular';
            const resonancePeaks = Array.from({length: 3}, () => Math.random() * 100);
            
            setWorkflowData(prev => ({
              ...prev,
              dataset: {
                name: file.name,
                size: formatFileSize(file.size),
                type: detectedType,
                resonancePeaks,
                features: Math.floor(Math.random() * 500) + 50,
                samples: Math.floor(Math.random() * 10000) + 1000
              }
            }));
            
            setZPEMetrics(prev => ({
              ...prev,
              resonance: Math.max(...resonancePeaks),
              coherence: 85 + Math.random() * 10
            }));
            
            setIsUploading(false);
            setIsProcessing(false);
            setCurrentStep(1);
            toast({
              title: "Dataset Uploaded Successfully",
              description: `Quantum resonance scanning complete. Detected ${detectedType} dataset with ${resonancePeaks.length} resonance peaks.`
            });
          }, 1500);
          
          return 100;
        }
        return prev + Math.random() * 20;
      });
    }, 200);
  };

  const detectTaskType = async () => {
    setIsProcessing(true);
    
    // Simulate quantum entropy analysis
    setTimeout(() => {
      const entropy = 3 + Math.random() * 4;
      const taskType = entropy > 5.0 ? 'classification' : 'regression';
      const confidence = 85 + Math.random() * 10;
      
      setWorkflowData(prev => ({
        ...prev,
        taskType: {
          type: taskType,
          confidence: confidence,
          entropy: entropy,
          reasoning: `Quantum entropy analysis (H=${entropy.toFixed(2)}) suggests ${taskType} task with ${confidence.toFixed(1)}% confidence`
        }
      }));
      
      setZPEMetrics(prev => ({
        ...prev,
        entropy: entropy,
        coherence: confidence
      }));
      
      setIsProcessing(false);
      setCurrentStep(2);
      toast({
        title: "Task Type Detected",
        description: `${taskType.charAt(0).toUpperCase() + taskType.slice(1)} task identified with ${confidence.toFixed(1)}% confidence.`
      });
    }, 2000);
  };

  const generateArchitecture = async () => {
    setIsProcessing(true);
    
    // Simulate quantum annealing optimization
    setTimeout(() => {
      const architecture = {
        name: "ZPE-ResNet-Quantum",
        layers: [
          { type: "ZPEConvLayer", params: "64 filters, 3x3" },
          { type: "ZPEResidualBlock", params: "128 filters" },
          { type: "ZPEDenseField", params: "256 units" },
          { type: "QuantumMeasurement", params: "10 outputs" }
        ],
        hyperparams: {
          learning_rate: 0.042,
          zpe_flows: 6,
          quantum_depth: 4,
          entanglement_strength: 0.73
        },
        estimated_accuracy: 94.7,
        model_size: "3.7MB (compressed)"
      };
      
      setWorkflowData(prev => ({
        ...prev,
        architecture
      }));
      
      setZPEMetrics(prev => ({
        ...prev,
        tunneling: 75 + Math.random() * 20
      }));
      
      setIsProcessing(false);
      setCurrentStep(3);
      toast({
        title: "Architecture Generated",
        description: `ZPE-ResNet-Quantum architecture optimized with ${architecture.estimated_accuracy}% estimated accuracy.`
      });
    }, 3000);
  };

  const startTraining = async () => {
    setIsProcessing(true);
    
    // Simulate ZPE training
    setTimeout(() => {
      const training = {
        status: "completed",
        final_accuracy: 94.7,
        training_time: "12.3 minutes",
        zpe_effects: [0.12, 0.15, 0.09, 0.18, 0.11, 0.14],
        quantum_coherence: 96.2,
        entanglement_score: 0.87
      };
      
      setWorkflowData(prev => ({
        ...prev,
        training
      }));
      
      setIsProcessing(false);
      setCurrentStep(4);
      toast({
        title: "Training Complete",
        description: `Model achieved ${training.final_accuracy}% accuracy with ${training.quantum_coherence}% quantum coherence.`
      });
    }, 5000);
  };

  const deployModel = async () => {
    setIsProcessing(true);
    
    setTimeout(() => {
      const deployment = {
        status: "deployed",
        endpoint: "https://api.zpe-ai.com/v1/model/quantum-classifier",
        compression_ratio: "68% size reduction",
        quantum_state_size: "1.2MB",
        inference_speed: "0.8ms",
        deployment_time: "2.1 seconds"
      };
      
      setWorkflowData(prev => ({
        ...prev,
        deployment
      }));
      
      setIsProcessing(false);
      toast({
        title: "Model Deployed Successfully",
        description: `Model deployed with ${deployment.compression_ratio} compression and ${deployment.inference_speed} inference speed.`
      });
    }, 2500);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const generateResonanceData = () => {
    if (!workflowData.dataset?.resonancePeaks) return [];
    
    return Array.from({length: 50}, (_, i) => ({
      frequency: i * 2,
      amplitude: workflowData.dataset.resonancePeaks[0] * Math.exp(-Math.pow(i - 25, 2) / 200) + Math.random() * 10,
      amplitude2: workflowData.dataset.resonancePeaks[1] * Math.exp(-Math.pow(i - 15, 2) / 150) + Math.random() * 8,
      amplitude3: workflowData.dataset.resonancePeaks[2] * Math.exp(-Math.pow(i - 35, 2) / 180) + Math.random() * 6
    }));
  };

  const generateTrainingData = () => {
    return Array.from({length: 20}, (_, i) => ({
      epoch: i + 1,
      accuracy: Math.min(95, 10 + i * 4 + Math.random() * 5),
      loss: 2.5 * Math.exp(-i * 0.15) + Math.random() * 0.1,
      zpe_coherence: 80 + i * 0.8 + Math.random() * 5,
      quantum_tunneling: 20 + i * 3 + Math.random() * 10
    }));
  };

  return (
    <div className="container mx-auto p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full border border-blue-200"
        >
          <Atom className="w-6 h-6 text-blue-600" />
          <span className="font-semibold text-blue-800">No-Code Quantum AI Workflow</span>
        </motion.div>
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-800 via-blue-700 to-purple-600 bg-clip-text text-transparent">
          ZPE-Enhanced Model Builder
        </h1>
        
        <p className="text-xl text-slate-600 max-w-3xl mx-auto">
          Build quantum-inspired AI models with zero coding. From data upload to deployment in minutes.
        </p>
      </div>

      {/* Workflow Progress */}
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="w-5 h-5 text-purple-600" />
            Workflow Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-6">
            {workflowSteps.map((step, index) => (
              <div key={step.id} className="flex flex-col items-center flex-1">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                  currentStep > index 
                    ? 'bg-green-500 border-green-500 text-white' 
                    : currentStep === index 
                      ? 'bg-blue-500 border-blue-500 text-white animate-pulse' 
                      : 'bg-gray-100 border-gray-300 text-gray-400'
                }`}>
                  {currentStep > index ? <CheckCircle className="w-5 h-5" /> : step.icon}
                </div>
                <span className={`text-sm font-medium mt-2 ${
                  currentStep >= index ? 'text-slate-800' : 'text-slate-400'
                }`}>
                  {step.title}
                </span>
                <span className="text-xs text-slate-500 text-center max-w-24">
                  {step.description}
                </span>
                {index < workflowSteps.length - 1 && (
                  <ArrowRight className={`w-4 h-4 mt-2 ${
                    currentStep > index ? 'text-green-500' : 'text-gray-300'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ZPE Metrics Dashboard */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-5 h-5 text-purple-600" />
            Real-Time ZPE Quantum Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-white/50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{zpeMetrics.resonance.toFixed(1)}</div>
              <div className="text-sm text-slate-600">Resonance Peaks</div>
              <Progress value={zpeMetrics.resonance} className="mt-2" />
            </div>
            <div className="text-center p-4 bg-white/50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{zpeMetrics.coherence.toFixed(1)}%</div>
              <div className="text-sm text-slate-600">Quantum Coherence</div>
              <Progress value={zpeMetrics.coherence} className="mt-2" />
            </div>
            <div className="text-center p-4 bg-white/50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{zpeMetrics.entropy.toFixed(2)}</div>
              <div className="text-sm text-slate-600">Entropy Level</div>
              <Progress value={(zpeMetrics.entropy / 10) * 100} className="mt-2" />
            </div>
            <div className="text-center p-4 bg-white/50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{zpeMetrics.tunneling.toFixed(1)}%</div>
              <div className="text-sm text-slate-600">Tunneling Rate</div>
              <Progress value={zpeMetrics.tunneling} className="mt-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Workflow Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Step Content */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {workflowSteps[currentStep]?.icon}
              {workflowSteps[currentStep]?.title}
            </CardTitle>
            <CardDescription>
              ZPE Feature: {workflowSteps[currentStep]?.zpeFeature}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AnimatePresence mode="wait">
              {/* Step 0: Dataset Upload */}
              {currentStep === 0 && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                    <input
                      type="file"
                      onChange={handleFileUpload}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      accept=".csv,.json,.txt,.png,.jpg,.jpeg,.xlsx"
                    />
                    <Upload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-slate-800 mb-2">
                      Drop your dataset here
                    </h3>
                    <p className="text-slate-600">
                      Supports CSV, JSON, images, and more. ZPE quantum resonance scanning will auto-detect patterns.
                    </p>
                  </div>

                  {isUploading && (
                    <div className="space-y-4">
                      <div className="flex justify-between text-sm">
                        <span>Uploading and scanning quantum resonance...</span>
                        <span>{Math.round(uploadProgress)}%</span>
                      </div>
                      <Progress value={uploadProgress} />
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <p className="text-sm text-blue-800">
                          🔬 Quantum resonance scanning in progress...
                        </p>
                        <p className="text-xs text-blue-600 mt-1">
                          Analyzing data patterns using ZPE field detectors
                        </p>
                      </div>
                    </div>
                  )}

                  {workflowData.dataset && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-green-50 p-4 rounded-lg border border-green-200"
                    >
                      <h4 className="font-semibold text-green-800 mb-2">✨ Dataset Uploaded Successfully</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-green-600">Name:</span> {workflowData.dataset.name}
                        </div>
                        <div>
                          <span className="text-green-600">Size:</span> {workflowData.dataset.size}
                        </div>
                        <div>
                          <span className="text-green-600">Type:</span> {workflowData.dataset.type}
                        </div>
                        <div>
                          <span className="text-green-600">Features:</span> {workflowData.dataset.features}
                        </div>
                      </div>
                      <Button 
                        onClick={() => setCurrentStep(1)} 
                        className="mt-4 bg-green-600 hover:bg-green-700"
                      >
                        Continue to Task Detection
                      </Button>
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* Step 1: Task Detection */}
              {currentStep === 1 && (
                <motion.div
                  key="detection"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
                    <h3 className="text-lg font-semibold text-purple-800 mb-4">
                      🔬 Quantum Entropy Analysis
                    </h3>
                    <p className="text-purple-700 mb-4">
                      Analyzing data covariance eigenvalues to determine optimal task type using quantum entropy measurements.
                    </p>
                    
                    {!isProcessing && !workflowData.taskType && (
                      <Button onClick={detectTaskType} className="bg-purple-600 hover:bg-purple-700">
                        <Sparkles className="w-4 h-4 mr-2" />
                        Start Quantum Analysis
                      </Button>
                    )}

                    {isProcessing && (
                      <div className="space-y-3">
                        <Progress value={33} />
                        <p className="text-sm text-purple-600">
                          Computing quantum entropy from data covariance matrix...
                        </p>
                      </div>
                    )}

                    {workflowData.taskType && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white p-4 rounded-lg border border-purple-300"
                      >
                        <h4 className="font-semibold text-purple-800 mb-2">
                          📊 Task Type Detected
                        </h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span>Task Type:</span>
                            <Badge className="bg-purple-100 text-purple-800 capitalize">
                              {workflowData.taskType.type}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Confidence:</span>
                            <span className="font-semibold">{workflowData.taskType.confidence.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Quantum Entropy:</span>
                            <span className="font-semibold">{workflowData.taskType.entropy.toFixed(3)}</span>
                          </div>
                        </div>
                        <p className="text-xs text-purple-600 mt-3 p-2 bg-purple-50 rounded">
                          {workflowData.taskType.reasoning}
                        </p>
                        <Button 
                          onClick={() => setCurrentStep(2)} 
                          className="mt-4 bg-purple-600 hover:bg-purple-700"
                        >
                          Design Architecture
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 2: Architecture Design */}
              {currentStep === 2 && (
                <motion.div
                  key="architecture"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
                    <h3 className="text-lg font-semibold text-blue-800 mb-4">
                      🧠 Quantum Annealing Architecture Search
                    </h3>
                    <p className="text-blue-700 mb-4">
                      Using quantum-inspired optimization to find the optimal neural architecture for your task.
                    </p>
                    
                    {!isProcessing && !workflowData.architecture && (
                      <Button onClick={generateArchitecture} className="bg-blue-600 hover:bg-blue-700">
                        <Brain className="w-4 h-4 mr-2" />
                        Generate ZPE Architecture
                      </Button>
                    )}

                    {isProcessing && (
                      <div className="space-y-3">
                        <Progress value={66} />
                        <p className="text-sm text-blue-600">
                          Quantum annealing optimization in progress...
                        </p>
                      </div>
                    )}

                    {workflowData.architecture && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white p-4 rounded-lg border border-blue-300"
                      >
                        <h4 className="font-semibold text-blue-800 mb-4">
                          🏗️ Optimal Architecture Found
                        </h4>
                        
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <Label className="text-sm font-medium text-blue-600">Model</Label>
                            <p className="font-semibold">{workflowData.architecture.name}</p>
                          </div>
                          <div>
                            <Label className="text-sm font-medium text-blue-600">Est. Accuracy</Label>
                            <p className="font-semibold text-green-600">{workflowData.architecture.estimated_accuracy}%</p>
                          </div>
                        </div>

                        <div className="space-y-2 mb-4">
                          <Label className="text-sm font-medium text-blue-600">Layer Architecture</Label>
                          {workflowData.architecture.layers.map((layer, index) => (
                            <div key={index} className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                              <Layers className="w-4 h-4 text-blue-500" />
                              <span className="font-medium">{layer.type}</span>
                              <span className="text-sm text-blue-600">({layer.params})</span>
                            </div>
                          ))}
                        </div>

                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-blue-600">Learning Rate:</span>
                            <span className="font-semibold ml-2">{workflowData.architecture.hyperparams.learning_rate}</span>
                          </div>
                          <div>
                            <span className="text-blue-600">ZPE Flows:</span>
                            <span className="font-semibold ml-2">{workflowData.architecture.hyperparams.zpe_flows}</span>
                          </div>
                          <div>
                            <span className="text-blue-600">Quantum Depth:</span>
                            <span className="font-semibold ml-2">{workflowData.architecture.hyperparams.quantum_depth}</span>
                          </div>
                          <div>
                            <span className="text-blue-600">Model Size:</span>
                            <span className="font-semibold ml-2">{workflowData.architecture.model_size}</span>
                          </div>
                        </div>

                        <Button 
                          onClick={() => setCurrentStep(3)} 
                          className="mt-4 bg-blue-600 hover:bg-blue-700 w-full"
                        >
                          Start ZPE Training
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 3: Training */}
              {currentStep === 3 && (
                <motion.div
                  key="training"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                    <h3 className="text-lg font-semibold text-green-800 mb-4">
                      ⚡ ZPE Quantum Training
                    </h3>
                    
                    {!isProcessing && !workflowData.training && (
                      <div>
                        <p className="text-green-700 mb-4">
                          Ready to train your quantum-enhanced model with real-time adaptive learning.
                        </p>
                        <Button onClick={startTraining} className="bg-green-600 hover:bg-green-700">
                          <Play className="w-4 h-4 mr-2" />
                          Begin Training
                        </Button>
                      </div>
                    )}

                    {isProcessing && (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <RefreshCw className="w-4 h-4 animate-spin text-green-600" />
                          <span className="text-green-700">Training in progress...</span>
                        </div>
                        
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={generateTrainingData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="epoch" />
                              <YAxis />
                              <Tooltip />
                              <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} name="Accuracy %" />
                              <Line type="monotone" dataKey="zpe_coherence" stroke="#8b5cf6" strokeWidth={2} name="ZPE Coherence %" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}

                    {workflowData.training && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white p-4 rounded-lg border border-green-300"
                      >
                        <h4 className="font-semibold text-green-800 mb-4">
                          🎯 Training Complete!
                        </h4>
                        
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <Label className="text-sm font-medium text-green-600">Final Accuracy</Label>
                            <p className="text-2xl font-bold text-green-700">{workflowData.training.final_accuracy}%</p>
                          </div>
                          <div>
                            <Label className="text-sm font-medium text-green-600">Training Time</Label>
                            <p className="text-lg font-semibold">{workflowData.training.training_time}</p>
                          </div>
                          <div>
                            <Label className="text-sm font-medium text-green-600">Quantum Coherence</Label>
                            <p className="text-lg font-semibold">{workflowData.training.quantum_coherence}%</p>
                          </div>
                          <div>
                            <Label className="text-sm font-medium text-green-600">Entanglement Score</Label>
                            <p className="text-lg font-semibold">{workflowData.training.entanglement_score}</p>
                          </div>
                        </div>

                        <Button 
                          onClick={() => setCurrentStep(4)} 
                          className="mt-4 bg-green-600 hover:bg-green-700 w-full"
                        >
                          Deploy Model
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 4: Deployment */}
              {currentStep === 4 && (
                <motion.div
                  key="deployment"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
                    <h3 className="text-lg font-semibold text-purple-800 mb-4">
                      🚀 Quantum State Deployment
                    </h3>
                    
                    {!isProcessing && !workflowData.deployment && (
                      <div>
                        <p className="text-purple-700 mb-4">
                          Deploy your model as compressed quantum state for edge inference.
                        </p>
                        <Button onClick={deployModel} className="bg-purple-600 hover:bg-purple-700">
                          <Download className="w-4 w-4 mr-2" />
                          Deploy with Quantum Compression
                        </Button>
                      </div>
                    )}

                    {isProcessing && (
                      <div className="space-y-3">
                        <Progress value={90} />
                        <p className="text-sm text-purple-600">
                          Compressing model to quantum state representation...
                        </p>
                      </div>
                    )}

                    {workflowData.deployment && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white p-4 rounded-lg border border-purple-300"
                      >
                        <h4 className="font-semibold text-purple-800 mb-4">
                          ✅ Model Deployed Successfully!
                        </h4>
                        
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-purple-600">API Endpoint:</span>
                            <code className="text-sm bg-purple-100 px-2 py-1 rounded">
                              {workflowData.deployment.endpoint}
                            </code>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-purple-600">Compression:</span>
                            <span className="font-semibold">{workflowData.deployment.compression_ratio}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-purple-600">Quantum State Size:</span>
                            <span className="font-semibold">{workflowData.deployment.quantum_state_size}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-purple-600">Inference Speed:</span>
                            <span className="font-semibold">{workflowData.deployment.inference_speed}</span>
                          </div>
                        </div>

                        <div className="flex gap-2 mt-4">
                          <Button className="bg-purple-600 hover:bg-purple-700 flex-1">
                            <Eye className="w-4 h-4 mr-2" />
                            Test Model
                          </Button>
                          <Button variant="outline" className="flex-1">
                            <Download className="w-4 h-4 mr-2" />
                            Download SDK
                          </Button>
                        </div>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>

        {/* ZPE Visualization Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-600" />
              ZPE Effects Monitor
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="resonance" className="space-y-4">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="resonance">Resonance</TabsTrigger>
                <TabsTrigger value="training">Training</TabsTrigger>
              </TabsList>

              <TabsContent value="resonance" className="space-y-4">
                {workflowData.dataset ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={generateResonanceData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="frequency" />
                        <YAxis />
                        <Tooltip />
                        <Area type="monotone" dataKey="amplitude" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                        <Area type="monotone" dataKey="amplitude2" stackId="2" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-slate-500">
                    Upload dataset to see quantum resonance patterns
                  </div>
                )}
              </TabsContent>

              <TabsContent value="training" className="space-y-4">
                {workflowData.training ? (
                  <div className="space-y-4">
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={generateTrainingData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} />
                          <Line type="monotone" dataKey="quantum_tunneling" stroke="#f59e0b" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      {workflowData.training.zpe_effects.map((effect, index) => (
                        <div key={index} className="text-center p-2 bg-blue-50 rounded">
                          <div className="font-bold text-blue-700">L{index + 1}</div>
                          <div>{effect.toFixed(3)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-slate-500">
                    Start training to see ZPE dynamics
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 