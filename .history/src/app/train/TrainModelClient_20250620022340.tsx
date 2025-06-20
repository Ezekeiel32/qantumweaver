"use client";

import React, { useState, useEffect, useCallback, useRef, Suspense } from "react";
import { useForm, Controller, Control, FieldPath, FieldValues } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import type { TrainingParameters, TrainingJob, TrainingJobSummary } from "@/types/training";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/hooks/use-toast";
import { Play, StopCircle, List, Zap, Settings, RefreshCw, AlertTriangle, ExternalLink, SlidersHorizontal, Atom, Brain, Waves, BrainCircuit, Wand2, Save, Download, ArrowDownCircle, PlayCircle } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { TooltipProvider, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { defaultZPEParams } from "@/lib/constants";
import NeonAnalyzerChart from "@/components/visualizations/NeonAnalyzerChart";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import InstrumentCard from '../../components/InstrumentCard';
import { useJobStatusPolling } from '../../hooks/useJobStatusPolling';
import { createPortal } from "react-dom";
import ChartPanel from "@/components/ChartPanel";
import TerminalPanel from "@/components/TerminalPanel";
import ControlsPanel from "@/components/ControlsPanel";

// ...all the rest of your code from TrainModelPage, unchanged...

export default function TrainModelClient() {
  const job = useJobStatusPolling();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [hasStartedFromAdvisor, setHasStartedFromAdvisor] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [pendingParams, setPendingParams] = useState<any>(defaultZPEParams);
  const [advisorParams, setAdvisorParams] = useState<any>(null);

  useEffect(() => {
    let paramsStr = searchParams.get('advisorParams') || searchParams.get('advisedParams');
    if (paramsStr) {
      let parsedParams: any = null;
      try {
        if (/^[A-Za-z0-9+/=]+$/.test(paramsStr) && paramsStr.length % 4 === 0) {
          parsedParams = JSON.parse(atob(paramsStr));
        } else {
          parsedParams = JSON.parse(decodeURIComponent(paramsStr));
        }
      } catch (e) {
        const err = e as Error;
        toast({ title: "Error loading advisor parameters", description: err.message, variant: "destructive" });
        return;
      }
      const mergedParams = {
        ...defaultZPEParams,
        ...parsedParams,
        couplingParams: parsedParams.couplingParams || defaultZPEParams.couplingParams,
        mixupAlpha: parsedParams.mixupAlpha ?? 0.2,
      };
      setPendingParams(mergedParams);
      setAdvisorParams(mergedParams);
      if (!hasStartedFromAdvisor) {
        job.startJob(mergedParams);
        setHasStartedFromAdvisor(true);
        toast({ title: "Advisor Parameters Loaded", description: `Started training with advisor-suggested parameters for ${mergedParams.modelName}.`, variant: "default" });
      }
    } else {
      setAdvisorParams(null);
    }
  }, [searchParams]);

  // Patch startJob to handle non-JSON responses
  const safeStartJob = async (params: any) => {
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      let data = null;
      try {
        data = await res.json();
      } catch {
        toast({ title: "Backend Error", description: "Failed to start training: backend did not return valid JSON.", variant: "destructive" });
        return;
      }
      if (!data || !data.job_id) {
        toast({ title: "Backend Error", description: "No job_id returned from backend.", variant: "destructive" });
        return;
      }
      job.setJobId(data.job_id);
      job.resume();
    } catch (e) {
      const err = e as Error;
      toast({ title: "Network Error", description: err.message, variant: "destructive" });
    }
  };

  const handleStopJob = () => {
    if (job.jobId) {
      job.stopJob(job.jobId);
    }
  };

  // Handler for ChartPanel button
  const handleStartAdvisorTraining = (params: any) => {
    setAdvisorParams(null);
    setPendingParams(params);
    safeStartJob(params);
    setHasStartedFromAdvisor(true);
    toast({ title: "Started Training", description: `Started training with HS-QNN suggested parameters for ${params.modelName}.`, variant: "default" });
  };

  const isJobActive = !!job.jobId && (job.isPolling || job.isLoading);

  return (
    <div className="w-full h-full flex flex-col relative items-center bg-[#0a0f1c] min-h-screen p-4 sm:p-6 lg:p-8">
      {/* Neon grid overlay */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="instrument-grid-overlay w-full h-full opacity-20" />
      </div>
      <div className="absolute top-4 right-8 z-30 flex gap-3 items-center">
        {advisorParams && (
          <Button
            size="sm"
            className="instrument-btn px-4 py-2 text-base font-bold border-2 border-[#ffe066] text-[#ffe066] bg-[#18181b] shadow-[0_0_8px_#ffe066] hover:bg-[#ffe066] hover:text-[#18181b] transition-all rounded-full"
            style={{ minWidth: 0, height: 38 }}
            onClick={() => handleStartAdvisorTraining(advisorParams)}
          >
            Start Training with HS-QNN Suggested Parameters
          </Button>
        )}
        <Button variant="outline" onClick={() => { setShowConfig(true); }} className="neon-btn" size="sm">Configure</Button>
      </div>
      <main className="w-full h-full max-w-screen-2xl mx-auto z-10 dashboard-grid">
        <div className="chart-panel-container">
            <ChartPanel metrics={job.metrics} />
        </div>
        <div className="terminal-panel-container">
            <TerminalPanel logs={job.logs} />
        </div>
        <div className="controls-panel-container">
            <ControlsPanel
              startJob={safeStartJob}
              stopJob={handleStopJob}
              isJobActive={isJobActive}
              jobId={job.jobId}
              jobStatus={job.jobStatus}
            />
        </div>
      </main>
      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent className="neon-card" style={{
          fontFamily: 'Share Tech Mono, Space Mono, VT323, monospace',
          background: '#0a0f1c',
          border: '3px solid #00ffe7',
          boxShadow: '0 0 64px 12px #00f0ffcc, 0 0 128px 24px #00aaff99',
          color: '#00ffe7',
          minWidth: '900px',
          minHeight: '600px',
          maxWidth: '96vw',
          maxHeight: '90vh',
          zIndex: 99999,
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          overflow: 'visible',
        }}>
          {/* Neon grid background */}
          <div style={{
            position: 'absolute',
            inset: 0,
            zIndex: 0,
            pointerEvents: 'none',
            background: 'repeating-linear-gradient(90deg, #00ffe71a 0 1px, transparent 1px 60px), repeating-linear-gradient(180deg, #00ffe71a 0 1px, transparent 1px 60px)',
            opacity: 0.12,
          }} />
          <DialogHeader className="z-10">
            <DialogTitle className="text-3xl font-bold text-neon-cyan mb-2 tracking-widest drop-shadow-[0_0_8px_#00ffe7]">Train Configuration</DialogTitle>
            <DialogDescription className="text-neon-blue/80 mb-6 text-lg tracking-wide">Configure your ZPE-enhanced neural network training job.</DialogDescription>
          </DialogHeader>
          {/* Tab state and content */}
          {(() => {
            const [activeTab, setActiveTab] = useState<'general'|'zpe'|'quantum'>('zpe');
            return (
              <>
                <div className="w-full flex-1 overflow-y-auto z-10" style={{paddingBottom: '3.5rem'}}>
                  {activeTab === 'general' && (
                    <div className="space-y-8 px-6 pt-2">
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Model Name</Label>
                          <Input type="text" value={pendingParams.modelName} onChange={e => setPendingParams((p:any) => ({...p, modelName: e.target.value}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Total Epochs</Label>
                          <Input type="number" value={pendingParams.totalEpochs} onChange={e => setPendingParams((p:any) => ({...p, totalEpochs: parseInt(e.target.value,10)||0}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Batch Size</Label>
                          <Input type="number" value={pendingParams.batchSize} onChange={e => setPendingParams((p:any) => ({...p, batchSize: parseInt(e.target.value,10)||0}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Learning Rate</Label>
                          <Input type="number" step="0.0001" value={pendingParams.learningRate} onChange={e => setPendingParams((p:any) => ({...p, learningRate: parseFloat(e.target.value)||0}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Weight Decay</Label>
                          <Input type="number" step="0.0001" value={pendingParams.weightDecay} onChange={e => setPendingParams((p:any) => ({...p, weightDecay: parseFloat(e.target.value)||0}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Mixup Alpha</Label>
                          <Input type="number" step="0.01" value={pendingParams.mixupAlpha} onChange={e => setPendingParams((p:any) => ({...p, mixupAlpha: parseFloat(e.target.value)||0}))} className="neon-input text-lg" />
                        </div>
                        <div className="col-span-2">
                          <Label className="text-neon-cyan text-lg mb-1 block">Base Config ID (Optional)</Label>
                          <Input type="text" value={pendingParams.baseConfigId || ''} onChange={e => setPendingParams((p:any) => ({...p, baseConfigId: e.target.value}))} className="neon-input text-lg" placeholder="e.g., config_123456" />
                        </div>
                      </div>
                    </div>
                  )}
                  {activeTab === 'zpe' && (
                    <div className="space-y-10 px-6 pt-2">
                      {['momentumParams', 'strengthParams', 'noiseParams', 'couplingParams'].map((paramKey) => (
                        <div key={paramKey} className="mb-2">
                          <Label className="text-neon-yellow text-2xl font-bold mb-4 block tracking-widest uppercase drop-shadow-[0_0_8px_#ffe066]">{paramKey.replace('Params', '')} <span className="text-base">(6 layers)</span></Label>
                          <div className="grid grid-cols-3 gap-8">
                            {[0,1,2,3,4,5].map(idx => (
                              <div key={idx} className="flex flex-col items-center">
                                <span className="text-neon-blue text-lg mb-2 font-bold">Layer {idx+1}</span>
                                <Input type="number" step="0.01" value={pendingParams[paramKey][idx]} onChange={e => setPendingParams((p:any) => {
                                  const arr = [...p[paramKey]];
                                  arr[idx] = parseFloat(e.target.value)||0;
                                  return { ...p, [paramKey]: arr };
                                })} className="neon-input w-28 text-center text-lg" />
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {activeTab === 'quantum' && (
                    <div className="space-y-8 px-6 pt-2">
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Quantum Mode</Label>
                          <Switch checked={pendingParams.quantumMode} onCheckedChange={checked => setPendingParams((p:any) => ({...p, quantumMode: checked}))} />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Quantum Circuit Size</Label>
                          <Input type="number" value={pendingParams.quantumCircuitSize} onChange={e => setPendingParams((p:any) => ({...p, quantumCircuitSize: parseInt(e.target.value,10)||0}))} className="neon-input text-lg" />
                        </div>
                        <div>
                          <Label className="text-neon-cyan text-lg mb-1 block">Label Smoothing</Label>
                          <Input type="number" step="0.01" value={pendingParams.labelSmoothing} onChange={e => setPendingParams((p:any) => ({...p, labelSmoothing: parseFloat(e.target.value)||0}))} className="neon-input text-lg" />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                {/* Tab navigation at the bottom */}
                <div className="w-full flex justify-center gap-8 mt-8 z-20">
                  <button className={`neon-btn text-xl px-8 py-3 rounded-full font-bold tracking-widest ${activeTab==='general' ? 'bg-[#00ffe7]/30 border-2 border-[#00ffe7] shadow-[0_0_16px_#00ffe7]' : 'bg-transparent border-2 border-[#00ffe7]/40'}`} onClick={() => setActiveTab('general')}>General</button>
                  <button className={`neon-btn text-xl px-8 py-3 rounded-full font-bold tracking-widest ${activeTab==='zpe' ? 'bg-[#ffe066]/20 border-2 border-[#ffe066] shadow-[0_0_16px_#ffe066]' : 'bg-transparent border-2 border-[#ffe066]/40'}`} onClick={() => setActiveTab('zpe')}>ZPE</button>
                  <button className={`neon-btn text-xl px-8 py-3 rounded-full font-bold tracking-widest ${activeTab==='quantum' ? 'bg-[#a0d4f5]/20 border-2 border-[#a0d4f5] shadow-[0_0_16px_#a0d4f5]' : 'bg-transparent border-2 border-[#a0d4f5]/40'}`} onClick={() => setActiveTab('quantum')}>Quantum</button>
                </div>
                {/* Action buttons */}
                <div className="flex justify-end gap-6 mt-8 w-full z-20">
                  <Button variant="outline" onClick={() => setShowConfig(false)} className="neon-btn text-lg px-8 py-3">Cancel</Button>
                  <Button variant="default" onClick={() => { setShowConfig(false); toast({ title: "Configuration updated", description: "Parameters will be used for the next training run." }); }} className="neon-btn text-lg px-8 py-3">Save</Button>
                  <Button variant="default" onClick={() => { setShowConfig(false); safeStartJob(pendingParams); }} className="neon-btn text-lg px-8 py-3">Start New Training Session</Button>
                </div>
              </>
            );
          })()}
        </DialogContent>
      </Dialog>
    </div>
  );
} 