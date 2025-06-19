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

// ...all the rest of your code from TrainModelPage, unchanged...

export default function TrainModelClient() {
  const job = useJobStatusPolling();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [hasStartedFromAdvisor, setHasStartedFromAdvisor] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [pendingParams, setPendingParams] = useState<any>(defaultZPEParams);

  useEffect(() => {
    if (hasStartedFromAdvisor) return;
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
      job.startJob(mergedParams);
      setHasStartedFromAdvisor(true);
      toast({ title: "Advisor Parameters Loaded", description: `Started training with advisor-suggested parameters for ${mergedParams.modelName}.`, variant: "default" });
    }
  }, [searchParams, job, hasStartedFromAdvisor]);

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

  const isJobActive = !!job.jobId && job.isPolling;

  return (
    <div className="w-full h-full flex flex-col relative">
      <div className="absolute top-4 right-8 z-30 flex gap-3">
        <Button variant="outline" onClick={() => { console.log('Configure modal open'); setShowConfig(true); }} className="neon-btn">Configure</Button>
      </div>
      <InstrumentCard
        jobId={job.jobId}
        jobStatus={job.jobStatus}
        logs={job.logs}
        metrics={job.metrics}
        isPolling={job.isPolling}
        isLoading={job.isLoading}
        startJob={job.startJob}
        stopJob={job.stopJob}
        freeze={job.freeze}
        resume={job.resume}
        clearLogs={job.clearLogs}
        exportLogs={job.exportLogs}
        exportMetrics={job.exportMetrics}
        setJobId={job.setJobId}
      />
      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent className="neon-card" style={{
          fontFamily: 'Share Tech Mono, Space Mono, VT323, monospace',
          background: '#18181b',
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
        }}>
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-neon-cyan">Train Configuration</DialogTitle>
            <DialogDescription className="text-neon-blue/80 mb-4">Configure your ZPE-enhanced neural network training job.</DialogDescription>
          </DialogHeader>
          <Tabs defaultValue="general" className="w-full">
            <TabsList className="flex w-full mb-4 neon-btn">
              <TabsTrigger value="general" className="flex-1 neon-btn">General</TabsTrigger>
              <TabsTrigger value="zpe" className="flex-1 neon-btn">ZPE</TabsTrigger>
              <TabsTrigger value="quantum" className="flex-1 neon-btn">Quantum</TabsTrigger>
            </TabsList>
            <TabsContent value="general" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-neon-cyan">Model Name</Label>
                  <Input type="text" value={pendingParams.modelName} onChange={e => setPendingParams((p:any) => ({...p, modelName: e.target.value}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Total Epochs</Label>
                  <Input type="number" value={pendingParams.totalEpochs} onChange={e => setPendingParams((p:any) => ({...p, totalEpochs: parseInt(e.target.value,10)||0}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Batch Size</Label>
                  <Input type="number" value={pendingParams.batchSize} onChange={e => setPendingParams((p:any) => ({...p, batchSize: parseInt(e.target.value,10)||0}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Learning Rate</Label>
                  <Input type="number" step="0.0001" value={pendingParams.learningRate} onChange={e => setPendingParams((p:any) => ({...p, learningRate: parseFloat(e.target.value)||0}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Weight Decay</Label>
                  <Input type="number" step="0.0001" value={pendingParams.weightDecay} onChange={e => setPendingParams((p:any) => ({...p, weightDecay: parseFloat(e.target.value)||0}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Mixup Alpha</Label>
                  <Input type="number" step="0.01" value={pendingParams.mixupAlpha} onChange={e => setPendingParams((p:any) => ({...p, mixupAlpha: parseFloat(e.target.value)||0}))} className="neon-input" />
                </div>
                <div className="col-span-2">
                  <Label className="text-neon-cyan">Base Config ID (Optional)</Label>
                  <Input type="text" value={pendingParams.baseConfigId || ''} onChange={e => setPendingParams((p:any) => ({...p, baseConfigId: e.target.value}))} className="neon-input" placeholder="e.g., config_123456" />
                </div>
              </div>
            </TabsContent>
            <TabsContent value="zpe" className="space-y-6">
              {['momentumParams', 'strengthParams', 'noiseParams', 'couplingParams'].map((paramKey) => (
                <div key={paramKey} className="mb-2">
                  <Label className="text-neon-yellow text-lg font-bold mb-2 block">{paramKey.replace('Params', '')} <span className="text-xs">(6 layers)</span></Label>
                  <div className="grid grid-cols-3 gap-3">
                    {[0,1,2,3,4,5].map(idx => (
                      <div key={idx} className="flex flex-col items-center">
                        <span className="text-neon-blue text-xs mb-1">Layer {idx+1}</span>
                        <Input type="number" step="0.01" value={pendingParams[paramKey][idx]} onChange={e => setPendingParams((p:any) => {
                          const arr = [...p[paramKey]];
                          arr[idx] = parseFloat(e.target.value)||0;
                          return { ...p, [paramKey]: arr };
                        })} className="neon-input w-20 text-center" />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </TabsContent>
            <TabsContent value="quantum" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-neon-cyan">Quantum Mode</Label>
                  <Switch checked={pendingParams.quantumMode} onCheckedChange={checked => setPendingParams((p:any) => ({...p, quantumMode: checked}))} />
                </div>
                <div>
                  <Label className="text-neon-cyan">Quantum Circuit Size</Label>
                  <Input type="number" value={pendingParams.quantumCircuitSize} onChange={e => setPendingParams((p:any) => ({...p, quantumCircuitSize: parseInt(e.target.value,10)||0}))} className="neon-input" />
                </div>
                <div>
                  <Label className="text-neon-cyan">Label Smoothing</Label>
                  <Input type="number" step="0.01" value={pendingParams.labelSmoothing} onChange={e => setPendingParams((p:any) => ({...p, labelSmoothing: parseFloat(e.target.value)||0}))} className="neon-input" />
                </div>
              </div>
            </TabsContent>
          </Tabs>
          <div className="flex justify-end gap-3 mt-8">
            <Button variant="outline" onClick={() => setShowConfig(false)} className="neon-btn">Cancel</Button>
            <Button variant="default" onClick={() => { setShowConfig(false); toast({ title: "Configuration updated", description: "Parameters will be used for the next training run." }); }} className="neon-btn">Save</Button>
            <Button variant="default" onClick={() => { setShowConfig(false); safeStartJob(pendingParams); }} className="neon-btn">Start New Training Session</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
} 