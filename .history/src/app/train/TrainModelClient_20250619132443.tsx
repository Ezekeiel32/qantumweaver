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
      {/* Fallback debug modal */}
      {showConfig && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(0,0,0,0.85)',
          zIndex: 9999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{ background: 'white', color: 'black', padding: 40, borderRadius: 16, fontSize: 32, textAlign: 'center' }}>
            FALLBACK MODAL<br />
            If you see this, React state and rendering are working.<br />
            <button style={{ marginTop: 32, fontSize: 24 }} onClick={() => setShowConfig(false)}>Close</button>
          </div>
        </div>
      )}
      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent className="max-w-3xl w-full">
          <div style={{ color: 'red', fontSize: '2rem', textAlign: 'center', marginBottom: '2rem' }}>
            TEST MODAL: If you see this, the modal is rendering.
          </div>
          <DialogHeader>
            <DialogTitle>Training Configuration</DialogTitle>
            <DialogDescription>Edit all training parameters below. Changes will apply to the next training run.</DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <Label>Total Epochs</Label>
              <Input type="number" value={pendingParams.totalEpochs} onChange={e => setPendingParams((p:any) => ({...p, totalEpochs: parseInt(e.target.value,10)||0}))} />
            </div>
            <div>
              <Label>Batch Size</Label>
              <Input type="number" value={pendingParams.batchSize} onChange={e => setPendingParams((p:any) => ({...p, batchSize: parseInt(e.target.value,10)||0}))} />
            </div>
            <div>
              <Label>Learning Rate</Label>
              <Input type="number" step="0.0001" value={pendingParams.learningRate} onChange={e => setPendingParams((p:any) => ({...p, learningRate: parseFloat(e.target.value)||0}))} />
            </div>
            <div>
              <Label>Weight Decay</Label>
              <Input type="number" step="0.0001" value={pendingParams.weightDecay} onChange={e => setPendingParams((p:any) => ({...p, weightDecay: parseFloat(e.target.value)||0}))} />
            </div>
            <div>
              <Label>Quantum Circuit Size</Label>
              <Input type="number" value={pendingParams.quantumCircuitSize} onChange={e => setPendingParams((p:any) => ({...p, quantumCircuitSize: parseInt(e.target.value,10)||0}))} />
            </div>
            <div>
              <Label>Label Smoothing</Label>
              <Input type="number" step="0.01" value={pendingParams.labelSmoothing} onChange={e => setPendingParams((p:any) => ({...p, labelSmoothing: parseFloat(e.target.value)||0}))} />
            </div>
            <div>
              <Label>Model Name</Label>
              <Input type="text" value={pendingParams.modelName} onChange={e => setPendingParams((p:any) => ({...p, modelName: e.target.value}))} />
            </div>
            <div>
              <Label>Quantum Mode</Label>
              <Switch checked={pendingParams.quantumMode} onCheckedChange={checked => setPendingParams((p:any) => ({...p, quantumMode: checked}))} />
            </div>
            <div>
              <Label>Mixup Alpha</Label>
              <Input type="number" step="0.01" value={pendingParams.mixupAlpha} onChange={e => setPendingParams((p:any) => ({...p, mixupAlpha: parseFloat(e.target.value)||0}))} />
            </div>
          </div>
          <div className="mt-4">
            <Label>ZPE Parameters (arrays of 6, comma separated)</Label>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <div>
                <Label>Momentum</Label>
                <Input type="text" value={pendingParams.momentumParams.join(", ")} onChange={e => setPendingParams((p:any) => ({...p, momentumParams: e.target.value.split(",").map((v:string)=>parseFloat(v.trim())||0)}))} />
              </div>
              <div>
                <Label>Strength</Label>
                <Input type="text" value={pendingParams.strengthParams.join(", ")} onChange={e => setPendingParams((p:any) => ({...p, strengthParams: e.target.value.split(",").map((v:string)=>parseFloat(v.trim())||0)}))} />
              </div>
              <div>
                <Label>Noise</Label>
                <Input type="text" value={pendingParams.noiseParams.join(", ")} onChange={e => setPendingParams((p:any) => ({...p, noiseParams: e.target.value.split(",").map((v:string)=>parseFloat(v.trim())||0)}))} />
              </div>
              <div>
                <Label>Coupling</Label>
                <Input type="text" value={pendingParams.couplingParams.join(", ")} onChange={e => setPendingParams((p:any) => ({...p, couplingParams: e.target.value.split(",").map((v:string)=>parseFloat(v.trim())||0)}))} />
              </div>
            </div>
          </div>
          <div className="flex justify-end gap-3 mt-6">
            <Button variant="outline" onClick={() => setShowConfig(false)}>Cancel</Button>
            <Button variant="default" onClick={() => { setShowConfig(false); toast({ title: "Configuration updated", description: "Parameters will be used for the next training run." }); }}>Save</Button>
            <Button variant="default" onClick={() => { setShowConfig(false); safeStartJob(pendingParams); }}>Start Training</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
} 