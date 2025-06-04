
"use client";
import React, { useState, useEffect, useCallback, useRef } from "react";
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
import { Play, StopCircle, List, Zap, Settings, RefreshCw, AlertTriangle, CheckCircle, ExternalLink, SlidersHorizontal, Atom, Brain } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useRouter, useSearchParams } from "next/navigation";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

const trainingFormSchema = z.object({
  modelName: z.string().min(3, "Model name must be at least 3 characters"),
  totalEpochs: z.coerce.number().int().min(1).max(200),
  batchSize: z.coerce.number().int().min(8).max(256),
  learningRate: z.coerce.number().min(0.00001).max(0.1),
  weightDecay: z.coerce.number().min(0).max(0.1),
  momentumParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 momentum values"),
  strengthParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 strength values"),
  noiseParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 noise values"),
  couplingParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 coupling values"),
  quantumCircuitSize: z.coerce.number().int().min(4).max(64),
  labelSmoothing: z.coerce.number().min(0).max(0.5),
  quantumMode: z.boolean(),
  baseConfigId: z.string().nullable().optional(),
});

const defaultZPEParams = {
  momentum: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
  strength: [0.35, 0.33, 0.31, 0.60, 0.27, 0.50],
  noise: [0.3, 0.28, 0.26, 0.35, 0.22, 0.25],
  coupling: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7],
};

export default function TrainModelPage() {
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [jobsList, setJobsList] = useState<TrainingJobSummary[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pollingIntervalId, setPollingIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  
  const router = useRouter();
  const searchParams = useSearchParams();
  const prefillEffectRan = useRef(false);

  const defaultFormValues: TrainingParameters = {
    modelName: "ZPE-QuantumWeaver-V1",
    totalEpochs: 30,
    batchSize: 32,
    learningRate: 0.001,
    weightDecay: 0.0001,
    momentumParams: defaultZPEParams.momentum,
    strengthParams: defaultZPEParams.strength,
    noiseParams: defaultZPEParams.noise,
    couplingParams: defaultZPEParams.coupling,
    quantumCircuitSize: 32,
    labelSmoothing: 0.1,
    quantumMode: true,
    baseConfigId: undefined,
  };

  const { control, handleSubmit, reset, watch, formState: { errors }, setValue } = useForm<TrainingParameters>({
    resolver: zodResolver(trainingFormSchema),
    defaultValues: defaultFormValues,
  });

  useEffect(() => {
    if (prefillEffectRan.current || typeof window === 'undefined') return;

    const paramsToPrefill: Partial<TrainingParameters> = {};
    let hasDirectParams = false;

    for (const key in defaultFormValues) {
        if (searchParams.has(key)) {
            hasDirectParams = true;
            const value = searchParams.get(key);
            if (value !== null) {
                 try {
                    if (Array.isArray(defaultFormValues[key as keyof TrainingParameters])) {
                        (paramsToPrefill as any)[key] = JSON.parse(value);
                    } else if (typeof defaultFormValues[key as keyof TrainingParameters] === 'number') {
                        (paramsToPrefill as any)[key] = parseFloat(value);
                    } else if (typeof defaultFormValues[key as keyof TrainingParameters] === 'boolean') {
                        (paramsToPrefill as any)[key] = value === 'true';
                    } else {
                        (paramsToPrefill as any)[key] = value;
                    }
                 } catch (e) {
                    console.error(`Error parsing query param ${key}:`, e);
                    toast({ title: "Prefill Error", description: `Could not parse ${key} from URL. Using default.`, variant: "destructive" });
                 }
            }
        }
    }
    
    const loadAndPrefill = async () => {
      if (hasDirectParams) {
          toast({ title: "Pre-filling form...", description: "Loading parameters from URL." });
          const validatedParams = trainingFormSchema.safeParse({...defaultFormValues, ...paramsToPrefill});
          if (validatedParams.success) {
              reset(validatedParams.data);
              toast({ title: "Form Pre-filled", description: "Parameters loaded from URL." });
          } else {
              console.error("Direct prefill validation error:", validatedParams.error);
              toast({ title: "Prefill Validation Error", description: "Some parameters from URL were invalid. Defaults used where necessary.", variant: "destructive" });
              reset({...defaultFormValues, ...paramsToPrefill}); 
          }
      } else {
        const prefillJobId = searchParams.get("prefill");
        if (prefillJobId) {
          toast({ title: "Pre-filling form...", description: `Loading parameters from job ${prefillJobId.slice(-6)}` });
          try {
            const response = await fetch(`${API_BASE_URL}/status/${prefillJobId}`);
            if (!response.ok) {
              throw new Error(`Failed to fetch job details for prefill: ${response.statusText}`);
            }
            const jobToPrefill: TrainingJob = await response.json();
            
            const paramsWithNewNameAndBaseId = {
                ...defaultFormValues, // Start with defaults to ensure all fields are present
                ...jobToPrefill.parameters, // Overlay with job parameters
                modelName: `${jobToPrefill.parameters.modelName}_retrain_${Date.now().toString().slice(-4)}`,
                baseConfigId: prefillJobId,
            };
            
            const validatedParams = trainingFormSchema.safeParse(paramsWithNewNameAndBaseId);
            if (validatedParams.success) {
                reset(validatedParams.data);
                toast({ title: "Form Pre-filled", description: `Loaded parameters from job ${jobToPrefill.parameters.modelName}. Model name updated.` });
            } else {
                console.error("Job prefill validation error:", validatedParams.error);
                let errorMessages = "Validation errors: ";
                validatedParams.error.errors.forEach(err => {
                    errorMessages += `${err.path.join('.')}: ${err.message}. `;
                });
                throw new Error(`Parameters from job ${prefillJobId.slice(-6)} are not valid. ${errorMessages}`);
            }
          } catch (e: any) {
            console.error("Error pre-filling form from job ID:", e);
            toast({ title: "Pre-fill Failed", description: e.message, variant: "destructive" });
          }
        }
      }
      prefillEffectRan.current = true;
    };

    loadAndPrefill();

  }, [searchParams, reset, setValue]);


  const fetchJobsList = useCallback(async () => {
    setIsLoadingJobs(true);
    try {
      const response = await fetch(API_BASE_URL + '/jobs?limit=20');
      if (!response.ok) throw new Error("Failed to fetch jobs list");
      const data = await response.json();
      setJobsList((data.jobs || []).sort((a: TrainingJobSummary, b: TrainingJobSummary) => new Date(b.start_time || 0).getTime() - new Date(a.start_time || 0).getTime()));
    } catch (error: any) {
      toast({ title: "Error fetching jobs", description: error.message, variant: "destructive" });
    } finally {
      setIsLoadingJobs(false);
    }
  }, []);

  const pollJobStatus = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) {
        if (pollingIntervalId) clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      const jobData: TrainingJob = await response.json();
      setActiveJob(jobData);
      
      if (jobData.status === "running" || jobData.status === "completed") {
        setChartData(prev => {
          const newPoint = { 
            epoch: jobData.current_epoch, 
            accuracy: jobData.accuracy, 
            loss: jobData.loss,
            avg_zpe: jobData.zpe_effects && jobData.zpe_effects.length > 0 ? jobData.zpe_effects.reduce((a,b)=>a+b,0) / (jobData.zpe_effects.length) : 0
          };
          const existingPointIndex = prev.findIndex(p => p.epoch === newPoint.epoch);
          if (existingPointIndex > -1) {
            const updatedPrev = [...prev];
            updatedPrev[existingPointIndex] = newPoint;
            return updatedPrev;
          }
          return [...prev, newPoint].sort((a,b) => a.epoch - b.epoch);
        });
      }

      if (jobData.status === "completed" || jobData.status === "failed" || jobData.status === "stopped") {
        if (pollingIntervalId) clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
        fetchJobsList(); 
        toast({ title: `Job ${jobData.status}`, description: `Job ${jobId.replace('zpe_job_','')} (${jobData.parameters.modelName}) finished with status: ${jobData.status}. Accuracy: ${jobData.accuracy.toFixed(2)}%` });
      }
    } catch (error: any) {
      toast({ title: "Error polling job status", description: error.message, variant: "destructive" });
      if (pollingIntervalId) clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
  }, [pollingIntervalId, fetchJobsList]);

  useEffect(() => {
    fetchJobsList();
    // If there's an active job ID from a previous session or navigation, try to poll it
    const currentJobId = activeJob?.job_id;
    if (currentJobId && (activeJob?.status === 'running' || activeJob?.status === 'pending')) {
        const intervalId = setInterval(() => pollJobStatus(currentJobId), 2000);
        setPollingIntervalId(intervalId);
    }
    return () => {
      if (pollingIntervalId) clearInterval(pollingIntervalId);
    };
  }, [fetchJobsList]); // Removed pollingIntervalId from deps to avoid re-triggering clearInterval on its own change

  const onSubmit = async (data: TrainingParameters) => {
    setIsSubmitting(true);
    setChartData([]); 
    try {
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error(`Failed to start training job. Server responded with ${response.status}`);
      const result = await response.json();
      if (!result.job_id) throw new Error("Backend did not return a job_id.");
      
      setActiveJob({
        job_id: result.job_id,
        status: "pending",
        current_epoch: 0,
        total_epochs: data.totalEpochs,
        accuracy: 0,
        loss: 0,
        zpe_effects: Array(6).fill(0), // Initialize with default
        log_messages: [`Job ${result.job_id} submitted.`],
        parameters: data,
      });
      const intervalId = setInterval(() => pollJobStatus(result.job_id), 2000);
      setPollingIntervalId(intervalId);
      fetchJobsList(); 
      toast({ title: "Training Started", description: `Job ID: ${result.job_id.replace('zpe_job_','')}` });
    } catch (error: any) {
      toast({ title: "Error starting training", description: error.message, variant: "destructive" });
      setActiveJob(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStopJob = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/stop/${jobId}`, { method: "POST" });
      if (!response.ok) throw new Error("Failed to stop job");
      toast({ title: "Stop Request Sent", description: `Requested stop for job ${jobId.replace('zpe_job_','')}` });
    } catch (error: any) {
      toast({ title: "Error stopping job", description: error.message, variant: "destructive" });
    }
  };
  
  const handleViewJobDetails = async (jobId: string) => {
    if (pollingIntervalId) clearInterval(pollingIntervalId); 
    setChartData([]); 
    const jobSummary = jobsList.find(j => j.job_id === jobId);
    
    setIsSubmitting(true); // Use isSubmitting to show loading state for details fetch
    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) throw new Error("Failed to fetch job details");
      const data: TrainingJob = await response.json();
      setActiveJob(data);

      // Reconstruct chart data from logs if job is not running/pending
      if (data.status !== "running" && data.status !== "pending" && data.log_messages) {
        const parsedChartData = data.log_messages
          .map(log => {
            const epochMatch = log.match(/E(\d+)\s+B\d+\/\d+\s+L:\s*([\d.]+)/); // Training batch log
            const endEpochMatch = log.match(/E(\d+) END - TrainL: [\d.]+, ValAcc: ([\d.]+)%, ValL: ([\d.]+)/); // End of epoch log
            const zpeMatch = log.match(/ZPE: \[([,\d\s.]+)\]/); // ZPE log

            let epochData: any = {};

            if (endEpochMatch) {
                epochData.epoch = parseInt(endEpochMatch[1]);
                epochData.accuracy = parseFloat(endEpochMatch[2]);
                epochData.loss = parseFloat(endEpochMatch[3]);
            } else if (epochMatch) {
                // Use training batch log if end of epoch log is not available for that epoch
                // This might be less accurate for final epoch representation
                epochData.epoch = parseInt(epochMatch[1]);
                epochData.loss = parseFloat(epochMatch[2]);
                 // Accuracy typically from validation, so might be undefined here
            }
            
            if (zpeMatch && epochData.epoch) { // Add ZPE if epoch data was found
                const zpeValues = zpeMatch[1].split(',').map(s => parseFloat(s.trim()));
                epochData.avg_zpe = zpeValues.reduce((a, b) => a + b, 0) / zpeValues.length;
            }
            
            return epochData.epoch ? epochData : null;
          })
          .filter(Boolean)
          .reduce((acc: any[], current: any) => { // Deduplicate by epoch, keeping last entry (likely most complete)
            const existing = acc.find(item => item.epoch === current.epoch);
            if (existing) {
                Object.assign(existing, current); // Merge, new data overrides old for same epoch
            } else {
                acc.push(current);
            }
            return acc;
          }, [])
          .sort((a,b) => a!.epoch - b!.epoch);

          if (parsedChartData.length > 0) {
            setChartData(parsedChartData);
          } else if (data.status === "completed" || data.status === "stopped") { 
             // If no logs parsed, use the final job status for a single point
             setChartData([{ 
                epoch: data.current_epoch, 
                accuracy: data.accuracy, 
                loss: data.loss, 
                avg_zpe: data.zpe_effects && data.zpe_effects.length > 0 ? data.zpe_effects.reduce((a,b)=>a+b,0) / data.zpe_effects.length : 0
             }]);
          }
      } else if (data.status === "running" || data.status === "pending") {
        // If job is active, start polling
        pollJobStatus(jobId); 
        const intervalId = setInterval(() => pollJobStatus(jobId), 2000);
        setPollingIntervalId(intervalId);
      }
    } catch (error: any) {
      toast({ title: "Error fetching job details", description: error.message, variant: "destructive"});
      setActiveJob(null); // Clear active job on error
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderParamArrayFields = (paramName: "momentumParams" | "strengthParams" | "noiseParams" | "couplingParams", labelPrefix: string, Icon: React.ElementType) => (
    <div className="space-y-3 p-3 border rounded-md bg-muted/30">
      <Label className="text-base flex items-center gap-2"><Icon className="h-4 w-4 text-primary"/>{labelPrefix} <span className="text-xs text-muted-foreground">(6 layers)</span></Label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-3 gap-y-2">
        {(watch(paramName) || Array(6).fill(0)).map((_, index) => (
          <div key={index} className="space-y-1">
            <Label htmlFor={`${paramName}.${index}`} className="text-xs text-muted-foreground">Layer {index + 1}</Label>
            <Controller
              name={`${paramName}.${index}` as any}
              control={control}
              render={({ field, fieldState }) => (
                <>
                  <Input
                    {...field}
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    className="h-8 text-sm"
                    value={typeof field.value === 'number' ? field.value : ''}
                    onChange={e => field.onChange(e.target.value === '' ? '' : parseFloat(e.target.value))}
                  />
                  {fieldState.error && <p className="text-xs text-destructive">{fieldState.error.message}</p>}
                </>
              )}
            />
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="container mx-auto p-4 md:p-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between space-y-2 md:space-y-0 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary">ZPE Model Training Orchestrator</h1>
          <p className="text-muted-foreground">Configure, initiate, and monitor ZPE-enhanced neural network training jobs.</p>
        </div>
         <Button onClick={fetchJobsList} variant="outline" disabled={isLoadingJobs}>
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoadingJobs ? 'animate-spin' : ''}`} /> Refresh Job List
        </Button>
      </div>
      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Settings className="h-5 w-5 text-primary" /> Training Configuration</CardTitle>
            <CardDescription>Define parameters for your ZPE model training job.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <Tabs defaultValue="general">
                <TabsList className="grid w-full grid-cols-3 mb-4">
                  <TabsTrigger value="general">General</TabsTrigger>
                  <TabsTrigger value="zpe">ZPE</TabsTrigger>
                  <TabsTrigger value="quantum">Quantum</TabsTrigger>
                </TabsList>
                <TabsContent value="general" className="space-y-3">
                  <FieldController control={control} name="modelName" label="Model Name" placeholder="e.g., MyZPEModel_V1" />
                  <FieldController control={control} name="totalEpochs" label="Total Epochs" type="number" min="1" />
                  <FieldController control={control} name="batchSize" label="Batch Size" type="number" min="1" />
                  <FieldController control={control} name="learningRate" label="Learning Rate" type="number" step="0.0001" />
                  <FieldController control={control} name="weightDecay" label="Weight Decay" type="number" step="0.00001" />
                  <FieldController control={control} name="labelSmoothing" label="Label Smoothing" type="number" step="0.01" />
                </TabsContent>
                <TabsContent value="zpe" className="space-y-3">
                  {renderParamArrayFields("momentumParams", "Momentum Parameters", SlidersHorizontal)}
                  {renderParamArrayFields("strengthParams", "Strength Parameters", Zap)}
                  {renderParamArrayFields("noiseParams", "Noise Parameters", Waves)}
                  {renderParamArrayFields("couplingParams", "Coupling Parameters", Brain)}
                </TabsContent>
                <TabsContent value="quantum" className="space-y-3">
                  <FieldControllerSwitch control={control} name="quantumMode" label="Enable Quantum Mode" />
                  <FieldController control={control} name="quantumCircuitSize" label="Quantum Circuit Size (Qubits)" type="number" min="1" />
                </TabsContent>
              </Tabs>
               <div className="space-y-1 pt-2">
                <Label htmlFor="baseConfigId">Base Config ID (Optional for HNN)</Label>
                <Controller
                    name="baseConfigId"
                    control={control}
                    render={({ field, fieldState }) => (
                        <>
                            <Input
                                {...field}
                                id="baseConfigId"
                                placeholder="Previous job ID to build upon"
                                value={field.value || ''}
                                onChange={(e) => field.onChange(e.target.value || null)} // Handle empty string as null
                                className={fieldState.error ? "border-destructive" : ""}
                            />
                            {fieldState.error && <p className="text-xs text-destructive">{fieldState.error.message}</p>}
                        </>
                    )}
                />
                <p className="text-xs text-muted-foreground">If continuing/evolving from a previous job (for HS-QNN).</p>
              </div>
              <Button type="submit" className="w-full mt-6" disabled={isSubmitting || (activeJob?.status === "running" || activeJob?.status === "pending")}>
                {isSubmitting ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                Start Training
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Zap className="h-5 w-5 text-primary"/> Training Monitor</CardTitle>
            {activeJob ? (
              <CardDescription>Status for Job ID: <span className="font-mono text-xs">{activeJob.job_id.replace('zpe_job_','')}</span> ({activeJob.parameters.modelName})</CardDescription>
            ) : (
              <CardDescription>No active training job. Start one or select from history.</CardDescription>
            )}
          </CardHeader>
          <CardContent>
            {activeJob ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Badge variant={
                    activeJob.status === "running" ? "default" :
                    activeJob.status === "completed" ? "default" : 
                    activeJob.status === "failed" || activeJob.status === "stopped" ? "destructive" : "secondary"
                  }
                  className={cn("font-semibold",
                    activeJob.status === "completed" ? "bg-green-500 hover:bg-green-600 text-primary-foreground" : 
                    activeJob.status === "running" ? "bg-blue-500 hover:bg-blue-600 text-primary-foreground animate-pulse" : ""
                  )}
                  >
                    {activeJob.status.toUpperCase()}
                  </Badge>
                  {(activeJob.status === "running" || activeJob.status === "pending") && (
                    <Button variant="destructive" size="sm" onClick={() => handleStopJob(activeJob.job_id)}>
                      <StopCircle className="mr-2 h-4 w-4" /> Stop Job
                    </Button>
                  )}
                </div>
                <Progress value={(activeJob.current_epoch / activeJob.total_epochs) * 100} className="w-full h-3" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <MetricDisplay label="Epoch" value={`${activeJob.current_epoch}/${activeJob.total_epochs}`} />
                  <MetricDisplay label="Accuracy" value={`${activeJob.accuracy.toFixed(2)}%`} />
                  <MetricDisplay label="Loss" value={`${activeJob.loss.toFixed(4)}`} />
                  <MetricDisplay label="Avg ZPE Effect" value={`${(activeJob.zpe_effects && activeJob.zpe_effects.length > 0 ? activeJob.zpe_effects.reduce((a,b)=>a+b,0) / activeJob.zpe_effects.length : 0).toFixed(3)}`} />
                </div>
                
                <div className="h-[250px] mt-4 bg-muted/30 p-2 rounded-md">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" />
                      <XAxis dataKey="epoch" name="Epoch" stroke="hsl(var(--muted-foreground))" fontSize={10} />
                      <YAxis yAxisId="left" name="Accuracy" stroke="hsl(var(--primary))" fontSize={10} domain={[70,100]} tickFormatter={(val) => `${val}%`}/>
                      <YAxis yAxisId="right" name="Loss" orientation="right" stroke="hsl(var(--destructive))" fontSize={10} domain={[0, 'auto']}/>
                      <Tooltip wrapperClassName="text-xs rounded-md shadow-lg !bg-popover !border-border" labelClassName="font-bold" />
                      <Legend wrapperStyle={{fontSize: "10px"}}/>
                      <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="hsl(var(--primary))" name="Accuracy" dot={false} strokeWidth={2} />
                      <Line yAxisId="right" type="monotone" dataKey="loss" stroke="hsl(var(--destructive))" name="Loss" dot={false} strokeWidth={2} />
                      <Line yAxisId="left" type="monotone" dataKey="avg_zpe" stroke="hsl(var(--accent))" name="Avg ZPE" dot={false} strokeDasharray="3 3" strokeWidth={1.5}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <Label className="text-sm">Training Logs</Label>
                <ScrollArea className="h-40 w-full rounded-md border p-2 bg-muted/30">
                  {activeJob.log_messages && activeJob.log_messages.slice().reverse().map((log, index) => (
                    <p key={index} className="text-xs font-mono text-muted-foreground leading-relaxed">{log}</p>
                  ))}
                   {!activeJob.log_messages && <p className="text-xs font-mono text-muted-foreground">No logs yet.</p>}
                </ScrollArea>
              </div>
            ) : (
              <div className="text-center py-10 text-muted-foreground">
                <Zap className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>Select a job from history or start a new training session to monitor progress.</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><List className="h-5 w-5 text-primary"/> Training Job History</CardTitle>
            <CardDescription>View past and ongoing training jobs. Click "View" to load details into the monitor.</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingJobs && <div className="flex justify-center py-10"><RefreshCw className="h-8 w-8 animate-spin text-primary" /></div>}
            {!isLoadingJobs && jobsList.length === 0 && (
                <div className="text-center py-10 text-muted-foreground">
                    <List className="mx-auto h-12 w-12 mb-4 opacity-50" />
                    No training jobs found. Start a new one using the configuration panel.
                </div>
            )}
            {!isLoadingJobs && jobsList.length > 0 && (
              <ScrollArea className="h-[400px]">
                <Table>
                  <TableHeader className="sticky top-0 bg-card z-10">
                    <TableRow>
                      <TableHead>Job ID</TableHead>
                      <TableHead>Model Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>Epochs</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {jobsList.map(job => (
                      <TableRow key={job.job_id} className={cn("cursor-pointer hover:bg-muted/50", activeJob?.job_id === job.job_id ? "bg-primary/10" : "")} onClick={() => handleViewJobDetails(job.job_id)}>
                        <TableCell className="font-mono text-xs">{job.job_id.replace('zpe_job_', '')}</TableCell>
                        <TableCell className="font-medium">{job.model_name}</TableCell>
                        <TableCell>
                          <Badge variant={
                            job.status === "running" ? "default" :
                            job.status === "completed" ? "default" :
                            job.status === "failed" || job.status === "stopped" ? "destructive" : "secondary"
                          }
                          className={cn("font-semibold text-xs", job.status === "completed" ? "bg-green-500 hover:bg-green-600 text-primary-foreground" : job.status === "running" ? "bg-blue-500 hover:bg-blue-600 text-primary-foreground animate-pulse" : "")}
                          >{job.status}</Badge>
                        </TableCell>
                        <TableCell>{job.accuracy > 0 ? `${job.accuracy.toFixed(2)}%` : '-'}</TableCell>
                        <TableCell>{`${job.current_epoch}/${job.total_epochs}`}</TableCell>
                        <TableCell className="text-right">
                          <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); handleViewJobDetails(job.job_id); }}>
                            <ExternalLink className="mr-1 h-3 w-3"/>View Details
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

interface FieldControllerProps<TFieldValues extends FieldValues, TName extends FieldPath<TFieldValues>> {
  control: Control<TFieldValues>;
  name: TName;
  label: string;
  type?: string;
  placeholder?: string;
  min?: string | number;
  step?: string | number;
}

const FieldController = <TFieldValues extends FieldValues = TrainingParameters>({
  control,
  name,
  label,
  type = "text",
  placeholder,
  min,
  step,
}: FieldControllerProps<TFieldValues, FieldPath<TFieldValues>>) => (
  <div className="space-y-1">
    <Label htmlFor={name as string} className="text-sm">{label}</Label>
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => {
        const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            if (type === 'number') {
                const value = e.target.value;
                field.onChange(value === '' ? '' : parseFloat(value));
            } else {
                field.onChange(e.target.value);
            }
        };
        return (
            <>
                <Input
                    {...field}
                    id={name as string}
                    type={type}
                    placeholder={placeholder}
                    min={min}
                    step={step}
                    value={(type === 'number' && (typeof field.value === 'number' && !isNaN(field.value))) ? field.value : (field.value || '')}
                    onChange={handleChange}
                    className={cn("h-9 text-sm", fieldState.error ? "border-destructive focus-visible:ring-destructive" : "")}
                />
                {fieldState.error && <p className="text-xs text-destructive mt-1">{fieldState.error.message}</p>}
            </>
        );
    }}
    />
  </div>
);

interface FieldControllerSwitchProps<TFieldValues extends FieldValues, TName extends FieldPath<TFieldValues>> {
  control: Control<TFieldValues>;
  name: TName;
  label: string;
}

const FieldControllerSwitch = <TFieldValues extends FieldValues = TrainingParameters>({
  control,
  name,
  label,
}: FieldControllerSwitchProps<TFieldValues, FieldPath<TFieldValues>>) => (
  <div className="flex items-center justify-between space-x-2 border p-3 rounded-md bg-muted/30">
    <Label htmlFor={name as string} className="text-sm cursor-pointer">{label}</Label>
    <Controller
      name={name}
      control={control}
      render={({ field }) => (
        <Switch
          id={name as string}
          checked={field.value as boolean}
          onCheckedChange={field.onChange}
        />
      )}
    />
  </div>
);

const MetricDisplay = ({ label, value }: { label: string; value: string | number }) => (
  <div className="bg-muted/50 p-3 rounded-md text-center ring-1 ring-inset ring-border/70">
    <p className="text-xs text-muted-foreground">{label}</p>
    <p className="font-semibold text-lg font-mono">{value}</p>
  </div>
);

