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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

const trainingFormSchema = z.object({
  modelName: z.string().min(3, "Model name must be at least 3 characters"),
  totalEpochs: z.coerce.number().int().min(1).max(200),
  batchSize: z.coerce.number().int().min(8).max(256),
  learningRate: z.coerce.number().min(0.00001).max(0.1),
  weightDecay: z.coerce.number().min(1e-9).max(0.1),
  momentumParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 momentum values"),
  strengthParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 strength values"),
  noiseParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 noise values"),
  couplingParams: z.array(z.coerce.number().min(0).max(1)).length(6, "Must have 6 coupling values"),
  quantumCircuitSize: z.coerce.number().int().min(4).max(64),
  labelSmoothing: z.coerce.number().min(0).max(0.5),
  quantumMode: z.boolean(),
  baseConfigId: z.string().nullable().optional(),
  mixupAlpha: z.coerce.number().min(0).max(1),
});

interface FieldControllerProps<TName extends FieldPath<TrainingParameters>> {
  control: Control<TrainingParameters>;
  name: TName;
  label: string;
  type?: string;
  placeholder?: string;
  min?: string | number;
  step?: string | number;
 max?: string | number;
}

const FieldController = <TFieldValues extends FieldValues = TrainingParameters>({
  control,
  name,
  label,
  type = "text",
  placeholder,
  min,
  step,
 max,
}: FieldControllerProps<FieldPath<TrainingParameters>>) => (
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
 max={max}
              value={`${field.value ?? ''}`}
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

// Utility to save ZPE stats/logs to persistent storage for advisor
function saveZpeStatsToDataset(jobId: string, logs: string[], zpeHistory: Array<{ epoch: number; zpe_effects: number[] }>) {
  try {
    const key = 'zpeStatsDataset';
    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    // Avoid duplicates by jobId and epoch
    const newEntries = zpeHistory.map(h => ({ jobId, epoch: h.epoch, zpe_effects: h.zpe_effects }));
    const filtered = newEntries.filter(entry => !existing.some((e: any) => e.jobId === entry.jobId && e.epoch === entry.epoch));
    if (filtered.length > 0) {
      localStorage.setItem(key, JSON.stringify([...existing, ...filtered]));
    }
  } catch (e) {
    // Ignore storage errors
  }
}

export default function TrainModelPage() {
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [jobsList, setJobsList] = useState<TrainingJobSummary[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingRef = useRef<boolean>(false);
  const [isMonitorLoading, setIsMonitorLoading] = useState(false);
  const [initialLoadDone, setInitialLoadDone] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [continueFromJobId, setContinueFromJobId] = useState<string | null>(null);
  const [jobToPrefill, setJobToPrefill] = useState<TrainingJob | null>(null);
  const [trainMode, setTrainMode] = useState<'new' | 'continue'>('new');
  const [showConfig, setShowConfig] = useState(false);

  const router = useRouter();
  const searchParams = useSearchParams();

  const [formKey, setFormKey] = useState(0);
  const defaultFormValues: TrainingParameters = {
    modelName: "ZPE-QuantumWeaver-V1",
    totalEpochs: 30,
    batchSize: 32,
    learningRate: 0.001,
    weightDecay: 0.0001,
    momentumParams: defaultZPEParams.momentumParams,
    strengthParams: defaultZPEParams.strengthParams,
    noiseParams: defaultZPEParams.noiseParams,
    couplingParams: defaultZPEParams.couplingParams,
    quantumCircuitSize: 32,
    labelSmoothing: 0.1,
    quantumMode: true,
    baseConfigId: undefined,
    mixupAlpha: 0.2,
  };

  const { control, handleSubmit, reset, watch, formState: { errors }, setValue } = useForm<TrainingParameters>({
    resolver: zodResolver(trainingFormSchema),
    defaultValues: defaultFormValues,
  });

  useEffect(() => {
    const momentum = watch("momentumParams");
    const strength = watch("strengthParams");
    const noise = watch("noiseParams");
    const coupling = watch("couplingParams");
  }, [watch]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
      isPollingRef.current = false;
      console.log("Polling stopped.");
    }
  }, []);

  const fetchJobsList = useCallback(async () => {
    setIsLoadingJobs(true);
    try {
      const response = await fetch(API_BASE_URL + '/jobs?limit=20');
      if (!response.ok) throw new Error("Failed to fetch jobs list");
      const data = await response.json();
      const sortedJobs = (data.jobs || []).sort((a: TrainingJobSummary, b: TrainingJobSummary) => new Date(b.start_time || 0).getTime() - new Date(a.start_time || 0).getTime());
      setJobsList(sortedJobs);
      return sortedJobs;
    } catch (error: any) {
      toast({ title: "Error fetching jobs", description: error.message, variant: "destructive" });
      return [];
    } finally {
      setIsLoadingJobs(false);
    }
  }, []);

  const pollJobStatus = useCallback(async (jobId: string, isInitialLoad = false) => {
    if (currentJobId !== jobId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      const jobData: TrainingJob = await response.json();
      setActiveJob(jobData);
      if (jobData.status === "running" || jobData.status === "completed") {
        const newPoint = {
          epoch: jobData.current_epoch,
          accuracy: jobData.accuracy,
          loss: jobData.loss,
          avg_zpe: jobData.zpe_effects && jobData.zpe_effects.length > 0 ? jobData.zpe_effects.reduce((a,b)=>a+b,0) / (jobData.zpe_effects.length) : 0
        };
        setChartData(prev => {
          const existingPointIndex = prev.findIndex(p => p.epoch === newPoint.epoch);
          if (existingPointIndex > -1) {
            const updatedPrev = [...prev];
            updatedPrev[existingPointIndex] = newPoint;
            return updatedPrev;
          }
          return [...prev, newPoint].sort((a,b) => a.epoch - b.epoch);
        });
        // Save ZPE stats for advisor dataset
        const zpeHistory = parseLogMessagesToZpeHistory(jobData.log_messages || []);
        saveZpeStatsToDataset(jobId, jobData.log_messages || [], zpeHistory);
      }
      if (jobData.status === "completed" || jobData.status === "failed" || jobData.status === "stopped") {
        stopPolling();
        fetchJobsList();
        toast({ 
          title: `Job ${jobData.status}`, 
          description: `Job ${jobId.replace('zpe_job_','')} (${jobData.parameters.modelName}) finished with status: ${jobData.status}. Accuracy: ${jobData.accuracy.toFixed(2)}%` 
        });
      }
    } catch (error: any) {
      console.error("Error polling job status:", error);
      toast({ title: "Error polling job status", description: error.message, variant: "destructive" });
      stopPolling();
    }
  }, [currentJobId, fetchJobsList, stopPolling]);

  const parseLogMessagesToZpeHistory = (logMessages: string[]): Array<{ epoch: number; zpe_effects: number[] }> => {
    if (!logMessages) return [];
    const zpeHistory: Array<{ epoch: number; zpe_effects: number[] } | null> = logMessages
      .map(log => {
        const zpeMatch = log.match(/E(\d+) END - TrainL: [\d.]+, ValAcc: [\d.]+, ValL: [\d.]+ ZPE: \[([,\d\s.]+)\]/);
        if (zpeMatch && zpeMatch[1] && zpeMatch[2]) {
          const epoch = parseInt(zpeMatch[1]);
          const zpeValues = zpeMatch[2].split(',').map(s => parseFloat(s.trim())).filter(s => !isNaN(s));
          if (zpeValues.length === 6) {
            return { epoch, zpe_effects: zpeValues };
          }
        }
        return null;
      });

    return zpeHistory.filter(Boolean) as Array<{ epoch: number; zpe_effects: number[] }>;
  };

  const logEpochMatch = (log: string) => log.match(/E(\d+)\s+B\d+\/\d+\s+L:\s*([\d.]+)/);
  const logEndEpochMatch = (log: string) => log.match(/E(\d+) END - TrainL: [\d.]+, ValAcc: ([\d.]+)%, ValL: ([\d.]+)/);
  const logZpeMatch = (log: string) => log.match(/ZPE: \[([,\d\s.]+)\]/);

  const handleViewJobDetails = useCallback(async (jobId: string, showLoading = true) => {
    if (currentJobId !== jobId) {
    stopPolling();
    setChartData([]);
    setActiveJob(null);
      setCurrentJobId(jobId);
      if (showLoading) setIsMonitorLoading(true);
    }
    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      const data: TrainingJob = await response.json();
      if (data.log_messages) {
        const chartDataFromLog = (data.log_messages || [])
          .map((log): { epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; } | null => {
            const epochMatch = logEpochMatch(log);
            const endEpochMatch = logEndEpochMatch(log);
            const zpeMatch = logZpeMatch(log);

            let epochData: { epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; } = { epoch: 0 };
            if (endEpochMatch) { epochData = { epoch: parseInt(endEpochMatch[1]), accuracy: parseFloat(endEpochMatch[2]), loss: parseFloat(endEpochMatch[3]) }; }
            else if (epochMatch) { epochData = { epoch: parseInt(epochMatch[1]), loss: parseFloat(epochMatch[2]) }; }
            if (zpeMatch && epochData.epoch) {
              const zpeValues = zpeMatch[1].split(',').map((s: string) => parseFloat(s.trim())).filter((s: number) => !isNaN(s));
              if (zpeValues.length > 0) epochData.avg_zpe = zpeValues.reduce((a, b) => a + b, 0) / zpeValues.length;
            }
            return epochData.epoch > 0 ? epochData : null;
          })
          .filter(Boolean) as Array<{ epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; }>;

        if (chartDataFromLog.length > 0) {
          setChartData(chartDataFromLog.sort((a, b) => a.epoch - b.epoch));
        } else if (data.status === "completed" || data.status === "stopped") {
        }
        // Save ZPE stats for advisor dataset
        const zpeHistory = parseLogMessagesToZpeHistory(data.log_messages);
        saveZpeStatsToDataset(jobId, data.log_messages, zpeHistory);
      }

      if (data.status === "running" || data.status === "pending") {
        if (!isPollingRef.current) {
          pollingRef.current = setInterval(() => pollJobStatus(jobId, false), 2000);
          isPollingRef.current = true;
          console.log(`Started polling for job ${jobId}`);
        }
        await pollJobStatus(jobId, true);
      } else {
        stopPolling();
      }
      setActiveJob(data);
      setIsMonitorLoading(false);
      setInitialLoadDone(true);
      if (data.status === "completed") {
        setContinueFromJobId(jobId);
      }
      setJobToPrefill(data);
      setTrainMode('new');
      // Prefill form with job parameters, but clear baseConfigId
      const paramsWithNewName = {
        ...defaultFormValues,
        ...data.parameters,
        modelName: `${data.parameters.modelName}_retrain_${Date.now().toString().slice(-4)}`,
        baseConfigId: undefined,
      };
      reset(paramsWithNewName);
      toast({ title: "Form Pre-filled", description: `Loaded parameters from job ${data.parameters.modelName}. Model name updated.` });
      setContinueFromJobId(data.job_id);
    } catch (error: any) {
      toast({ title: "Error fetching job details", description: error.message, variant: "destructive"});
      setActiveJob(null);
      stopPolling();
      setIsMonitorLoading(false);
      setInitialLoadDone(true);
    } finally {
      setIsSubmitting(false);
    }
  }, [pollJobStatus, stopPolling, currentJobId]);

  const onSubmit = async (data: TrainingParameters) => {
    setIsSubmitting(true);
    setChartData([]);
    stopPolling();

    try {
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error(`Failed to start training job. Server responded with ${response.status}`);
      const result = await response.json();
      if (!result.job_id) throw new Error("Backend did not return a job_id.");

      const newJob: TrainingJob = {
        job_id: result.job_id,
        status: "pending",
        current_epoch: 0,
        total_epochs: data.totalEpochs,
        accuracy: 0,
        loss: 0,
        zpe_effects: Array(6).fill(0),
        log_messages: [`Job ${result.job_id} submitted.`],
        parameters: data,
      };
      setActiveJob(newJob);

      if (!isPollingRef.current) {
        pollingRef.current = setInterval(() => pollJobStatus(result.job_id), 2000);
        isPollingRef.current = true;
        console.log(`Started polling for new job ${result.job_id}`);
      }
      fetchJobsList();
      toast({ title: "Training Started", description: `Job ID: ${result.job_id.replace('zpe_job_','')}` });
    } catch (error: any) {
      toast({ title: "Error starting training", description: error.message, variant: "destructive" });
      setActiveJob(null);
      stopPolling();
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

  useEffect(() => {
    const initializePage = async () => {
      console.log("useEffect: Starting initialization");
      try {
        // --- ADVISOR PARAMS FROM URL ---
        const advisedParamsEncoded = searchParams.get("advisedParams");
        if (advisedParamsEncoded) {
          try {
            const decoded = atob(advisedParamsEncoded);
            const parsed = JSON.parse(decoded);
            // Merge with defaults and validate
            const safeParams = {
              ...defaultFormValues,
              ...parsed,
              momentumParams: (parsed.momentumParams && parsed.momentumParams.length === 6) ? parsed.momentumParams : defaultFormValues.momentumParams,
              strengthParams: (parsed.strengthParams && parsed.strengthParams.length === 6) ? parsed.strengthParams : defaultFormValues.strengthParams,
              noiseParams: (parsed.noiseParams && parsed.noiseParams.length === 6) ? parsed.noiseParams : defaultFormValues.noiseParams,
              couplingParams: (parsed.couplingParams && parsed.couplingParams.length === 6) ? parsed.couplingParams : defaultFormValues.couplingParams,
            };
            reset(safeParams);
            toast({ title: "Loading in Trainer", description: `Parameters for \"${safeParams.modelName}\" pre-filled.` });
            return; // Don't do other prefill logic if we loaded from advisor
          } catch (e) {
            toast({ title: "Advisor Prefill Failed", description: "Could not decode or parse advisor parameters." });
          }
        }
        // --- END ADVISOR PARAMS FROM URL ---

        console.log("useEffect: Fetching jobs list...");
        const recentJobs = await fetchJobsList();
        console.log(`useEffect: Fetched ${recentJobs.length} jobs.`);

        const prefillJobId = searchParams.get("prefill");

        if (prefillJobId) {
          console.log(`useEffect: Found prefill job ID: ${prefillJobId}. Prioritizing pre-filling.`);
          try {
            const jobToPrefillResponse = await fetch(`${API_BASE_URL}/status/${prefillJobId}`);
            if (!jobToPrefillResponse.ok) {
              throw new Error(`Failed to fetch job details for prefill ID: ${prefillJobId}`);
            }
            const jobToPrefill: TrainingJob = await jobToPrefillResponse.json();

            const paramsWithNewNameAndBaseId = {
              ...defaultFormValues,
              ...jobToPrefill.parameters,
              modelName: `${jobToPrefill.parameters.modelName}_retrain_${Date.now().toString().slice(-4)}`,
              baseConfigId: prefillJobId,
            };

            const validatedParams = trainingFormSchema.safeParse(paramsWithNewNameAndBaseId);
            if (validatedParams.success) {
              reset(validatedParams.data);
              toast({ title: "Form Pre-filled", description: `Loaded parameters from job ${jobToPrefill.parameters.modelName}. Model name updated.` });
              console.log(`useEffect: Successfully pre-filled form from job ${prefillJobId}`);
            } else {
              console.error("useEffect: Job prefill validation error:", validatedParams.error);
              let errorMessages = "Validation errors: ";
              validatedParams.error.errors.forEach(err => {
                errorMessages += `${err.path.join('.')}: ${err.message}. `;
              });
              toast({ title: "Prefill Validation Error", description: `Parameters from job ${prefillJobId.slice(-6)} are not valid. ${errorMessages}`, variant: "destructive" });
              reset({...defaultFormValues, ...paramsWithNewNameAndBaseId});
            }

            if (jobToPrefill.status === "running" || jobToPrefill.status === "pending") {
              await handleViewJobDetails(prefillJobId, true);
              console.log(`useEffect: Loaded details for prefill job ${prefillJobId} into monitor (running/pending).`);
            } else {
              setActiveJob(jobToPrefill);
              setChartData([{
                epoch: jobToPrefill.current_epoch,
                accuracy: jobToPrefill.accuracy,
                loss: jobToPrefill.loss,
                avg_zpe: jobToPrefill.zpe_effects && jobToPrefill.zpe_effects.length > 0 ? jobToPrefill.zpe_effects.reduce((a,b)=>a+b,0) / jobToPrefill.zpe_effects.length : 0
              }]);
              setIsMonitorLoading(false);
              setInitialLoadDone(true);
              console.log(`useEffect: Loaded details for prefill job ${prefillJobId} into monitor (completed/failed/stopped).`);
            }
          } catch (e: any) {
            console.error(`useEffect: Error handling prefill job ${prefillJobId}:`, e);
            toast({ title: "Prefill Failed", description: e.message, variant: "destructive" });
          }
        } else {
          console.log("useEffect: No prefill job ID found. Checking for active jobs.");
          const runningOrPendingJob = recentJobs.find((j: TrainingJobSummary) => j.status === 'running' || j.status === 'pending');
          if (runningOrPendingJob) {
            setIsMonitorLoading(true);
            setInitialLoadDone(false);
            console.log(`useEffect: Found active job: ${runningOrPendingJob.job_id}. Loading details.`);
            toast({ title: "Active Job Found", description: `Monitoring job ${runningOrPendingJob.job_id.slice(-6)} (${runningOrPendingJob.model_name}).` });
            await handleViewJobDetails(runningOrPendingJob.job_id, true);
          } else {
            setIsMonitorLoading(false);
            setInitialLoadDone(true);
            console.log("useEffect: No active job found.");
          }
        }
      } catch (error: any) {
        console.error("useEffect: Error during initialization:", error);
        toast({ title: "Initialization Error", description: error.message, variant: "destructive" });
        setIsMonitorLoading(false);
        setInitialLoadDone(true);
      }
      console.log("useEffect: Initialization complete.");
    };

    initializePage();

    return () => {
      console.log("useEffect cleanup: Clearing polling interval.");
      stopPolling();
    };
  }, [searchParams, fetchJobsList, handleViewJobDetails, reset, stopPolling]);

  useEffect(() => {
    // Auto-load advisorParams from query param if present
    const advisorParams = searchParams.get("advisorParams");
    if (advisorParams) {
      try {
        const parsed = JSON.parse(decodeURIComponent(advisorParams));
        // Ensure baseConfigId is preserved if present
        reset({ ...defaultFormValues, ...parsed, baseConfigId: parsed.baseConfigId });
        toast({ title: "Advisor Parameters Loaded", description: "Parameters from the HS-QNN Advisor have been loaded into the form." });
      } catch (e) {
        toast({ title: "Failed to load advisor parameters", description: String(e), variant: "destructive" });
      }
    }
  }, [searchParams, reset, toast]);

  const renderParamArrayFields = (paramName: "momentumParams" | "strengthParams" | "noiseParams" | "couplingParams", labelPrefix: string, Icon: React.ElementType) => (
    <div className="space-y-2">
      <Label className="text-sm flex items-center gap-2"><Icon className="h-4 w-4 text-primary" /> {labelPrefix} <span className="text-xs text-muted-foreground">(6 layers)</span></Label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {(watch(paramName) || Array(6).fill(0)).map((_, index) => (
          <FieldController
            key={`${paramName}.${index}`}
            control={control}
            name={`${paramName}.${index}` as const}
            label={`Layer ${index + 1}`}
            type="number"
            min="0"
            max="1"
            step="0.01"
          />
        ))}
      </div>
    </div>
  );

  const handleSaveSuggestedParameters = async (params: TrainingParameters) => {
    if (!activeJob) {
      toast({ title: "Error", description: "No active job to save parameters for.", variant: "destructive" });
      return;
    }

    const configToSave = {
      name: params.modelName || `${params.modelName}_advised_${Date.now().toString().slice(-4)}`,
      parameters: {
        ...defaultZPEParams,
        ...params,
        baseConfigId: params.baseConfigId || activeJob.parameters.baseConfigId,
        modelName: params.modelName || `${params.modelName}_advised_${Date.now().toString().slice(-4)}`,
      },
      date_created: new Date().toISOString().split('T')[0],
      accuracy: 0,
      loss: 0,
      use_quantum_noise: params.quantumMode !== undefined ? params.quantumMode : activeJob.parameters.quantumMode,
    };

    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/configs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configToSave),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to save configuration: ${response.status} ${response.statusText}. ${errorData.detail || ''}`);
      }
      const saved = await response.json();
      toast({ title: "Configuration Saved", description: `Config "${configToSave.name}" (ID: ${saved.config_id.slice(-6)}) saved.` });
    } catch (error: any) {
      console.error("Error saving suggested parameters:", error);
      toast({ title: "Save Failed", description: error.message, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Tooltip state for job actions
  const [hoveredAction, setHoveredAction] = useState<{ jobId: string; action: string; x: number; y: number } | null>(null);

  useEffect(() => {
    let pollInterval: NodeJS.Timeout | null = null;

    const startPolling = () => {
      if (activeJob && (activeJob.status === 'running' || activeJob.status === 'pending')) {
        // Poll immediately and then set up interval
        pollJobStatus(activeJob.job_id, false);
        pollInterval = setInterval(() => {
          pollJobStatus(activeJob.job_id, false);
        }, 1000); // Reduced to 1 second for more frequent updates
      }
    };

    const stopPolling = () => {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    };

    startPolling();

    return () => {
      stopPolling();
    };
  }, [activeJob?.job_id, activeJob?.status, pollJobStatus]);

  // Add a handler to load PTH file when loading advice from advisor
  const handleAdvisorApplyParameters = (params: TrainingParameters, previousJobId?: string) => {
    // Ensure all required fields are present and arrays are correct length
    const safeParams = {
      ...defaultFormValues,
      ...params,
      momentumParams: (params.momentumParams && params.momentumParams.length === 6) ? params.momentumParams : defaultFormValues.momentumParams,
      strengthParams: (params.strengthParams && params.strengthParams.length === 6) ? params.strengthParams : defaultFormValues.strengthParams,
      noiseParams: (params.noiseParams && params.noiseParams.length === 6) ? params.noiseParams : defaultFormValues.noiseParams,
      couplingParams: (params.couplingParams && params.couplingParams.length === 6) ? params.couplingParams : defaultFormValues.couplingParams,
    };
    reset(safeParams);
    setFormKey(prevKey => prevKey + 1);
    if (previousJobId) {
      setValue('baseConfigId', previousJobId);
      setTrainMode('continue'); // Set immediately for UI feedback
      fetch(`${API_BASE_URL}/status/${previousJobId}`)
        .then(res => res.json())
        .then(job => {
          setJobToPrefill(job);
        });
    }
  };

  const [advisorOpen, setAdvisorOpen] = useState(true);

  // Add two explicit submit handlers for the two buttons
  const handleStartNewTraining = handleSubmit((data) => {
    setValue('baseConfigId', undefined);
    setTrainMode('new');
    onSubmit({ ...data, baseConfigId: undefined });
  });
  const handleContinueTraining = handleSubmit((data) => {
    // Always use baseConfigId from the form if present
    const baseConfigId = data.baseConfigId;
    setTrainMode('continue');
    onSubmit({ ...data, baseConfigId });
  });

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="container mx-auto p-4 md:p-6">
        {/* Train Model Button */}
        <div className="flex justify-end mb-6">
          <Dialog open={showConfig} onOpenChange={setShowConfig}>
            <DialogTrigger asChild>
              <Button variant="default" size="lg" onClick={() => setShowConfig(true)}>
                <Zap className="h-5 w-5 mr-2" /> Train Model
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl w-full">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 text-xl">
                  <Zap className="h-6 w-6 text-primary" /> Train Configuration
                </DialogTitle>
                <DialogDescription>Configure your ZPE-enhanced neural network training job.</DialogDescription>
              </DialogHeader>
              <form onSubmit={handleStartNewTraining} className="space-y-6" key={formKey}>
                <Tabs defaultValue="general" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="general">General</TabsTrigger>
                    <TabsTrigger value="zpe">ZPE</TabsTrigger>
                    <TabsTrigger value="quantum">Quantum</TabsTrigger>
                  </TabsList>
                  <TabsContent value="general">
                    <div className="grid gap-4">
                      <FieldController control={control} name="modelName" label="Model Name" placeholder="e.g., ZPE-QuantumWeaver-V1" />
                      <div className="grid grid-cols-2 gap-4">
                        <FieldController control={control} name="totalEpochs" label="Total Epochs" type="number" min="1" max="200" />
                        <FieldController control={control} name="batchSize" label="Batch Size" type="number" min="8" max="256" />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <FieldController control={control} name="learningRate" label="Learning Rate" type="number" min="0.00001" max="0.1" step="0.00001" />
                        <FieldController control={control} name="weightDecay" label="Weight Decay" type="number" min="1e-9" max="0.1" step="any" />
                      </div>
                      <FieldController control={control} name="mixupAlpha" label="Mixup Alpha" type="number" min="0" max="1" step="0.01" />
                      <FieldController control={control} name="baseConfigId" label="Base Config ID (Optional)" placeholder="e.g., config_123456" />
                    </div>
                  </TabsContent>
                  <TabsContent value="zpe">
                    <div className="space-y-6">
                      {renderParamArrayFields("momentumParams", "Momentum", Waves)}
                      {renderParamArrayFields("strengthParams", "Strength", SlidersHorizontal)}
                      {renderParamArrayFields("noiseParams", "Noise", Atom)}
                      {renderParamArrayFields("couplingParams", "Coupling", BrainCircuit)}
                    </div>
                  </TabsContent>
                  <TabsContent value="quantum">
                    <div className="space-y-4">
                      <FieldController control={control} name="quantumCircuitSize" label="Quantum Circuit Size" type="number" min="4" max="64" />
                      <FieldController control={control} name="labelSmoothing" label="Label Smoothing" type="number" min="0" max="0.5" step="0.01" />
                      <FieldControllerSwitch control={control} name="quantumMode" label="Quantum Mode" />
                    </div>
                  </TabsContent>
                </Tabs>
                <div className="flex gap-4 mt-4">
                  <Button
                    type="button"
                    variant={trainMode === 'new' ? 'default' : 'outline'}
                    onClick={handleStartNewTraining}
                    disabled={isSubmitting}
                  >
                    Start New Training Session
                  </Button>
                  <Button
                    type="button"
                    variant={trainMode === 'continue' ? 'default' : 'outline'}
                    onClick={handleContinueTraining}
                    disabled={isSubmitting || !((watch('baseConfigId') ?? '').length > 0)}
                  >
                    Train on Previously Trained Model (.pth)
                  </Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>
        {/* Main grid: Only monitor and job history */}
        <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
          <Card className="w-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl"><List className="h-6 w-6 text-primary" /> Training Monitor</CardTitle>
              <CardDescription>Status for Job ID: {activeJob?.job_id ? activeJob.job_id.replace('zpe_job_', '') : 'None'}</CardDescription>
            </CardHeader>
            <CardContent>
              {isMonitorLoading ? (
                <div className="flex flex-col items-center justify-center h-64">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                  <p className="text-muted-foreground text-lg">Loading job stats...</p>
                </div>
              ) : activeJob ? (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <Label>Status</Label>
                    <Badge className={activeJob.status === "completed" ? "bg-green-500 text-white" : ""}>
                      {activeJob.status.toUpperCase()}
                    </Badge>
                    {activeJob.status === "running" && (
                      <Button variant="default" onClick={() => handleStopJob(activeJob.job_id)} disabled={isSubmitting}>
                        <StopCircle className="mr-2 h-4 w-4" /> Stop Job
                      </Button>
                    )}
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricDisplay label="Epoch" value={`${activeJob.current_epoch}/${activeJob.total_epochs}`} />
                    <MetricDisplay label="Accuracy" value={`${activeJob.accuracy.toFixed(2)}%`} />
                    <MetricDisplay label="Loss" value={activeJob.loss.toFixed(4)} />
                    <MetricDisplay label="Avg ZPE" value={(activeJob.zpe_effects && activeJob.zpe_effects.length > 0 ? activeJob.zpe_effects.reduce((a, b) => a + b, 0) / activeJob.zpe_effects.length : 0).toFixed(2)} />
                  </div>
                  <div className="flex-1 w-full flex items-center justify-center" style={{ minHeight: 420, padding: 0, margin: 0 }}>
                    <div
                      style={{
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <div
                        className="neon-border"
                        style={{
                          width: 'calc(100% - 16px)',
                          height: 'calc(100% - 16px)',
                          margin: 8,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          background: 'transparent',
                          boxSizing: 'border-box',
                        }}
                      >
                        <div
                          style={{
                            background: 'rgba(10,20,40,0.85)',
                            width: 'calc(100% - 16px)',
                            height: 'calc(100% - 16px)',
                            margin: 8,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <NeonAnalyzerChart
                            traces={[
                              { data: chartData.map((d) => d.train_acc), color: '#00ffe7', label: 'Train Acc' },
                              { data: chartData.map((d) => d.val_acc), color: '#ff00ff', label: 'Val Acc' },
                            ]}
                            width={600}
                            height={380}
                            overlays={[
                              { text: 'Training Metrics', color: '#39ff14', x: 120, y: 40, fontSize: 24 },
                            ]}
                            showLegend
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  <ScrollArea className="h-[200px] w-full rounded-md border">
                    <div className="p-4">
                      <h4 className="mb-4 text-sm font-medium leading-none">Training Logs</h4>
                      {activeJob.log_messages.map((msg, idx) => (
                        <p key={idx} className="text-sm text-muted-foreground whitespace-pre-wrap">{msg}</p>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              ) : (
                initialLoadDone && (
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>No Active Job</AlertTitle>
                  <AlertDescription>Select a job from the history or start a new training job.</AlertDescription>
                </Alert>
                )
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl"><List className="h-6 w-6 text-primary" /> Job History</CardTitle>
            <CardDescription>View and manage past and ongoing training jobs.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex justify-end mb-4">
              <Button variant="outline" onClick={fetchJobsList} disabled={isLoadingJobs}>
                <RefreshCw className="mr-2 h-4 w-4" /> Refresh
              </Button>
            </div>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">Job ID</TableHead>
                    <TableHead>Model Name</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Epoch</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobsList.map((job) => (
                    <TableRow key={job.job_id}>
                      <TableCell className="font-medium">{job.job_id.replace('zpe_job_', '').slice(-6)}</TableCell>
                      <TableCell>{job.model_name}</TableCell>
                      <TableCell>
                        <Badge className={job.status === "completed" ? "bg-green-500 text-white" : ""}>
                          {job.status.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>{job.current_epoch}/{job.total_epochs || 0}</TableCell>
                      <TableCell>{job.accuracy.toFixed(2)}%</TableCell>
                      <TableCell className="relative">
                        <span
                          onMouseEnter={e => setHoveredAction({ jobId: job.job_id, action: "trainer", x: e.clientX, y: e.clientY })}
                          onMouseLeave={() => setHoveredAction(null)}
                        >
                        <Button
                          variant="ghost"
                            size="icon"
                            style={{ width: 40, height: 40 }}
                            onClick={() => handleViewJobDetails(job.job_id, true)}
                          disabled={isSubmitting}
                            className="mx-1"
                        >
                            <ArrowDownCircle className="h-6 w-6" />
                        </Button>
                        </span>
                        <span
                          onMouseEnter={e => setHoveredAction({ jobId: job.job_id, action: "advisor", x: e.clientX, y: e.clientY })}
                          onMouseLeave={() => setHoveredAction(null)}
                        >
                          <Button
                            variant="ghost"
                            size="icon"
                            style={{ width: 40, height: 40 }}
                            onClick={async () => {
                              const response = await fetch(`${API_BASE_URL}/status/${job.job_id}`);
                              if (response.ok) {
                                const data: TrainingJob = await response.json();
                                window.dispatchEvent(new CustomEvent("advisorPrefill", { detail: data.parameters }));
                                toast({ title: "Loaded in HS-QNN Advisor", description: `Parameters from job ${job.model_name} sent to advisor.` });
                              } else {
                                toast({ title: "Error", description: "Could not load job details for advisor.", variant: "destructive" });
                              }
                            }}
                            disabled={isSubmitting}
                            className="mx-1"
                          >
                            <Wand2 className="h-6 w-6" />
                          </Button>
                        </span>
                        <span
                          onMouseEnter={e => setHoveredAction({ jobId: job.job_id, action: "continue", x: e.clientX, y: e.clientY })}
                          onMouseLeave={() => setHoveredAction(null)}
                        >
                          <Button
                            variant="ghost"
                            size="icon"
                            style={{ width: 40, height: 40 }}
                            onClick={async () => {
                              const response = await fetch(`${API_BASE_URL}/status/${job.job_id}`);
                              if (response.ok) {
                                const data: TrainingJob = await response.json();
                                await onSubmit({ ...data.parameters, modelName: `${data.parameters.modelName}_cont_${Date.now().toString().slice(-4)}` });
                              } else {
                                toast({ title: "Error", description: "Could not load job details to continue training.", variant: "destructive" });
                              }
                            }}
                            disabled={isSubmitting}
                            className="mx-1"
                          >
                            <PlayCircle className="h-6 w-6" />
                          </Button>
                        </span>
                        {hoveredAction && hoveredAction.jobId === job.job_id && (
                          <div
                            style={{
                              position: 'fixed',
                              left: hoveredAction.x + 12,
                              top: hoveredAction.y + 12,
                              zIndex: 1000,
                              background: 'rgba(30,30,40,0.97)',
                              color: '#fff',
                              padding: '7px 14px',
                              borderRadius: 8,
                              fontSize: 15,
                              pointerEvents: 'none',
                              boxShadow: '0 2px 8px rgba(0,0,0,0.18)'
                            }}
                          >
                            {hoveredAction.action === "trainer" && "Load in Trainer"}
                            {hoveredAction.action === "advisor" && "Load in HS-QNN Advisor"}
                            {hoveredAction.action === "continue" && "Continue Training Now!"}
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </Suspense>
  );
}