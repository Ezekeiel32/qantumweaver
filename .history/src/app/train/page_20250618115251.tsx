"use client";

import React, { useState, useEffect, useCallback, useRef, useLayoutEffect } from "react";
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
import { Play, StopCircle, List, Zap, Settings, RefreshCw, AlertTriangle, ExternalLink, SlidersHorizontal, Atom, Brain, Waves, BrainCircuit, Wand2, Save, Download, ArrowDownCircle, PlayCircle, ChevronLeft, ChevronRight, Maximize2, Minimize2, BarChart3 } from "lucide-react";
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
import NeonAnalyzerChart from '@/components/visualizations/NeonAnalyzerChart';

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

  const retroLogRef = useRef<HTMLDivElement | null>(null);
  const lastLogRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    if (!activeJob || !activeJob.log_messages) return;
    if (lastLogRef.current) {
      lastLogRef.current.scrollIntoView({ behavior: 'auto', block: 'end' });
    }
  }, [activeJob?.log_messages?.length]);

  // --- New state for flexbox resizable layout ---
  const minPanelWidth = 320;
  const [configWidth, setConfigWidth] = useState(420); // px
  const [monitorWidth, setMonitorWidth] = useState(700); // px
  const [draggingDivider, setDraggingDivider] = useState(false);
  const dragStartX = useRef<number | null>(null);
  const dragStartConfigWidth = useRef<number | null>(null);
  const dragStartMonitorWidth = useRef<number | null>(null);
  // When minimized, configWidth is 0 and monitor fills all
  useEffect(() => {
    if (!draggingDivider) return;
    const onMouseMove = (e: MouseEvent) => {
      if (dragStartX.current !== null && dragStartConfigWidth.current !== null && dragStartMonitorWidth.current !== null) {
        const dx = e.clientX - dragStartX.current;
        let newConfigWidth = dragStartConfigWidth.current + dx;
        let newMonitorWidth = dragStartMonitorWidth.current - dx;
        if (newConfigWidth < minPanelWidth) {
          newMonitorWidth += newConfigWidth - minPanelWidth;
          newConfigWidth = minPanelWidth;
        }
        if (newMonitorWidth < minPanelWidth) {
          newConfigWidth += newMonitorWidth - minPanelWidth;
          newMonitorWidth = minPanelWidth;
        }
        setConfigWidth(newConfigWidth);
        setMonitorWidth(newMonitorWidth);
      }
    };
    const onMouseUp = () => {
      setDraggingDivider(false);
      dragStartX.current = null;
      dragStartConfigWidth.current = null;
      dragStartMonitorWidth.current = null;
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "";
    };
  }, [draggingDivider]);

  // --- Minimize/restore config logic ---
  const [configMinimized, setConfigMinimized] = useState(false);
  const [lastConfigWidth, setLastConfigWidth] = useState(configWidth);
  useEffect(() => {
    if (!configMinimized && configWidth === 0) {
      setConfigWidth(lastConfigWidth > minPanelWidth ? lastConfigWidth : 420);
    }
    if (configMinimized && configWidth > 0) {
      setLastConfigWidth(configWidth);
      setConfigWidth(0);
    }
  }, [configMinimized]);

  // --- Maximized monitor state (for overlay/fullscreen mode) ---
  const [monitorMaximized, setMonitorMaximized] = useState(false);

  // Ref for maximized chart container
  const maximizedChartContainerRef = useRef<HTMLDivElement>(null);

  // Chart dimensions for maximized mode (avoid SSR window usage)
  const [chartDims, setChartDims] = useState({ width: 1200, height: 500 });

  // --- Main flexbox layout ---
  return (
    <>
      {/* Overlay when maximized to block interaction with rest of page, but always below the monitor */}
      {monitorMaximized && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 900,
          background: 'rgba(16,19,26,0.85)',
          pointerEvents: 'auto',
        }} />
      )}
      <div style={{
        display: 'flex',
        flexDirection: 'row',
        width: '100%',
        height: '100vh',
        alignItems: 'stretch',
        background: 'none',
        position: 'relative',
      }}>
        {/* Monitor Panel (left) */}
        <div
          style={{
            flex: 2,
            minWidth: minPanelWidth,
            maxWidth: '100%',
            width: monitorWidth,
            transition: 'width 0.25s cubic-bezier(.4,2,.6,1)',
            position: 'relative',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            minHeight: 0,
            marginRight: '2.5rem',
          }}
        >
          <Card className="w-full">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg py-1">
                  <List className="h-5 w-5 text-primary" />
                  <span style={{ fontSize: 18 }}>Training Monitor</span>
                  {/* Maximize button */}
                  <button
                    onClick={() => setMonitorMaximized(true)}
                    style={{
                      marginLeft: 'auto',
                      background: '#10131a',
                      border: '2px solid #00ffe7',
                      borderRadius: 8,
                      boxShadow: '0 0 8px #00ffe7',
                      width: 36,
                      height: 36,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      marginRight: 0,
                    }}
                    title="Maximize monitor"
                  >
                    <Maximize2 className="h-5 w-5" style={{ color: '#00ffe7' }} />
                  </button>
                </CardTitle>
                <CardDescription className="text-xs pb-1">Status for Job ID: {activeJob?.job_id ? activeJob.job_id.replace('zpe_job_', '') : 'None'}</CardDescription>
            </CardHeader>
              <CardContent style={{ display: 'flex', flexDirection: 'column', height: '100%', paddingTop: 8, paddingBottom: 8 }}>
                {/* Margin bar with widgets */}
                <div className="flex items-center gap-8 px-8 py-4 mb-2 bg-black/30 rounded-lg border border-neon-cyan" style={{marginBottom: '1.5rem'}}>
                  <h2 className="neon-heading text-2xl flex-shrink-0" style={{minWidth: 220}}>Training Progress</h2>
                  <div className="flex gap-6 flex-wrap items-center">
                    <span className="font-mono text-base px-4 py-1 rounded bg-black/40 border border-neon-cyan text-neon-cyan">{activeJob?.status?.toUpperCase() || 'NONE'}</span>
                    <span className="font-mono text-base px-4 py-1 rounded bg-black/40 border border-neon-cyan text-neon-cyan">Epoch: {activeJob?.current_epoch ?? '--'}/{activeJob?.total_epochs ?? '--'}</span>
                    <span className="font-mono text-base px-4 py-1 rounded bg-black/40 border border-neon-cyan text-neon-cyan">Accuracy: {activeJob ? `${activeJob.accuracy.toFixed(2)}%` : '--'}</span>
                    <span className="font-mono text-base px-4 py-1 rounded bg-black/40 border border-neon-cyan text-neon-cyan">Loss: {activeJob ? activeJob.loss.toFixed(4) : '--'}</span>
                    <span className="font-mono text-base px-4 py-1 rounded bg-black/40 border border-neon-cyan text-neon-cyan">Avg ZPE: {activeJob?.zpe_effects && activeJob.zpe_effects.length > 0 ? (activeJob.zpe_effects.reduce((a, b) => a + b, 0) / activeJob.zpe_effects.length).toFixed(2) : '--'}</span>
                    {activeJob?.status === 'running' && (
                      <button onClick={() => handleStopJob(activeJob.job_id)} className="ml-2 px-4 py-1 rounded border border-neon-cyan text-neon-cyan bg-black/40 font-mono hover:text-neon-yellow hover:border-neon-yellow transition">Stop Job</button>
                    )}
                  </div>
                </div>
                {/* Chart always visible below widgets, takes all available space */}
                <div className="w-full flex-1" style={{ minHeight: 220, maxHeight: 340, flexGrow: 1, flexShrink: 1, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                  <NeonAnalyzerChart
                    traces={[
                      { data: chartData.map(d => typeof d.accuracy === 'number' ? d.accuracy : null).filter(v => v !== null), color: '#00ffe7', label: 'Accuracy' },
                      { data: chartData.map(d => typeof d.loss === 'number' ? d.loss : null).filter(v => v !== null), color: '#ff00ff', label: 'Loss' },
                      { data: chartData.map(d => typeof d.avg_zpe === 'number' ? d.avg_zpe : null).filter(v => v !== null), color: '#ffe066', label: 'Avg ZPE' },
                    ]}
                    width={chartDims.width}
                    height={chartDims.height}
                    overlays={[
                      { text: 'Training Progress', color: '#39ff14', x: 120, y: 40, fontSize: 24 },
                      { text: activeJob?.parameters?.modelName || '', color: '#00ffe7', x: 80, y: 300, fontSize: 16 },
                    ]}
                    showLegend
                  />
                  {/* Terminal output: separate modal/panel below chart, left-aligned, fixed width/height */}
                  <div className="glass neon-border mt-3" style={{ width: chartDims.width, minWidth: 320, maxWidth: '100%', height: 130, minHeight: 120, maxHeight: 140, overflowY: 'auto', borderRadius: 12, boxShadow: '0 0 16px #00ffe7', border: '2px solid #00ffe7', background: 'rgba(20, 20, 40, 0.85)', marginLeft: 0, marginRight: 'auto', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', padding: '0.5em 0.7em' }}>
                    <div className="p-0" ref={retroLogRef} id="retro-log-container" style={{ height: '100%', overflowY: 'auto', position: 'relative' }}>
                      <h4 className="mb-1 text-xs font-medium leading-none text-[#00ffcc] drop-shadow-[0_0_4px_#00ffcc]">Training Logs</h4>
                      <div style={{ fontSize: '0.95em', lineHeight: '1.2', letterSpacing: '0.01em' }}>
                        {activeJob?.log_messages?.slice(-6).map((msg, idx, arr) => {
                          let color = '#00ffcc';
                          let shadow = '0 0 6px #00ffcc';
                          if (/error|fail|nan|not found|exception|critical/i.test(msg)) {
                            color = '#ff3c3c'; shadow = '0 0 8px #ff3c3c';
                          } else if (/warn|deprecated|unstable|caution|stop/i.test(msg)) {
                            color = '#ffe066'; shadow = '0 0 8px #ffe066';
                          } else if (/success|completed|done|epoch|init|starting|device|cuda|optimizer|initialized/i.test(msg)) {
                            color = '#00ffcc'; shadow = '0 0 8px #00ffcc';
                          }
                          return (
                            <div
                              key={idx}
                              className="retro-log-line"
                              ref={idx === arr.length - 1 ? lastLogRef : undefined}
                              style={{
                                color,
                                textShadow: shadow,
                                fontWeight: 500,
                                fontFamily: 'inherit',
                                whiteSpace: 'pre-wrap',
                                marginBottom: 2,
                                letterSpacing: '0.01em',
                                filter: 'brightness(1.15) contrast(1.1)',
                                userSelect: 'text',
                              }}
                            >
                              {msg}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
        </div>
        {/* Config Panel (right) */}
        <div
          style={{
            width: configMinimized ? 0 : configWidth,
            minWidth: configMinimized ? 0 : minPanelWidth,
            maxWidth: 600,
            transition: 'width 0.25s cubic-bezier(.4,2,.6,1)',
            overflow: 'hidden',
            position: 'relative',
            zIndex: 20,
            display: configMinimized ? 'none' : 'block',
            height: '100%',
            minHeight: 0,
          }}
        >
          <Card className="w-full h-full relative" style={{ minWidth: minPanelWidth, maxWidth: 600, height: '100%', borderRadius: 18, boxShadow: '0 0 32px #00ffe7AA', border: '3px solid #00ffe7', background: '#181a22F2' }}>
          <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg py-1"><Zap className="h-5 w-5 text-primary" /> <span style={{ fontSize: 18 }}>Train Configuration</span></CardTitle>
              <CardDescription className="text-xs pb-1">Configure your ZPE-enhanced neural network training job.</CardDescription>
              {/* Minimize button, bottom left, retro-futuristic neon chevron */}
              <button
                onClick={() => setConfigMinimized(true)}
                style={{
                  position: "absolute",
                  left: -22,
                  bottom: 18,
                  zIndex: 30,
                  background: "#10131a",
                  border: "2px solid #00ffe7",
                  borderRadius: "50%",
                  boxShadow: "0 0 12px #00ffe7, 0 2px 8px #0008",
                  width: 36,
                  height: 36,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  transition: "background 0.2s",
                }}
                title="Minimize configuration panel"
              >
                <ChevronLeft className="h-6 w-6" style={{ color: "#00ffe7", filter: "drop-shadow(0 0 6px #00ffe7)" }} />
              </button>
          </CardHeader>
            <CardContent style={{ paddingTop: 8, paddingBottom: 8 }}>
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
          </CardContent>
        </Card>
        </div>
      </div>
    </>
  );
}