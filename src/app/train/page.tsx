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
import { Play, StopCircle, List, Zap, Settings, RefreshCw, AlertTriangle, ExternalLink, SlidersHorizontal, Atom, Brain, Waves, BrainCircuit, Wand2, Save, Download } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { adviseHSQNNParameters, type HSQNNAdvisorInput, type HSQNNAdvisorOutput } from "@/ai/flows/hs-qnn-parameter-advisor";

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

const defaultZPEParams = {
  momentum: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
  strength: [0.35, 0.33, 0.31, 0.60, 0.27, 0.50],
  noise: [0.3, 0.28, 0.26, 0.35, 0.22, 0.25],
  coupling: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7],
};

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

export default function TrainModelPage() {
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [jobsList, setJobsList] = useState<TrainingJobSummary[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingRef = useRef<boolean>(false);

  const router = useRouter();
  const searchParams = useSearchParams();

  const [completedJobsListForAdvisor, setCompletedJobsListForAdvisor] = useState<TrainingJobSummary[]>([]);
  const [selectedJobIdForMiniAdvisor, setSelectedJobIdForMiniAdvisor] = useState<string>("");
  const [miniAdvisorObjective, setMiniAdvisorObjective] = useState<string>("Maximize validation accuracy while maintaining ZPE stability and exploring a slight increase in learning rate if previous accuracy was high.");
  const [miniAdvisorResult, setMiniAdvisorResult] = useState<HSQNNAdvisorOutput | null>(null);
  const [isLoadingMiniAdvisor, setIsLoadingMiniAdvisor] = useState<boolean>(false);
  const [miniAdvisorError, setMiniAdvisorError] = useState<string | null>(null);
  const [selectedPreviousJobDetailsForMiniAdvisor, setSelectedPreviousJobDetailsForMiniAdvisor] = useState<TrainingJob | null>(null);
  const [isMiniAdvisorLoadingJobs, setIsMiniAdvisorLoadingJobs] = useState(false);

  const [formKey, setFormKey] = useState(0); // New state variable for the key
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

 // Debugging useEffect to watch form state changes for array parameters
  useEffect(() => {
    const momentum = watch("momentumParams");
    const strength = watch("strengthParams");
    const noise = watch("noiseParams");
    const coupling = watch("couplingParams");
  }, [watch]); // This effect will re-run whenever watch detects changes in any field

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
    if (isPollingRef.current && !isInitialLoad && activeJob?.job_id !== jobId) {
      console.log(`Polling skipped for job ${jobId.replace('zpe_job_','')}: Another job is being polled or this is not the active job.`);
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      const jobData: TrainingJob = await response.json();

      if (isInitialLoad || activeJob?.job_id === jobId) {
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
        }

        if (jobData.status === "completed" || jobData.status === "failed" || jobData.status === "stopped") {
          stopPolling();
          fetchJobsList();
          toast({ 
            title: `Job ${jobData.status}`, 
            description: `Job ${jobId.replace('zpe_job_','')} (${jobData.parameters.modelName}) finished with status: ${jobData.status}. Accuracy: ${jobData.accuracy.toFixed(2)}%` 
          });
        }
      }
    } catch (error: any) {
      console.error("Error polling job status:", error);
      toast({ title: "Error polling job status", description: error.message, variant: "destructive" });
      stopPolling();
    }
  }, [activeJob?.job_id, fetchJobsList, stopPolling]);

  // New function to parse log messages and extract ZPE history
  const parseLogMessagesToZpeHistory = (logMessages: string[]): Array<{ epoch: number; zpe_effects: number[] }> => {
    if (!logMessages) return [];
    const zpeHistory: Array<{ epoch: number; zpe_effects: number[] } | null> = logMessages
      .map(log => {
        // This regex captures epoch, accuracy, loss, and the ZPE array string
        const zpeMatch = log.match(/E(\d+) END - TrainL: [\d.]+, ValAcc: [\d.]+, ValL: [\d.]+ ZPE: \[([,\d\s.]+)\]/);
        if (zpeMatch && zpeMatch[1] && zpeMatch[2]) {
          const epoch = parseInt(zpeMatch[1]);
          const zpeValues = zpeMatch[2].split(',').map(s => parseFloat(s.trim())).filter(s => !isNaN(s)); // Ensure parsing to numbers and filter NaNs
          if (zpeValues.length === 6) { // Only include if all 6 ZPE values are present
            return { epoch, zpe_effects: zpeValues };
          }
        }
        return null; // Return null for logs that don't match the ZPE pattern or have incomplete data
      });

    return zpeHistory.filter(Boolean) as Array<{ epoch: number; zpe_effects: number[] }>; // Filter out nulls and assert type
  };

  // Regex helper functions for parsing log messages
  const logEpochMatch = (log: string) => log.match(/E(\d+)\s+B\d+\/\d+\s+L:\s*([\d.]+)/);
  const logEndEpochMatch = (log: string) => log.match(/E(\d+) END - TrainL: [\d.]+, ValAcc: ([\d.]+)%, ValL: ([\d.]+)/);
  const logZpeMatch = (log: string) => log.match(/ZPE: \[([,\d\s.]+)\]/);

  const handleViewJobDetails = useCallback(async (jobId: string) => {
    stopPolling();
    setChartData([]);
    setActiveJob(null);
    setMiniAdvisorResult(null);
    setMiniAdvisorError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      const data: TrainingJob = await response.json();

      if (data.log_messages) {
        const chartDataFromLog = (data.log_messages || [])
          .map((log): { epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; } | null => {
            // Use the helper functions to match log patterns
            const epochMatch = logEpochMatch(log);
            const endEpochMatch = logEndEpochMatch(log);
            const zpeMatch = logZpeMatch(log);

            let epochData: { epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; } = { epoch: 0 };
            if (endEpochMatch) { epochData = { epoch: parseInt(endEpochMatch[1]), accuracy: parseFloat(endEpochMatch[2]), loss: parseFloat(endEpochMatch[3]) }; } // Prioritize end epoch data
            else if (epochMatch) { epochData = { epoch: parseInt(epochMatch[1]), loss: parseFloat(epochMatch[2]) }; } // Fallback to per-batch epoch data
            if (zpeMatch && epochData.epoch) {
              const zpeValues = zpeMatch[1].split(',').map((s: string) => parseFloat(s.trim())).filter((s: number) => !isNaN(s)); // Explicitly type 's'
              if (zpeValues.length > 0) epochData.avg_zpe = zpeValues.reduce((a, b) => a + b, 0) / zpeValues.length; // Calculate average ZPE
            }
            return epochData.epoch > 0 ? epochData : null; // Only return if epoch is valid
          })
          .filter(Boolean) as Array<{ epoch: number; accuracy?: number; loss?: number; avg_zpe?: number; }>; // Filter out nulls and assert type

        if (chartDataFromLog.length > 0) { // Check if any valid data was parsed from logs
          setChartData(chartDataFromLog.sort((a, b) => a.epoch - b.epoch));
        } else if (data.status === "completed" || data.status === "stopped") { // Fallback for completed/stopped jobs if log parsing failed
           // This case might need to be reviewed if log messages should always contain data for completed jobs
        }
      }


      if (data.status === "running" || data.status === "pending") {
        if (!isPollingRef.current) {
          pollingRef.current = setInterval(() => pollJobStatus(jobId), 2000);
          isPollingRef.current = true;
          console.log(`Started polling for job ${jobId}`);
        }
        await pollJobStatus(jobId, true);
      } else {
        stopPolling();
      }
    } catch (error: any) {
      toast({ title: "Error fetching job details", description: error.message, variant: "destructive"});
      setActiveJob(null);
      stopPolling();
    } finally { // Ensure submitting state is reset
      setIsSubmitting(false);
    }
  }, [pollJobStatus, stopPolling]);

  const fetchCompletedJobsForMiniAdvisor = useCallback(async () => {
    setIsMiniAdvisorLoadingJobs(true);
    try {
      const response = await fetch(`${API_BASE_URL}/jobs?limit=50`);
      if (!response.ok) throw new Error("Failed to fetch completed jobs list for advisor");
      const data = await response.json();
      const completedJobs = (data.jobs || [])
        .filter((job: TrainingJobSummary) => job.status === "completed")
        .sort((a: TrainingJobSummary, b: TrainingJobSummary) => new Date(b.start_time || 0).getTime() - new Date(a.start_time || 0).getTime());
      setCompletedJobsListForAdvisor(completedJobs);
      if (completedJobs.length > 0 && !selectedJobIdForMiniAdvisor) {
        setSelectedJobIdForMiniAdvisor(completedJobs[0].job_id);
      }
    } catch (error: any) {
      toast({ title: "Error fetching completed jobs", description: error.message, variant: "destructive" });
    } finally {
      setIsMiniAdvisorLoadingJobs(false);
    }
  }, [selectedJobIdForMiniAdvisor]);

  useEffect(() => {
    fetchCompletedJobsForMiniAdvisor();
  }, [fetchCompletedJobsForMiniAdvisor]);

  useEffect(() => {
    if (selectedJobIdForMiniAdvisor) {
      const fetchDetails = async () => {
        setIsLoadingMiniAdvisor(true);
        setMiniAdvisorResult(null);
        setMiniAdvisorError(null);
        try {
          const response = await fetch(`${API_BASE_URL}/status/${selectedJobIdForMiniAdvisor}`);
          if (!response.ok) throw new Error(`Failed to fetch details for job ${selectedJobIdForMiniAdvisor}`);
          const data: TrainingJob = await response.json();
          if (data.status !== 'completed') {
            setSelectedPreviousJobDetailsForMiniAdvisor(null);
            throw new Error(`Job ${selectedJobIdForMiniAdvisor} is not completed. Current status: ${data.status}`);
          }
          setSelectedPreviousJobDetailsForMiniAdvisor(data);
        } catch (e: any) {
          setSelectedPreviousJobDetailsForMiniAdvisor(null);
          setMiniAdvisorError("Failed to fetch selected job details for advisor: " + e.message);
          toast({ title: "Error fetching job details", description: e.message, variant: "destructive" });
        } finally {
          setIsLoadingMiniAdvisor(false);
        }
      };
      fetchDetails();
    } else {
      setSelectedPreviousJobDetailsForMiniAdvisor(null);
    }
  }, [selectedJobIdForMiniAdvisor]);

  const handleGetMiniAdvice = async () => {
    if (!selectedPreviousJobDetailsForMiniAdvisor) {
      toast({ title: "Error", description: "No previous job selected for HNN advice.", variant: "destructive" });
      return;
    }
    if (selectedPreviousJobDetailsForMiniAdvisor.status !== 'completed') {
      toast({ title: "Invalid Job", description: "Please select a 'completed' job for HNN advice.", variant: "destructive" });
      return;
    }

    setIsLoadingMiniAdvisor(true);
    setMiniAdvisorError(null);
    setMiniAdvisorResult(null);

    let validatedPreviousParams: TrainingParameters;
    try {
      const paramsToValidate = {
        ...defaultFormValues,
        ...selectedPreviousJobDetailsForMiniAdvisor.parameters,
      };
      validatedPreviousParams = trainingFormSchema.parse(paramsToValidate);
    } catch (validationError: any) {
      console.error("Validation error for previousTrainingParameters (mini advisor):", validationError);
      setMiniAdvisorError("Previous job parameters are not in the expected format. Check console for details. Error: " + validationError.message);
      toast({ title: "Parameter Mismatch", description: "Previous job parameters format error. " + validationError.message, variant: "destructive" });
      setIsLoadingMiniAdvisor(false);
      return;
    }

    // Format the ZPE history into a readable string for the AI
    const zpeHistory = parseLogMessagesToZpeHistory(selectedPreviousJobDetailsForMiniAdvisor.log_messages || []);
    const zpeHistoryString = zpeHistory.map(entry => 
      `Epoch ${entry.epoch}: [${entry.zpe_effects.map(zpe => zpe.toFixed(4)).join(', ')}]`
 ).join('\n');

    const inputForAI: HSQNNAdvisorInput = {
      previousJobId: selectedPreviousJobDetailsForMiniAdvisor.job_id,
      hnnObjective: miniAdvisorObjective,
 previousJobZpeHistory: parseLogMessagesToZpeHistory(selectedPreviousJobDetailsForMiniAdvisor.log_messages || []),
 previousJobZpeHistoryString: zpeHistoryString, // Add the formatted string
      previousTrainingParameters: validatedPreviousParams,
    };

    try {
      const result = await adviseHSQNNParameters(inputForAI);
      setMiniAdvisorResult(result);
    } catch (error: any) {
      console.error("Error getting HNN advice:", error);
      setMiniAdvisorError("Failed to get HNN advice: " + error.message);
      toast({ title: "HNN Advice Failed", description: error.message, variant: "destructive" });
    } finally {
      setIsLoadingMiniAdvisor(false);
    }  };

 // Changed to use setValue for more granular control and better compatibility with react-hook-form for arrays
 const handleApplyMiniAdvice = () => {
    if (!miniAdvisorResult?.suggestedNextTrainingParameters || !selectedPreviousJobDetailsForMiniAdvisor) {
      toast({ title: "No advice to apply", variant: "destructive" });
      return;
    }

    const suggested = miniAdvisorResult.suggestedNextTrainingParameters;
    const previousParams = selectedPreviousJobDetailsForMiniAdvisor.parameters;

    // Start with defaults
    let mergedParams: TrainingParameters = {
      ...defaultFormValues,
    };

    // Override with previous job's parameters
    mergedParams = {
      ...mergedParams,
      ...previousParams,
    };

    // Finally, override with suggested parameters to ensure suggested values take precedence.
    // Explicitly handle array parameters and specific fields like modelName and baseConfigId.
    mergedParams = {
      ...mergedParams,
      ...suggested,
      // Ensure modelName is set correctly, prioritizing suggested or creating a new name
      // based on the previous model name.
      modelName: suggested.modelName || `${previousParams.modelName}_adv_${Date.now().toString().slice(-3)}`,
      // Ensure baseConfigId is set to the previous job's ID
      baseConfigId: selectedPreviousJobDetailsForMiniAdvisor.job_id,
    };

    const validationResult = trainingFormSchema.safeParse(mergedParams);

    if (validationResult.success) {
      reset(validationResult.data);
      toast({ title: "Parameters Applied", description: "Advisor suggestions applied to the form." });
      setFormKey(prevKey => prevKey + 1); // Increment key to force re-render
    } else {
      console.error("Validation error applying HNN mini-advice:", validationResult.error.errors);
      let errorSummary = "Error applying advice: " + validationResult.error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ');
      toast({ title: "Validation Error on Apply", description: errorSummary, variant: "destructive" });
    }
  };

  const handleSaveSuggestedParameters = async () => {
    if (!miniAdvisorResult?.suggestedNextTrainingParameters || !selectedPreviousJobDetailsForMiniAdvisor) {
      toast({ title: "Error", description: "No suggested parameters to save.", variant: "destructive" });
      return;
    }

    const suggested = miniAdvisorResult.suggestedNextTrainingParameters;
    const configToSave = {
      name: suggested.modelName || `${selectedPreviousJobDetailsForMiniAdvisor.parameters.modelName}_advised_${Date.now().toString().slice(-4)}`,
      parameters: {
        ...defaultFormValues,
        ...selectedPreviousJobDetailsForMiniAdvisor.parameters,
        ...suggested,
        baseConfigId: selectedPreviousJobDetailsForMiniAdvisor.job_id,
        modelName: suggested.modelName || `${selectedPreviousJobDetailsForMiniAdvisor.parameters.modelName}_advised_${Date.now().toString().slice(-4)}`,
      },
      date_created: new Date().toISOString().split('T')[0],
      accuracy: 0,
      loss: 0,
      use_quantum_noise: suggested.quantumMode !== undefined ? suggested.quantumMode : selectedPreviousJobDetailsForMiniAdvisor.parameters.quantumMode,
    };

    configToSave.parameters = {
      totalEpochs: configToSave.parameters.totalEpochs,
      batchSize: configToSave.parameters.batchSize,
      learningRate: configToSave.parameters.learningRate,
      weightDecay: configToSave.parameters.weightDecay,
      momentumParams: configToSave.parameters.momentumParams,
      strengthParams: configToSave.parameters.strengthParams,
      noiseParams: configToSave.parameters.noiseParams,
      couplingParams: configToSave.parameters.couplingParams || defaultZPEParams.coupling,
      quantumCircuitSize: configToSave.parameters.quantumCircuitSize,
      labelSmoothing: configToSave.parameters.labelSmoothing,
      quantumMode: configToSave.parameters.quantumMode,
      modelName: configToSave.parameters.modelName,
      baseConfigId: configToSave.parameters.baseConfigId,
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

  const handleDownloadSuggestedParameters = () => {
    if (!miniAdvisorResult?.suggestedNextTrainingParameters || !selectedPreviousJobDetailsForMiniAdvisor) {
      toast({ title: "Error", description: "No suggested parameters to download.", variant: "destructive" });
      return;
    }
    const suggested = miniAdvisorResult.suggestedNextTrainingParameters;
    const filename = `${suggested.modelName || `advised_params_${Date.now().toString().slice(-4)}`}.json`;
    const jsonStr = JSON.stringify({
      ...defaultFormValues,
      ...selectedPreviousJobDetailsForMiniAdvisor.parameters,
      ...suggested,
      modelName: suggested.modelName || `${selectedPreviousJobDetailsForMiniAdvisor.parameters.modelName}_advised_${Date.now().toString().slice(-4)}`,
      baseConfigId: selectedPreviousJobDetailsForMiniAdvisor.job_id,
    }, null, 2);

    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast({ title: "Parameters Downloaded", description: `Saved as ${filename}` });
  };

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
              await handleViewJobDetails(prefillJobId);
              console.log(`useEffect: Loaded details for prefill job ${prefillJobId} into monitor (running/pending).`);
            } else {
              setActiveJob(jobToPrefill);
              setChartData([{
                epoch: jobToPrefill.current_epoch,
                accuracy: jobToPrefill.accuracy,
                loss: jobToPrefill.loss,
                avg_zpe: jobToPrefill.zpe_effects && jobToPrefill.zpe_effects.length > 0 ? jobToPrefill.zpe_effects.reduce((a,b)=>a+b,0) / jobToPrefill.zpe_effects.length : 0
              }]);
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
            console.log(`useEffect: Found active job: ${runningOrPendingJob.job_id}. Loading details.`);
            toast({ title: "Active Job Found", description: `Monitoring job ${runningOrPendingJob.job_id.slice(-6)} (${runningOrPendingJob.model_name}).` });
            await handleViewJobDetails(runningOrPendingJob.job_id);
          } else {
            console.log("useEffect: No active job found.");
          }
        }
      } catch (error: any) {
        console.error("useEffect: Error during initialization:", error);
        toast({ title: "Initialization Error", description: error.message, variant: "destructive" });
      } finally {
        console.log("useEffect: Initialization complete.");
      }
    };

    initializePage();

    return () => {
      console.log("useEffect cleanup: Clearing polling interval.");
      stopPolling();
    };
  }, [searchParams, fetchJobsList, handleViewJobDetails, reset, stopPolling]);

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

  return (
    <div className="container mx-auto py-6 px-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Training Configuration */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl"><Zap className="h-6 w-6 text-primary" /> Train Configuration</CardTitle>
            <CardDescription>Configure your ZPE-enhanced neural network training job.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6" key={formKey}> {/* Add key prop here */}
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
              {/* HS-QNN Mini Advisor */}
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg"><Brain className="h-5 w-5 text-primary" /> HS-QNN Mini Advisor</CardTitle>
                  <CardDescription>Get AI-driven suggestions for your next training step.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="selectedJobIdForMiniAdvisor">Select Previous Job</Label>
                    <Select
                      value={selectedJobIdForMiniAdvisor}
                      onValueChange={setSelectedJobIdForMiniAdvisor}
                      disabled={isMiniAdvisorLoadingJobs || isLoadingMiniAdvisor}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select a completed job..." />
                      </SelectTrigger>
                      <SelectContent>
                        {completedJobsListForAdvisor.map((job) => (
                          <SelectItem key={job.job_id} value={job.job_id}>
                            {job.job_id.replace('zpe_job_', '')} ({job.model_name}, Acc: {job.accuracy.toFixed(2)}%)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="miniAdvisorObjective">Advisor Objective</Label>
                    <Textarea
                      id="miniAdvisorObjective"
                      value={miniAdvisorObjective}
                      onChange={(e) => setMiniAdvisorObjective(e.target.value)}
                      placeholder="e.g., Maximize validation accuracy while maintaining ZPE stability..."
                      className="min-h-[80px] w-full"
                    />
                  </div>
                  {selectedPreviousJobDetailsForMiniAdvisor && (
                    <div className="space-y-2">
                      <Label>Selected Job Details</Label>
                      <pre className="p-2 bg-muted rounded-md text-sm overflow-auto max-h-32">
                        {JSON.stringify(selectedPreviousJobDetailsForMiniAdvisor.parameters, null, 2)}
                      </pre>
                    </div>
                  )}
                  {miniAdvisorError && (
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Error</AlertTitle>
                      <AlertDescription>{miniAdvisorError}</AlertDescription>
                    </Alert>
                  )}
                  {miniAdvisorResult && (
                    <div className="space-y-2">
 <Label>Advisor Reasoning</Label>
                      <pre className="p-2 bg-muted rounded-md text-sm whitespace-pre-wrap overflow-auto max-h-32">
 {miniAdvisorResult.reasoning || "No reasoning provided."}
 </pre>
 <Separator />
                      <Label>Suggested Parameters</Label>
                      <pre className="p-2 bg-muted rounded-md text-sm overflow-auto max-h-32">
                        {JSON.stringify(miniAdvisorResult.suggestedNextTrainingParameters, null, 2)}
                      </pre>
                      <div className="flex gap-2">
                        <Button onClick={handleApplyMiniAdvice} disabled={isLoadingMiniAdvisor}>
                          <Wand2 className="mr-2 h-4 w-4" /> Apply to Form
                        </Button>
                        <Button variant="outline" onClick={handleSaveSuggestedParameters} disabled={isLoadingMiniAdvisor || isSubmitting}>
                          <Save className="mr-2 h-4 w-4" /> Save Config
                        </Button>
                        <Button variant="outline" onClick={handleDownloadSuggestedParameters}>
                          <Download className="mr-2 h-4 w-4" /> Download JSON
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
                <CardFooter>
                  <Button onClick={handleGetMiniAdvice} disabled={isLoadingMiniAdvisor || !selectedJobIdForMiniAdvisor || isSubmitting}>
                    {isLoadingMiniAdvisor ? "Generating Advice..." : "Get HNN Advice"}
                  </Button>
                </CardFooter>
              </Card>
              <CardFooter className="flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={() => reset(defaultFormValues)} disabled={isSubmitting}>
                  Reset
                </Button>
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting ? "Submitting..." : "Start Training"}
                </Button>
              </CardFooter>
            </form>
          </CardContent>
        </Card>

        {/* Training Monitor */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl"><List className="h-6 w-6 text-primary" /> Training Monitor</CardTitle>
            <CardDescription>Status for Job ID: {activeJob?.job_id ? activeJob.job_id.replace('zpe_job_', '') : 'None'}</CardDescription>
          </CardHeader>
          <CardContent>
            {activeJob ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <Label>Status</Label>
                  <Badge
                    variant={
                      activeJob.status === "running"
                        ? "default"
                        : activeJob.status === "completed"
                        ? "default"
                        : "destructive"
                    }
                    className={activeJob.status === "completed" ? "bg-green-500 text-white" : ""}
                  >
                    {activeJob.status.toUpperCase()}
                  </Badge>
                  {activeJob.status === "running" && (
                    <Button variant="destructive" onClick={() => handleStopJob(activeJob.job_id)} disabled={isSubmitting}>
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
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                      <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                      <Tooltip />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#8884d8" activeDot={{ r: 8 }} />
                      <Line yAxisId="left" type="monotone" dataKey="loss" name="Loss" stroke="#ff7300" />
                      <Line yAxisId="right" type="monotone" dataKey="avg_zpe" name="Avg ZPE" stroke="#82ca9d" />
                    </LineChart>
                  </ResponsiveContainer>
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
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>No Active Job</AlertTitle>
                <AlertDescription>Select a job from the history or start a new training job.</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Job History */}
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
                      <Badge
                        variant={
                          job.status === "running"
                            ? "default"
                            : job.status === "completed"
                            ? "default"
                            : "destructive"
                        }
                        className={job.status === "completed" ? "bg-green-500 text-white" : ""}
                      >
                        {job.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{job.current_epoch}/{job.total_epochs || 0}</TableCell>
                    <TableCell>{job.accuracy.toFixed(2)}%</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleViewJobDetails(job.job_id)}
                        disabled={isSubmitting}
                      >
                        <ExternalLink className="h-4 w-4" />
                        <span className="sr-only">View Details</span>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}