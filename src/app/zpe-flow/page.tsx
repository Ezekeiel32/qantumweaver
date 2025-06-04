"use client";
import React, { useState, useEffect, useCallback, useRef } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useRouter, useSearchParams } from "next/navigation";

import { adviseHSQNNParameters, type HSQNNAdvisorInput, type HSQNNAdvisorOutput } from "@/ai/flows/hs-qnn-parameter-advisor";
import type { TrainingParameters, TrainingJob, TrainingJobSummary } from "@/types/training";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/hooks/use-toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BrainCircuit, Lightbulb, Terminal, Wand2, ArrowRight, RefreshCw, SlidersHorizontal } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

// Define TrainingParametersSchema locally for client-side validation
const TrainingParametersSchema = z.object({
  totalEpochs: z.number().int().min(1).max(200),
  batchSize: z.number().int().min(8).max(256),
  learningRate: z.number().min(0.00001).max(0.1),
  weightDecay: z.number().min(0).max(0.1),
  momentumParams: z.array(z.number().min(0).max(1)).length(6, "Must have 6 momentum parameters"),
  strengthParams: z.array(z.number().min(0).max(1)).length(6, "Must have 6 strength parameters"),
  noiseParams: z.array(z.number().min(0).max(1)).length(6, "Must have 6 noise parameters"),
  couplingParams: z.array(z.number().min(0).max(1)).length(6, "Must have 6 coupling parameters"),
  quantumCircuitSize: z.number().int().min(4).max(64),
  labelSmoothing: z.number().min(0).max(0.5),
  quantumMode: z.boolean(),
  modelName: z.string().min(3, "Model name must be at least 3 characters"),
  baseConfigId: z.string().nullable().optional(),
});

// Define HSQNNAdvisorInputSchema locally for client-side validation
const HSQNNAdvisorInputSchemaClient = z.object({
  previousJobId: z.string().min(1, "Please select a previous job."),
  hnnObjective: z.string().min(20, "Objective must be at least 20 characters long.").max(500, "Objective is too long."),
});

type AdvisorFormValues = z.infer<typeof HSQNNAdvisorInputSchemaClient>;

export default function HSQNNParameterAdvisorPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [jobsList, setJobsList] = useState<TrainingJobSummary[]>([]);
  const [selectedJobDetails, setSelectedJobDetails] = useState<TrainingJob | null>(null);
  const [adviceResult, setAdviceResult] = useState<HSQNNAdvisorOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const router = useRouter();
  const searchParams = useSearchParams();

  const { control, handleSubmit, watch, setValue, formState: { errors } } = useForm<AdvisorFormValues>({
    resolver: zodResolver(HSQNNAdvisorInputSchemaClient),
    defaultValues: {
      previousJobId: "",
      hnnObjective: "Maximize validation accuracy while maintaining ZPE effects for all layers between 0.05 and 0.15. Explore slight increase in learning rate if previous accuracy was high.",
    },
  });

  const watchedJobId = watch("previousJobId");

  const fetchJobsList = useCallback(async () => {
    setIsLoadingJobs(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/jobs?limit=50`);
      if (!response.ok) throw new Error("Failed to fetch jobs list from backend.");
      const data = await response.json();
      const completedJobs = (data.jobs || []).filter((job: TrainingJobSummary) => job.status === "completed")
        .sort((a: TrainingJobSummary, b: TrainingJobSummary) => new Date(b.start_time || 0).getTime() - new Date(a.start_time || 0).getTime());
      setJobsList(completedJobs);
      
      const preselectJobId = searchParams.get("jobId");
      if (preselectJobId && completedJobs.find(j => j.job_id === preselectJobId)) {
        setValue("previousJobId", preselectJobId);
      } else if (completedJobs.length > 0 && !watchedJobId) { 
         setValue("previousJobId", completedJobs[0].job_id); 
      }
    } catch (e: any) {
      setError("Failed to fetch jobs: " + e.message);
      toast({ title: "Error fetching jobs", description: e.message, variant: "destructive" });
    } finally {
      setIsLoadingJobs(false);
    }
  }, [setValue, searchParams, watchedJobId]);

  useEffect(() => {
    fetchJobsList();
  }, [fetchJobsList]);

  useEffect(() => {
    if (watchedJobId) {
      const fetchJobDetails = async () => {
        setIsLoading(true);
        setAdviceResult(null); 
        setError(null);
        try {
          const response = await fetch(`${API_BASE_URL}/status/${watchedJobId}`);
          if (!response.ok) throw new Error(`Failed to fetch details for job ${watchedJobId}. Status: ${response.status}`);
          const data: TrainingJob = await response.json();
          setSelectedJobDetails(data);
        } catch (e: any) {
          setSelectedJobDetails(null);
          setError("Failed to fetch job details: " + e.message);
          toast({ title: "Error fetching job details", description: e.message, variant: "destructive" });
        } finally {
          setIsLoading(false);
        }
      };
      fetchJobDetails();
    } else {
      setSelectedJobDetails(null);
      setAdviceResult(null); 
    }
  }, [watchedJobId]);

  const onSubmit = async (data: AdvisorFormValues) => {
    if (!selectedJobDetails) {
      toast({ title: "Error", description: "Previous job details not loaded. Please select a job.", variant: "destructive" });
      return;
    }
    if (selectedJobDetails.status !== 'completed') {
      toast({ title: "Invalid Job", description: "Please select a 'completed' job for HNN advice.", variant: "destructive" });
      return;
    }

    setIsLoading(true);
    setError(null);
    setAdviceResult(null);

    let validatedPreviousParams: TrainingParameters;
    try {
        const paramsToValidate = {
            ...selectedJobDetails.parameters,
            totalEpochs: selectedJobDetails.parameters.totalEpochs || 0,
            batchSize: selectedJobDetails.parameters.batchSize || 0,
            learningRate: selectedJobDetails.parameters.learningRate || 0,
            weightDecay: selectedJobDetails.parameters.weightDecay || 0,
            momentumParams: selectedJobDetails.parameters.momentumParams || Array(6).fill(0),
            strengthParams: selectedJobDetails.parameters.strengthParams || Array(6).fill(0),
            noiseParams: selectedJobDetails.parameters.noiseParams || Array(6).fill(0),
            couplingParams: selectedJobDetails.parameters.couplingParams || Array(6).fill(0),
            quantumCircuitSize: selectedJobDetails.parameters.quantumCircuitSize || 0,
            labelSmoothing: selectedJobDetails.parameters.labelSmoothing || 0,
            quantumMode: selectedJobDetails.parameters.quantumMode || false,
            modelName: selectedJobDetails.parameters.modelName || "DefaultModel",
            ...(selectedJobDetails.parameters.baseConfigId !== null && selectedJobDetails.parameters.baseConfigId !== undefined
                ? { baseConfigId: selectedJobDetails.parameters.baseConfigId }
                : {}),
        };
        validatedPreviousParams = TrainingParametersSchema.parse(paramsToValidate);
    } catch (validationError: any) {
        console.error("Validation error for previousTrainingParameters:", validationError);
        setError("Previous job parameters are not in the expected format. Check console for details. Error: " + validationError.message);
        toast({ title: "Parameter Mismatch", description: "Previous job parameters are not in the expected format. " + validationError.message, variant: "destructive" });
        setIsLoading(false);
        return;
    }
    
    const inputForAI: HSQNNAdvisorInput = { 
      previousJobId: selectedJobDetails.job_id,
      previousZpeEffects: selectedJobDetails.zpe_effects || Array(6).fill(0),
      previousTrainingParameters: validatedPreviousParams, 
      hnnObjective: data.hnnObjective,
    };

    try {
      const output = await adviseHSQNNParameters(inputForAI);
      setAdviceResult(output);
      toast({ title: "Advice Generated", description: "AI has provided suggestions for the next HNN step." });
    } catch (e: any) {
      setError("AI advice generation failed: " + e.message);
      toast({ title: "Advice Generation Failed", description: e.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseParameters = () => {
    if (adviceResult?.suggestedNextTrainingParameters && selectedJobDetails) {
      const baseParams = { ...selectedJobDetails.parameters };
      const suggestedParams = adviceResult.suggestedNextTrainingParameters;
      
      const combinedParams: Partial<TrainingParameters> = { ...baseParams };

      for (const key in suggestedParams) {
        if (Object.prototype.hasOwnProperty.call(suggestedParams, key)) {
          (combinedParams as any)[key] = (suggestedParams as any)[key];
        }
      }
      
      if (suggestedParams.modelName) {
        combinedParams.modelName = suggestedParams.modelName;
      } else if (baseParams.modelName) {
        combinedParams.modelName = `${baseParams.modelName}_hnn_${Date.now().toString().slice(-4)}`;
      } else {
        combinedParams.modelName = `HNN_Model_${Date.now().toString().slice(-4)}`;
      }
      combinedParams.baseConfigId = selectedJobDetails.job_id;

      const queryParams = new URLSearchParams();
      for (const [key, value] of Object.entries(combinedParams)) {
        if (value !== undefined && value !== null) { 
          if (Array.isArray(value)) {
            queryParams.append(key, JSON.stringify(value));
          } else {
            queryParams.append(key, String(value));
          }
        }
      }
      router.push(`/train?${queryParams.toString()}`);
    }
  };
  
  const ParamList = ({ params, title }: { params: Partial<TrainingParameters> | TrainingParameters | undefined, title: string }) => {
    if (!params || Object.keys(params).length === 0) {
      let message = `${title}: No parameters to display or not applicable.`;
       if (title === "Suggested Changes" && selectedJobDetails && (!params || Object.keys(params).length === 0)) {
        message = `${title}: No specific changes suggested beyond inheriting from the previous job. A new model name and baseConfigId will be set.`;
      }
      return <p className="text-sm text-muted-foreground italic">{message}</p>;
    }

    const orderedKeys: (keyof TrainingParameters)[] = [
        "modelName", "totalEpochs", "batchSize", "learningRate", "weightDecay", 
        "labelSmoothing", "quantumMode", "quantumCircuitSize",
        "momentumParams", "strengthParams", "noiseParams", "couplingParams",
        "baseConfigId"
    ];

    return (
      <div className="space-y-1 text-sm">
         <h4 className="font-semibold text-muted-foreground">{title}:</h4>
         <ul className="list-disc list-inside pl-4 space-y-1 bg-background/50 p-2 rounded">
            {orderedKeys.map((key) => {
              const value = params[key];
              if (value === undefined || value === null) return null;
              if (title === "Suggested Changes" && key === "baseConfigId" && value === selectedJobDetails?.job_id) return null;
              if (title === "Suggested Changes" && key === "modelName" && value === selectedJobDetails?.parameters.modelName && !adviceResult?.suggestedNextTrainingParameters?.modelName) return null;

              return (
                <li key={key}>
                  <span className="font-medium">{key}:</span>{' '}
                  {Array.isArray(value) ? `[${value.map(v => typeof v === 'number' ? v.toFixed(4) : String(v)).join(', ')}]` : String(value)}
                </li>
              );
            })}
         </ul>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4 md:p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><BrainCircuit className="h-6 w-6 text-primary" />HS-QNN Parameter Advisor</CardTitle>
          <CardDescription>
            Get AI-driven advice for the next training parameters in your Hilbert Space Quantum Neural Network sequence based on a previous job&apos;s ZPE state and your objectives.
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle>Input Configuration</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <Label htmlFor="previousJobId">Select Previous Completed Job</Label>
                <Controller
                  name="previousJobId"
                  control={control}
                  render={({ field }) => (
                    <Select onValueChange={field.onChange} value={field.value || ""} disabled={isLoadingJobs}>
                      <SelectTrigger id="previousJobId">
                        <SelectValue placeholder={isLoadingJobs ? "Loading jobs..." : "Select a completed job"} />
                      </SelectTrigger>
                      <SelectContent>
                        {isLoadingJobs && <SelectItem value="loading" disabled>Loading jobs...</SelectItem>}
                        {!isLoadingJobs && jobsList.length === 0 && <SelectItem value="no-jobs" disabled>No completed jobs found.</SelectItem>}
                        {jobsList.map(job => (
                          <SelectItem key={job.job_id} value={job.job_id}>
                            {job.model_name} ({job.job_id.slice(-8)}) - Acc: {job.accuracy.toFixed(2)}%
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
                {errors.previousJobId && <p className="text-xs text-destructive mt-1">{errors.previousJobId.message}</p>}
              </div>

              {selectedJobDetails && (
                <Card className="bg-muted/50">
                  <CardHeader className="pb-2 pt-4"><CardTitle className="text-base">Previous Job Summary</CardTitle></CardHeader>
                  <CardContent className="text-sm space-y-1">
                    <p><strong>Model:</strong> {selectedJobDetails.parameters.modelName}</p>
                    <p><strong>Accuracy:</strong> {selectedJobDetails.accuracy.toFixed(2)}%</p>
                    <p><strong>Loss:</strong> {selectedJobDetails.loss.toFixed(4)}</p>
                    <p><strong>ZPE Effects (avg):</strong> [{selectedJobDetails.zpe_effects?.map(z => z.toFixed(3)).join(', ') || 'N/A'}]</p>
                     <details className="mt-2">
                        <summary className="cursor-pointer text-xs text-muted-foreground hover:underline">View All Previous Parameters</summary>
                        <ScrollArea className="h-32 mt-1 border p-2 rounded-md bg-background">
                           <ParamList params={selectedJobDetails.parameters} title="Previous Parameters"/>
                        </ScrollArea>
                    </details>
                  </CardContent>
                </Card>
              )}

              <div>
                <Label htmlFor="hnnObjective">HNN Objective for Next Step</Label>
                <Controller
                  name="hnnObjective"
                  control={control}
                  render={({ field }) => (
                    <Textarea
                      {...field}
                      id="hnnObjective"
                      rows={4}
                      placeholder="e.g., Maximize accuracy while exploring higher ZPE for layer 3..."
                      className="text-sm"
                    />
                  )}
                />
                {errors.hnnObjective && <p className="text-xs text-destructive mt-1">{errors.hnnObjective.message}</p>}
              </div>

              <Button type="submit" className="w-full" disabled={isLoading || !selectedJobDetails || (selectedJobDetails && selectedJobDetails.status !== 'completed')}>
                {isLoading ? <RefreshCw className="mr-2 h-4 w-4 animate-spin"/> : <Wand2 className="mr-2 h-4 w-4" />}
                Get HNN Advice
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-primary"/>AI Generated Advice</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <Terminal className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            {!isLoading && !adviceResult && !error && (
              <p className="text-muted-foreground text-center py-10">Submit configuration to get AI advice for the next HNN step.</p>
            )}
            {adviceResult && (
              <>
                <Card>
                  <CardHeader><CardTitle className="text-lg flex items-center gap-2"><SlidersHorizontal className="h-5 w-5"/>Suggested Parameter Changes</CardTitle></CardHeader>
                  <CardContent>
                    <ScrollArea className="max-h-60">
                      <ParamList params={adviceResult.suggestedNextTrainingParameters} title="Suggested Changes"/>
                       <p className="text-xs text-muted-foreground mt-2">
                        Note: AI suggested changes are shown. Other parameters will typically be inherited.
                        A new model name ({adviceResult.suggestedNextTrainingParameters?.modelName || `${selectedJobDetails?.parameters?.modelName}_hnn_${Date.now().toString().slice(-4)}`}) and baseConfigId will be set.
                      </p>
                    </ScrollArea>
                  </CardContent>
                   <CardFooter>
                    <Button 
                      onClick={handleUseParameters} 
                      className="w-full"
                      disabled={!selectedJobDetails || !adviceResult?.suggestedNextTrainingParameters}
                    >
                      <ArrowRight className="mr-2 h-4 w-4"/> Use these parameters in Train Model
                    </Button>
                  </CardFooter>
                </Card>
                <Card>
                  <CardHeader><CardTitle className="text-lg">Reasoning from AI</CardTitle></CardHeader>
                  <CardContent>
                    <ScrollArea className="h-48">
                      <p className="text-sm whitespace-pre-wrap">{adviceResult.reasoning}</p>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

