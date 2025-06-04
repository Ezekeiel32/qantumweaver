
"use client";
import React, { useState, useEffect, useCallback } from "react";
import type { TrainingParameters, TrainingJob, TrainingJobSummary } from "@/types/training";
import type { ModelConfig as SavedModelConfig } from "@/types/entities";
import {
  Settings, Plus, Trash, Copy, Eye, Layers, RefreshCw, Play, Filter, ListChecks, Info, BrainCircuit, MessageSquare, Send, Loader2, Wand2, SlidersHorizontal
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { format, formatDistanceToNow } from "date-fns";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { toast } from "@/hooks/use-toast";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

// Unified type for display
interface ZpeAiModelListItem {
  id: string;
  name: string;
  type: "config" | "job";
  status?: TrainingJob["status"];
  accuracy: number;
  date: string; // ISO string
  parameters: TrainingParameters;
  jobDetails?: TrainingJobSummary; // For jobs
  configDetails?: SavedModelConfig; // For configs
  rawItem: TrainingJobSummary | SavedModelConfig; // Store the original item
}

const defaultZPEParamsArrays: Pick<TrainingParameters, "momentumParams" | "strengthParams" | "noiseParams" | "couplingParams"> = {
  momentumParams: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
  strengthParams: [0.35, 0.33, 0.31, 0.6, 0.27, 0.5],
  noiseParams: [0.3, 0.28, 0.26, 0.35, 0.22, 0.25],
  couplingParams: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7],
};

const initialNewConfigParameters: TrainingParameters = {
  totalEpochs: 30,
  batchSize: 32,
  learningRate: 0.001,
  weightDecay: 0.0001,
  ...defaultZPEParamsArrays,
  quantumCircuitSize: 32,
  labelSmoothing: 0.1,
  quantumMode: true,
  modelName: "ZPE-New-Config",
  baseConfigId: null,
};

export default function MyZpeAiModelsPage() {
  const [allItems, setAllItems] = useState<ZpeAiModelListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ZpeAiModelListItem | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
  const [showDeletePrompt, setShowDeletePrompt] = useState(false);
  const router = useRouter();

  const [newConfigParams, setNewConfigParams] = useState<TrainingParameters>(initialNewConfigParameters);
  const [newConfigName, setNewConfigName] = useState<string>("ZPE-Custom-Config");

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [jobsResponse, configsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/jobs?limit=100`),
        fetch(`${API_BASE_URL}/configs`),
      ]);

      if (!jobsResponse.ok) throw new Error("Failed to fetch jobs list");
      if (!configsResponse.ok) throw new Error("Failed to fetch configurations");

      const jobsData = await jobsResponse.json();
      const configsData = await configsResponse.json();

      const jobItems: ZpeAiModelListItem[] = (jobsData.jobs || []).map((job: TrainingJobSummary) => ({
        id: job.job_id,
        name: job.model_name,
        type: "job" as "job",
        status: job.status,
        accuracy: job.accuracy || 0,
        date: job.start_time || new Date(0).toISOString(),
        parameters: job.jobDetails?.parameters || initialNewConfigParameters, // Placeholder, full job details needed for actual params
        jobDetails: job,
        rawItem: job,
      }));

      const configItems: ZpeAiModelListItem[] = (configsData.configs || []).map((config: SavedModelConfig) => ({
        id: config.id!,
        name: config.name,
        type: "config" as "config",
        accuracy: config.accuracy || 0,
        date: config.date_created,
        parameters: config.parameters,
        configDetails: config,
        rawItem: config,
      }));
      
      // Fetch full details for jobs to get parameters (if not already in summary)
      // This is a bit inefficient for a large number of jobs, ideally summary would have more
      // For now, this ensures parameters are available for "View Params"
      const detailedJobItems = await Promise.all(jobItems.map(async (jobItem) => {
        if (jobItem.type === 'job' && jobItem.jobDetails) {
          try {
            const detailRes = await fetch(`${API_BASE_URL}/status/${jobItem.id}`);
            if (detailRes.ok) {
              const fullJob: TrainingJob = await detailRes.json();
              return { ...jobItem, parameters: fullJob.parameters };
            }
          } catch (e) { console.error(`Failed to fetch details for job ${jobItem.id}`, e); }
        }
        return jobItem;
      }));


      const combinedItems = [...detailedJobItems, ...configItems].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
      setAllItems(combinedItems);

    } catch (error: any) {
      toast({ title: "Error fetching data", description: error.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleViewItemDetails = (item: ZpeAiModelListItem) => {
    setSelectedItem(item);
    setIsDetailModalOpen(true);
  };

  const handleCreateConfig = async () => {
    setIsSubmitting(true);
    try {
      const configPayload: Omit<SavedModelConfig, 'id' | 'accuracy' | 'loss' | 'date_created'> & { parameters: TrainingParameters, name: string, use_quantum_noise: boolean } = {
        name: newConfigName || `ZPE-Config-${Date.now().toString().slice(-4)}`,
        parameters: {
          ...newConfigParams,
          modelName: newConfigParams.modelName || newConfigName, // Ensure modelName inside parameters is also set
        },
        use_quantum_noise: newConfigParams.quantumMode,
        channel_sizes: [64, 128, 256, 512], // Default or make configurable
      };
      
      // Backend ModelConfig expects 'parameters' without 'couplingParams' if API is strict
      // For now, assuming the backend handles or ignores extra fields, or it's already aligned.
      // If backend errors on unexpected 'couplingParams', filter it out:
      // const { couplingParams, ...paramsForBackend } = configPayload.parameters;
      // const payloadToSend = { ...configPayload, parameters: paramsForBackend };

      const response = await fetch(`${API_BASE_URL}/configs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configPayload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status}. ${errorData.detail || JSON.stringify(errorData)}`);
      }
      await fetchData();
      setNewConfigParams(initialNewConfigParameters);
      setNewConfigName("ZPE-Custom-Config");
      setIsCreating(false);
      toast({ title: "Success", description: `Configuration "${configPayload.name}" created.` });
    } catch (error: any) {
      console.error("Error creating configuration:", error);
      toast({ title: "Error", description: `Failed to create configuration. ${error.message || ''}`, variant: "destructive" });
    }
    setIsSubmitting(false);
  };

  const handleCloneItem = async (itemToClone: ZpeAiModelListItem) => {
    try {
      const clonedParameters: TrainingParameters = {
        ...itemToClone.parameters,
        modelName: `${itemToClone.parameters.modelName}-Clone-${Date.now().toString().slice(-4)}`,
        baseConfigId: itemToClone.type === 'job' ? itemToClone.id : itemToClone.parameters.baseConfigId,
      };

      const configPayload: Omit<SavedModelConfig, 'id' | 'accuracy' | 'loss' | 'date_created'> & { parameters: TrainingParameters, name: string, use_quantum_noise: boolean } = {
        name: clonedParameters.modelName,
        parameters: clonedParameters,
        use_quantum_noise: clonedParameters.quantumMode,
        channel_sizes: itemToClone.configDetails?.channel_sizes || [64,128,256,512],
      };
      
      // const { couplingParams, ...paramsForBackend } = configPayload.parameters;
      // const payloadToSend = { ...configPayload, parameters: paramsForBackend };

      const response = await fetch(`${API_BASE_URL}/configs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configPayload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to clone: ${errorData.detail || response.statusText}`);
      }
      await fetchData();
      toast({ title: "Success", description: `Configuration "${configPayload.name}" cloned.` });
    } catch (error: any) {
      console.error("Error cloning item:", error);
      toast({ title: "Error", description: `Failed to clone. ${error.message || ''}`, variant: "destructive" });
    }
  };

  const confirmDeleteItem = async () => {
    if (!selectedItem || selectedItem.type !== 'config' || !selectedItem.id) {
      toast({ title: "Cannot Delete", description: "Only saved configurations can be deleted, or item not properly selected.", variant: "destructive" });
      setShowDeletePrompt(false);
      return;
    }
    const configNameToDelete = selectedItem.name;
    const currentSelectedId = selectedItem.id;

    try {
      const response = await fetch(`${API_BASE_URL}/configs/${selectedItem.id}`, { method: 'DELETE' });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to delete, server error." }));
        throw new Error(`Failed to delete configuration: ${response.status} - ${errorData.detail}`);
      }
      await fetchData(); // Refetch to update the list
      setSelectedItem(null); // Clear selection
      toast({ title: "Success", description: `Configuration "${configNameToDelete}" deleted.` });
    } catch (error: any) {
      console.error("Error deleting configuration:", error);
      toast({ title: "Error", description: `Failed to delete configuration. ${error.message || ''}`, variant: "destructive" });
    } finally {
      setShowDeletePrompt(false);
    }
  };

  const handleLoadInTrainer = (item: ZpeAiModelListItem) => {
    const paramsToPrefill: Partial<TrainingParameters> = { ...item.parameters };
    paramsToPrefill.modelName = `${item.parameters.modelName}_from_${item.type}`;
    paramsToPrefill.baseConfigId = item.type === 'config' ? item.id : item.id; // Use job_id as base if it's a job

    const queryParams = new URLSearchParams();
    for (const [key, value] of Object.entries(paramsToPrefill)) {
      if (value !== undefined && value !== null) {
        if (Array.isArray(value)) {
          queryParams.append(key, JSON.stringify(value));
        } else {
          queryParams.append(key, String(value));
        }
      }
    }
    router.push(`/train?${queryParams.toString()}`);
    toast({ title: "Loading in Trainer", description: `Parameters for "${item.name}" pre-filled.` });
  };
  
  const updateNewConfigParamArray = (paramType: keyof TrainingParameters, index: number, value: number) => {
    setNewConfigParams(prev => {
      const currentParams = prev;
      const paramArray = currentParams[paramType] as number[] | undefined;

      if (Array.isArray(paramArray)) {
        const updatedArray = [...paramArray];
        updatedArray[index] = value;
        return { ...currentParams, [paramType]: updatedArray, };
      }
      return prev;
    });
  };

  const ParamArrayDisplay = ({ label, values }: { label: string; values: number[] | undefined }) => (
    <div>
      <h4 className="font-medium text-sm">{label}:</h4>
      {values && values.length > 0 ? (
        <p className="text-xs font-mono bg-muted p-1 rounded overflow-x-auto whitespace-nowrap">
          [{values.map(v => v.toFixed(3)).join(", ")}]
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">N/A</p>
      )}
    </div>
  );

  const filteredItems = (typeFilter: "all" | "config" | "job") => {
    if (typeFilter === "all") return allItems;
    return allItems.filter(item => item.type === typeFilter);
  };

  return (
    <div className="container mx-auto p-4 md:p-6">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-2 md:space-y-0 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary flex items-center gap-2"><ListChecks /> My ZPE-AI Models</h1>
          <p className="text-muted-foreground">Manage saved configurations and review past training job history.</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={fetchData} variant="outline" disabled={isLoading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} /> Refresh Lists
          </Button>
          <Button variant="default" onClick={() => setIsCreating(!isCreating)}>
            {isCreating ? "Cancel" : <><Plus className="h-4 w-4 mr-2" />Create New Config</>}
          </Button>
        </div>
      </div>

      {isCreating && (
        <Card className="mb-6">
          <CardHeader><CardTitle>Create New Configuration</CardTitle><CardDescription>Define parameters for a new model configuration</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2"><Label htmlFor="config-name">Config Name</Label><Input id="config-name" placeholder="e.g., ZPE-ResNet-Variant1" value={newConfigName} onChange={(e) => setNewConfigName(e.target.value)}/></div>
            <div className="space-y-2"><div className="flex items-center justify-between"><Label htmlFor="quantum-noise">Use Quantum Noise</Label><Switch id="quantum-noise" checked={newConfigParams.quantumMode} onCheckedChange={(checked) => setNewConfigParams(p => ({...p, quantumMode: checked}))}/></div></div>
            
            <Tabs defaultValue="general" className="mt-6">
              <TabsList className="grid grid-cols-3">
                <TabsTrigger value="general">General</TabsTrigger>
                <TabsTrigger value="zpe">ZPE Params</TabsTrigger>
                <TabsTrigger value="architecture">Architecture</TabsTrigger>
              </TabsList>
              <TabsContent value="general" className="space-y-3 pt-3">
                 <div className="grid grid-cols-2 gap-4">
                   <div className="space-y-2"><Label htmlFor="totalEpochs">Total Epochs</Label><Input id="totalEpochs" type="number" value={newConfigParams.totalEpochs} onChange={(e) => setNewConfigParams(p => ({...p, totalEpochs: parseInt(e.target.value,10)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="batchSize">Batch Size</Label><Input id="batchSize" type="number" value={newConfigParams.batchSize} onChange={(e) => setNewConfigParams(p => ({...p, batchSize: parseInt(e.target.value,10)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="learningRate">Learning Rate</Label><Input id="learningRate" type="number" step="0.0001" value={newConfigParams.learningRate} onChange={(e) => setNewConfigParams(p => ({...p, learningRate: parseFloat(e.target.value)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="weightDecay">Weight Decay</Label><Input id="weightDecay" type="number" step="0.0001" value={newConfigParams.weightDecay} onChange={(e) => setNewConfigParams(p => ({...p, weightDecay: parseFloat(e.target.value)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="quantumCircuitSize">Quantum Circuit Size</Label><Input id="quantumCircuitSize" type="number" value={newConfigParams.quantumCircuitSize} onChange={(e) => setNewConfigParams(p => ({...p, quantumCircuitSize: parseInt(e.target.value,10)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="labelSmoothing">Label Smoothing</Label><Input id="labelSmoothing" type="number" step="0.01" value={newConfigParams.labelSmoothing} onChange={(e) => setNewConfigParams(p => ({...p, labelSmoothing: parseFloat(e.target.value)||0}))}/></div>
                   <div className="space-y-2"><Label htmlFor="modelNameParam">Model Name (Internal)</Label><Input id="modelNameParam" type="text" value={newConfigParams.modelName} onChange={(e) => setNewConfigParams(p => ({...p, modelName: e.target.value}))}/></div>
                 </div>
              </TabsContent>
              <TabsContent value="zpe" className="space-y-3 pt-3">
                {(["momentumParams", "strengthParams", "noiseParams", "couplingParams"] as (keyof TrainingParameters)[]).map((paramKey) => (
                  <div key={paramKey as string} className="space-y-3">
                    <h4 className="text-sm font-medium capitalize">{paramKey.replace('Params', '')} Params</h4>
                    {(newConfigParams[paramKey] as number[]).map((value, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs"><Label>Layer {idx + 1}</Label><span className="font-mono">{typeof value === 'number' ? value.toFixed(2) : 'N/A'}</span></div>
                        <Slider min={0} max={1} step={0.01} value={[typeof value === 'number' ? value : 0]} onValueChange={(newValue) => updateNewConfigParamArray(paramKey, idx, newValue[0])}/>
                      </div>
                    ))}
                  </div>
                ))}
              </TabsContent>
               <TabsContent value="architecture" className="space-y-3 pt-3">
                  <p className="text-sm text-muted-foreground">Channel sizes are defaulted. This section can be expanded for more architecture controls.</p>
              </TabsContent>
            </Tabs>
          </CardContent>
          <CardFooter className="border-t pt-6"><Button onClick={handleCreateConfig} className="w-full" disabled={isSubmitting}>{isSubmitting ? "Creating..." : <><Plus className="h-4 w-4 mr-2" />Create Config</>}</Button></CardFooter>
        </Card>
      )}

      <Tabs defaultValue="all">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="all"><Filter className="mr-2 h-4 w-4"/>All ({allItems.length})</TabsTrigger>
          <TabsTrigger value="config"><Settings className="mr-2 h-4 w-4"/>Saved Configs ({filteredItems('config').length})</TabsTrigger>
          <TabsTrigger value="job"><Layers className="mr-2 h-4 w-4"/>Job History ({filteredItems('job').length})</TabsTrigger>
        </TabsList>

        {["all", "config", "job"].map(tabValue => (
          <TabsContent key={tabValue} value={tabValue}>
            <Card>
              <CardHeader>
                <CardTitle>
                  {tabValue === "all" ? "All Models & Jobs" : tabValue === "config" ? "Saved Configurations" : "Training Job History"}
                </CardTitle>
                <CardDescription>
                  {tabValue === "all" ? "Combined list of saved configurations and past training jobs." :
                   tabValue === "config" ? "Manage your saved model parameter sets." :
                   "Review parameters and outcomes of past training sessions."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading && <div className="flex justify-center py-10"><RefreshCw className="h-8 w-8 animate-spin text-primary" /></div>}
                {!isLoading && filteredItems(tabValue as "all" | "config" | "job").length === 0 && (
                  <div className="text-center py-10 text-muted-foreground">
                    <ListChecks className="mx-auto h-12 w-12 mb-4" />
                    No items found for this view.
                  </div>
                )}
                {!isLoading && filteredItems(tabValue as "all" | "config" | "job").length > 0 && (
                  <ScrollArea className="h-[600px]">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Status/Acc.</TableHead>
                          <TableHead>Date</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredItems(tabValue as "all" | "config" | "job").map(item => (
                          <TableRow key={item.id}>
                            <TableCell className="font-medium">{item.name}</TableCell>
                            <TableCell>
                              <Badge variant={item.type === "config" ? "secondary" : "outline"}>
                                {item.type === "config" ? "Config" : "Job"}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              {item.type === "job" && item.status && (
                                <Badge variant={
                                  item.status === "running" ? "default" :
                                  item.status === "completed" ? "default" :
                                  item.status === "failed" || item.status === "stopped" ? "destructive" : "secondary"
                                }
                                className={cn(item.status === "completed" ? "bg-green-500 hover:bg-green-600 text-primary-foreground" : item.status === "running" ? "bg-blue-500 hover:bg-blue-600 text-primary-foreground animate-pulse" : "")}
                                >{item.status}</Badge>
                              )}
                              {item.type === "config" && `${item.accuracy.toFixed(2)}%`}
                              {item.type === "job" && item.status !== "running" && item.status !== "pending" && ` / ${item.accuracy.toFixed(2)}%`}
                            </TableCell>
                            <TableCell>
                              {item.date ? (
                                item.type === 'job' && item.jobDetails?.start_time ? formatDistanceToNow(new Date(item.jobDetails.start_time), { addSuffix: true }) :
                                format(new Date(item.date), "MMM d, yyyy")
                              ) : 'N/A'}
                            </TableCell>
                            <TableCell className="text-right space-x-1">
                              <Button variant="outline" size="sm" onClick={() => handleViewItemDetails(item)}><Eye className="mr-1 h-3 w-3"/>View</Button>
                              <Button variant="outline" size="sm" onClick={() => handleLoadInTrainer(item)}><Play className="mr-1 h-3 w-3"/>Train</Button>
                              <Button variant="outline" size="sm" onClick={() => handleCloneItem(item)}><Copy className="mr-1 h-3 w-3"/>Clone</Button>
                              {item.type === "config" && <Button variant="destructive" size="sm" onClick={() => { setSelectedItem(item); setShowDeletePrompt(true);}}><Trash className="mr-1 h-3 w-3"/>Delete</Button>}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      {selectedItem && (
        <Dialog open={isDetailModalOpen} onOpenChange={setIsDetailModalOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5 text-primary"/> {selectedItem.type === 'config' ? 'Config' : 'Job'} Parameters: {selectedItem.name}
              </DialogTitle>
              <DialogDescription>
                Configuration used for {selectedItem.type}: <span className="font-mono text-xs">{selectedItem.id.replace('zpe_job_','').replace('config_','')}</span>
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh] p-1 pr-3">
              <div className="space-y-3 text-sm py-4">
                <h3 className="font-semibold text-base mb-2 border-b pb-1">General Settings</h3>
                <p><strong>Total Epochs:</strong> {selectedItem.parameters.totalEpochs}</p>
                <p><strong>Batch Size:</strong> {selectedItem.parameters.batchSize}</p>
                <p><strong>Learning Rate:</strong> {selectedItem.parameters.learningRate}</p>
                <p><strong>Weight Decay:</strong> {selectedItem.parameters.weightDecay}</p>
                <p><strong>Label Smoothing:</strong> {selectedItem.parameters.labelSmoothing}</p>
                
                <h3 className="font-semibold text-base mt-3 mb-2 border-b pb-1">ZPE Settings</h3>
                <ParamArrayDisplay label="Momentum Params" values={selectedItem.parameters.momentumParams} />
                <ParamArrayDisplay label="Strength Params" values={selectedItem.parameters.strengthParams} />
                <ParamArrayDisplay label="Noise Params" values={selectedItem.parameters.noiseParams} />
                <ParamArrayDisplay label="Coupling Params" values={selectedItem.parameters.couplingParams} />
                
                <h3 className="font-semibold text-base mt-3 mb-2 border-b pb-1">Quantum Settings</h3>
                <p><strong>Quantum Mode:</strong> {selectedItem.parameters.quantumMode ? "Enabled" : "Disabled"}</p>
                <p><strong>Quantum Circuit Size:</strong> {selectedItem.parameters.quantumCircuitSize} Qubits</p>

                {selectedItem.parameters.baseConfigId && <p className="mt-2"><strong>Base Config ID:</strong> <span className="font-mono text-xs">{selectedItem.parameters.baseConfigId}</span></p>}
                 <div className="mt-4 pt-4 border-t">
                    <Button asChild variant="outline" size="sm" onClick={() => handleLoadInTrainer(selectedItem)}>
                        {/* Link component removed, direct call to handleLoadInTrainer */}
                        <span><RefreshCw className="mr-2 h-4 w-4"/> Retrain with these parameters</span>
                    </Button>
                 </div>
              </div>
            </ScrollArea>
            <DialogFooter>
              <DialogClose asChild><Button variant="outline">Close</Button></DialogClose>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {showDeletePrompt && selectedItem && selectedItem.type === 'config' && (
        <Dialog open={showDeletePrompt} onOpenChange={setShowDeletePrompt}>
          <DialogContent>
            <DialogHeader><DialogTitle>Confirm Deletion</DialogTitle>
            <DialogDescription>Are you sure you want to delete configuration "{selectedItem.name}"? This action cannot be undone.</DialogDescription></DialogHeader>
            <DialogFooter className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowDeletePrompt(false)}>Cancel</Button>
              <Button variant="destructive" onClick={confirmDeleteItem}>Delete</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

       <Card className="mt-6 bg-accent/30 border-accent">
        <CardHeader>
            <CardTitle className="flex items-center gap-2"><Info className="h-5 w-5 text-accent-foreground"/>Understanding Your ZPE-AI Models</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-accent-foreground/80 space-y-2">
            <p>This page combines your saved model configurations and training job history.</p>
            <p><strong>Saved Configurations:</strong> These are parameter sets you've explicitly created or cloned. You can load them into the trainer, clone, or delete them.</p>
            <p><strong>Job History:</strong> These are records of past training runs. You can view their parameters and load them into the trainer to start a new run based on that job's settings.</p>
            <p>Use the tabs to filter between views. The "All" tab shows a combined, chronologically sorted list.</p>
        </CardContent>
      </Card>

      {/* Placeholder for HS-QNN Mini Chat - To be implemented in a future step */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><BrainCircuit className="h-5 w-5 text-primary"/>HS-QNN Advisor Chat</CardTitle>
          <CardDescription>Interact with an AI to get advice on evolving your model parameters for the next training sequence. Select a model/job from the list above to load its context into the chat.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center bg-muted rounded-md">
            <p className="text-muted-foreground">HS-QNN Mini Chat functionality coming soon!</p>
          </div>
        </CardContent>
      </Card>

    </div>
  );
}

    
      