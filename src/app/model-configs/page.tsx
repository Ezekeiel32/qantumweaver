
"use client";
import React, { useState, useEffect, useCallback } from "react";
// import { ModelConfig } from "@/entities/all"; // Commented out
import type { ModelConfig } from "@/types/entities";
import {
  Settings, Plus, Trash, Copy, Eye, EyeOff, SlidersHorizontal, ChevronDown, CheckCircle, XCircle, Zap, Atom, Layers
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { format } from "date-fns";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { toast } from "@/hooks/use-toast"; 

// Define a type for the parameters that will be nested
interface TrainingParameters {
  totalEpochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  momentumParams: number[];
  strengthParams: number[];
  noiseParams: number[];
  couplingParams: number[]; // Added couplingParams
  quantumCircuitSize: number;
  labelSmoothing: number;
  quantumMode: boolean;
  modelName: string;
  baseConfigId?: string | null;
}

// Update ModelConfig type definition to match backend
interface FrontendModelConfig {
  id?: string | null;
  name: string;
  channel_sizes: number[];
  parameters: TrainingParameters; // Nested parameters object
  accuracy: number;
  date_created: string;
}


export default function ModelConfigurationsPage() {
  const [configs, setConfigs] = useState<FrontendModelConfig[]>([]); 
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<FrontendModelConfig | null>(null); 
  const [showDeletePrompt, setShowDeletePrompt] = useState(false);
  const [shownParameters, setShownParameters] = useState<Record<string, boolean>>({});

  const initialNewConfig: FrontendModelConfig = {
    name: "",
    channel_sizes: [64, 128, 256, 512],
    parameters: {
      totalEpochs: 30,
      batchSize: 32,
      learningRate: 0.001,
      weightDecay: 0.0001,
      momentumParams: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
      strengthParams: [0.35, 0.33, 0.31, 0.6, 0.27, 0.5],
      noiseParams: [0.3, 0.28, 0.26, 0.35, 0.22, 0.25],
      couplingParams: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7], // Added default couplingParams
      quantumCircuitSize: 32,
      labelSmoothing: 0.1,
      quantumMode: true,
      modelName: "ZPE-Sim-V1", 
      baseConfigId: null,
    },
    accuracy: 98.5, 
    date_created: new Date().toISOString().split('T')[0]
  };
  const [newConfig, setNewConfig] = useState<FrontendModelConfig>(initialNewConfig); 

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Placeholder: Replace with actual API call
      const modelConfigsData: FrontendModelConfig[] = []; 
      setConfigs(modelConfigsData);
      if (modelConfigsData.length > 0) {
        setSelectedConfig(modelConfigsData[0]);
        const initialShown: Record<string, boolean> = {};
        modelConfigsData.forEach(config => {
          if(config.id) initialShown[config.id] = false;
        });
        setShownParameters(initialShown);
      }
    } catch (error) {
      console.error("Error fetching configurations:", error);
      toast({ title: "Error", description: "Failed to fetch configurations.", variant: "destructive"});
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const toggleParameterVisibility = (configId: string) => {
    setShownParameters(prev => ({ ...prev, [configId]: !prev[configId] }));
  };

  const handleConfigSelect = (config: FrontendModelConfig) => { setSelectedConfig(config); };

  const handleCreateConfig = async () => {
    setIsSubmitting(true);
    try {
      const configPayload = {
        ...newConfig,
        name: newConfig.name || `ZPE-Config-${Date.now().toString().slice(-4)}`,
      };
      const response = await fetch('/api/model-configs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(configPayload),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const createdConfig: FrontendModelConfig = await response.json();
 setConfigs(prev => [...prev, createdConfig]);
      setSelectedConfig(createdConfig);
      if(createdConfig.id) setShownParameters(prev => ({ ...prev, [createdConfig.id!]: false }));
      setNewConfig(initialNewConfig); 
      setIsCreating(false);
      toast({ title: "Success", description: `Configuration "${createdConfig.name}" created.`});
    } catch (error: any) { 
      console.error("Error creating configuration:", error);
      toast({ title: "Error", description: `Failed to create configuration. ${error.message || ''}`, variant: "destructive"});
    }
    setIsSubmitting(false);
  };

  const handleCloneConfig = async (config: FrontendModelConfig) => { 
    try {
      const clonedPayload = {
        ...config, // Clone all properties
        name: `${config.name}-Clone-${Date.now().toString().slice(-4)}`,
        date_created: new Date().toISOString().split('T')[0],
        id: undefined, // Ensure new ID will be assigned by backend or placeholder
      };
      // Placeholder for API call
      const createdConfig: FrontendModelConfig = {...clonedPayload, id: `demo-clone-${Date.now()}`};
      setConfigs(prev => [...prev, createdConfig]);
      if(createdConfig.id) setShownParameters(prev => ({ ...prev, [createdConfig.id!]: false }));
      toast({ title: "Success", description: `Configuration "${createdConfig.name}" cloned.`});
    } catch (error: any) { 
      console.error("Error cloning configuration:", error);
      toast({ title: "Error", description: `Failed to clone configuration. ${error.message || ''}`, variant: "destructive"});
    }
  };

  const confirmDeleteConfig = async () => {
    if (!selectedConfig || !selectedConfig.id) return;
    try {
      // Placeholder for API call
      console.log(`Simulating delete for config ID: ${selectedConfig.id}`);
      setConfigs(prev => prev.filter(config => config.id !== selectedConfig.id));
      setSelectedConfig(configs.length > 1 ? configs.find(c => c.id !== selectedConfig.id) || null : null);
      setShowDeletePrompt(false);
      toast({ title: "Success", description: `Configuration "${selectedConfig.name}" deleted.`});
    } catch (error: any) { 
      console.error("Error deleting configuration:", error);
      toast({ title: "Error", description: `Failed to delete configuration. ${error.message || ''}`, variant: "destructive"});
    }
  };

  const updateNewConfigParam = (paramType: keyof TrainingParameters, index: number, value: number) => {
    setNewConfig(prev => {
      const currentParams = prev.parameters || initialNewConfig.parameters;
      const paramArray = currentParams[paramType] as number[] | undefined;

      if (Array.isArray(paramArray)) {
        const updatedArray = [...paramArray];
        updatedArray[index] = value;
        return {
          ...prev,
          parameters: {
            ...currentParams,
            [paramType]: updatedArray,
          },
        };
      }
      console.warn(`Parameter ${String(paramType)} is not an array or not found in newConfig.parameters`);
      return prev;
    });
  };

   const handleQuantumNoiseChange = (checked: boolean) => {
     setNewConfig(prev => ({
       ...prev,
       parameters: {
         ...(prev.parameters || initialNewConfig.parameters),
         quantumMode: checked,
       },
     }));
   };

  const formatArray = (arr?: number[]) => {
    if (!arr || !Array.isArray(arr)) return "[]";
    return `[${arr.map(v => typeof v === 'number' ? v.toFixed(2) : String(v)).join(', ')}]`;
  };


  return (
    <div className="p-6 bg-background text-foreground">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col space-y-2 mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Model Configurations</h1>
          <p className="text-muted-foreground">Manage and compare different neural network configurations</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-1/3 space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Configurations</CardTitle>
                <Button variant="outline" size="sm" onClick={() => setIsCreating(!isCreating)}>
                  {isCreating ? "Cancel" : <><Plus className="h-4 w-4 mr-2" />Create New</>}
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader className="sticky top-0 bg-card z-10">
                      <TableRow><TableHead>Name</TableHead><TableHead>Accuracy</TableHead><TableHead>Date</TableHead><TableHead className="w-12"></TableHead></TableRow>
                    </TableHeader>
                    <TableBody>
                      {isLoading && <TableRow><TableCell colSpan={4} className="text-center py-6">Loading...</TableCell></TableRow>}
                      {!isLoading && configs.length === 0 && <TableRow><TableCell colSpan={4} className="text-center py-6 text-muted-foreground">No configurations found.</TableCell></TableRow>}
                      {configs.map((config) => (
                        <TableRow key={config.id} className={`cursor-pointer hover:bg-muted/50 ${selectedConfig?.id === config.id ? 'bg-muted' : ''}`} onClick={() => handleConfigSelect(config)}>
                          <TableCell className="font-medium"><div className="flex items-center gap-2"><Layers className="h-4 w-4 text-muted-foreground" />{config.name}</div></TableCell>
                          <TableCell><span className="font-mono">{config.accuracy}%</span></TableCell>
                          <TableCell>{format(new Date(config.date_created), "MMM d, yyyy")}</TableCell>
                          <TableCell><Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); config.id && toggleParameterVisibility(config.id); }}>{config.id && shownParameters[config.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</Button></TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </CardContent>
            </Card>

            {isCreating && (
              <Card>
                <CardHeader><CardTitle>Create New Configuration</CardTitle><CardDescription>Define parameters for a new model configuration</CardDescription></CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2"><Label htmlFor="config-name">Name</Label><Input id="config-name" placeholder="Config Name" value={newConfig.name} onChange={(e) => setNewConfig(prev => ({ ...prev, name: e.target.value }))}/></div>
                  <div className="space-y-2"><div className="flex items-center justify-between"><Label htmlFor="quantum-noise">Use Quantum Noise</Label><Switch id="quantum-noise" checked={newConfig.parameters.quantumMode} onCheckedChange={handleQuantumNoiseChange}/></div></div>
                  <Tabs defaultValue="general" className="mt-6">
                    <TabsList className="grid grid-cols-3">
                      <TabsTrigger value="general">General</TabsTrigger>
                      <TabsTrigger value="zpe">ZPE Params</TabsTrigger>
                      <TabsTrigger value="architecture">Architecture</TabsTrigger>
                    </TabsList>
                    <TabsContent value="general" className="space-y-3 pt-3">
                       <div className="grid grid-cols-2 gap-4">
                         <div className="space-y-2">
                           <Label htmlFor="totalEpochs">Total Epochs</Label>
                           <Input id="totalEpochs" type="number" value={newConfig.parameters.totalEpochs} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, totalEpochs: parseInt(e.target.value)}}))}/>
                         </div>
                          <div className="space-y-2">
                           <Label htmlFor="batchSize">Batch Size</Label>
                           <Input id="batchSize" type="number" value={newConfig.parameters.batchSize} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, batchSize: parseInt(e.target.value)}}))}/>
                         </div>
                          <div className="space-y-2">
                           <Label htmlFor="learningRate">Learning Rate</Label>
                           <Input id="learningRate" type="number" step="0.0001" value={newConfig.parameters.learningRate} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, learningRate: parseFloat(e.target.value)}}))}/>
                         </div>
                          <div className="space-y-2">
                           <Label htmlFor="weightDecay">Weight Decay</Label>
                           <Input id="weightDecay" type="number" step="0.0001" value={newConfig.parameters.weightDecay} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, weightDecay: parseFloat(e.target.value)}}))}/>
                         </div>
                           <div className="space-y-2">
                           <Label htmlFor="quantumCircuitSize">Quantum Circuit Size</Label>
                           <Input id="quantumCircuitSize" type="number" value={newConfig.parameters.quantumCircuitSize} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, quantumCircuitSize: parseInt(e.target.value)}}))}/>
                         </div>
                          <div className="space-y-2">
                           <Label htmlFor="labelSmoothing">Label Smoothing</Label>
                           <Input id="labelSmoothing" type="number" step="0.01" value={newConfig.parameters.labelSmoothing} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, labelSmoothing: parseFloat(e.target.value)}}))}/>
                         </div>
                           <div className="space-y-2">
                           <Label htmlFor="modelNameParam">Model Name (Internal)</Label>
                           <Input id="modelNameParam" type="text" value={newConfig.parameters.modelName} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, modelName: e.target.value}}))}/>
                         </div>
                       </div>
                    </TabsContent>
                    <TabsContent value="zpe" className="space-y-3 pt-3">
                      {(["momentumParams", "strengthParams", "noiseParams", "couplingParams"] as (keyof TrainingParameters)[]).map((param) => (
                        param !== "couplingParams" && param !== "momentumParams" && param !== "strengthParams" && param !== "noiseParams" ? null : // type guard
                        <div key={param as string} className="space-y-3">
                          <h4 className="text-sm font-medium capitalize">{param.replace('Params', '')} Parameters</h4>
                          {(newConfig.parameters[param] as number[]).map((value, idx) => (
                            <div key={idx} className="space-y-1">
                              <div className="flex justify-between text-xs"><Label>Layer {idx + 1}</Label><span className="font-mono">{value.toFixed(2)}</span></div>
                              <Slider min={0} max={1} step={0.01} value={[value]} onValueChange={(newValue) => updateNewConfigParam(param, idx, newValue[0])}/>
                            </div>
                          ))}
                        </div>
                      ))}
                    </TabsContent>
                    <TabsContent value="architecture" className="space-y-3 pt-3">
                        <Label>Channel Sizes (e.g., 64,128,256,512)</Label>
                        {(newConfig.channel_sizes as number[]).map((value, idx) => (
                          <div key={idx} className="flex items-center gap-2">
                            <Label className="w-16 text-xs">Layer {idx + 1}:</Label>
                            <Input type="number" value={value} className="h-8" onChange={(e) => {
                                const updatedChannels = [...newConfig.channel_sizes];
                                updatedChannels[idx] = parseInt(e.target.value) || 0; // Ensure it's a number
                                setNewConfig(prev => ({...prev, channel_sizes: updatedChannels}));
                              }} />
                          </div>
                        ))}
                         {newConfig.channel_sizes.length < 6 && (
                           <Button variant="outline" size="sm" onClick={() => setNewConfig(prev => ({...prev, channel_sizes: [...prev.channel_sizes, 0]}))}>Add Channel Layer</Button>
                         )}
                         {newConfig.channel_sizes.length > 1 && (
                            <Button variant="destructive" size="sm" onClick={() => setNewConfig(prev => ({...prev, channel_sizes: prev.channel_sizes.slice(0, -1)}))}>Remove Last Channel Layer</Button>
                         )}
                    </TabsContent>
                  </Tabs>
                </CardContent>
                <CardFooter className="border-t pt-6"><Button onClick={handleCreateConfig} className="w-full" disabled={isSubmitting}>{isSubmitting ? "Creating..." : <><Plus className="h-4 w-4 mr-2" />Create</>}</Button></CardFooter>
              </Card>
            )}
          </div>

          <div className="w-full lg:w-2/3 space-y-6">
            {selectedConfig ? (
              <>
                <Card>
                  <CardHeader className="flex flex-row justify-between">
                    <div><CardTitle className="flex items-center gap-2"><Settings className="h-5 w-5 text-primary" />{selectedConfig.name}</CardTitle><CardDescription>Created on {format(new Date(selectedConfig.date_created), "MMMM d, yyyy")}</CardDescription></div>
                    <div className="flex gap-2"><Button variant="outline" size="sm" onClick={() => handleCloneConfig(selectedConfig)}><Copy className="h-4 w-4 mr-2" />Clone</Button><Button variant="destructive" size="sm" onClick={() => setShowDeletePrompt(true)}><Trash className="h-4 w-4 mr-2" />Delete</Button></div>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="bg-muted rounded-lg p-4 flex flex-col items-center justify-center"><Badge className="mb-2 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">Accuracy</Badge><div className="text-3xl font-bold">{selectedConfig.accuracy}%</div></div>
                      <div className="bg-muted rounded-lg p-4 flex flex-col items-center justify-center"><Badge className="mb-2 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">Quantum</Badge><div className="text-3xl font-bold">{selectedConfig.parameters.quantumMode ? <CheckCircle className="h-6 w-6 text-green-500" /> : <XCircle className="h-6 w-6 text-red-500" />}</div></div>
                      <div className="bg-muted rounded-lg p-4 flex flex-col items-center justify-center"><Badge className="mb-2 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">Channels</Badge><div className="text-lg font-mono">{selectedConfig.channel_sizes?.join('-') || "N/A"}</div></div>
                    </div>
                    <div className="border rounded-lg overflow-hidden">
                      <div className="p-4 border-b bg-muted/50 flex items-center justify-between"><div className="font-semibold">Parameter Summary</div><Badge variant="outline">{selectedConfig.parameters.momentumParams?.length || 0} ZPE layers</Badge></div>
                      <Table><TableHeader><TableRow><TableHead>Parameter</TableHead><TableHead>Values</TableHead><TableHead>Range</TableHead></TableRow></TableHeader>
                        <TableBody>
                          {[
                            { key: 'momentumParams', name: 'ZPE Momentum' },
                            { key: 'strengthParams', name: 'ZPE Strength' },
                            { key: 'noiseParams', name: 'ZPE Noise' },
                            { key: 'couplingParams', name: 'ZPE Coupling' },
                          ].map(paramInfo => {
                             const values = selectedConfig.parameters[paramInfo.key as keyof TrainingParameters] as number[] | undefined;
                             if (!values || !Array.isArray(values)) return null;
                             return (
                               <TableRow key={paramInfo.key}>
                                 <TableCell className="font-medium capitalize">{paramInfo.name}</TableCell>
                                 <TableCell className="font-mono text-xs">{formatArray(values)}</TableCell>
                                 <TableCell className="font-mono">{values.length > 0 ? `${Math.min(...values).toFixed(2)} - ${Math.max(...values).toFixed(2)}` : 'N/A'}</TableCell>
                               </TableRow>
                             );
                          })}
                           <TableRow><TableCell className="font-medium capitalize">Total Epochs</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.totalEpochs}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Batch Size</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.batchSize}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Learning Rate</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.learningRate.toFixed(4)}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Weight Decay</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.weightDecay.toFixed(4)}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Quantum Circuit Size</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.quantumCircuitSize}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Label Smoothing</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.labelSmoothing.toFixed(2)}</TableCell><TableCell></TableCell></TableRow>
                           <TableRow><TableCell className="font-medium capitalize">Model Name (Internal)</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.modelName}</TableCell><TableCell></TableCell></TableRow>
                           {selectedConfig.parameters.baseConfigId && <TableRow><TableCell className="font-medium capitalize">Base Config ID</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.baseConfigId}</TableCell><TableCell></TableCell></TableRow>}
                        </TableBody>
                      </Table>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Parameter Visualization</CardTitle><CardDescription>Visual representation of ZPE parameters across layers</CardDescription></CardHeader>
                  <CardContent className="pt-0">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div><h3 className="text-sm font-medium mb-4 text-muted-foreground">Values by Layer</h3><div className="space-y-6">
                        {(["momentumParams", "strengthParams", "noiseParams", "couplingParams"] as (keyof TrainingParameters)[]).map((paramKey) => {
                             if (paramKey === "couplingParams" && !selectedConfig.parameters.couplingParams) return null; // Skip if couplingParams is not defined
                             const paramName = paramKey.replace('Params', '');
                             const values = selectedConfig.parameters[paramKey] as number[] | undefined;
                             if (!values || !Array.isArray(values)) return null;

                             return (<Collapsible key={paramKey as string}>
                               <CollapsibleTrigger asChild><div className="flex items-center justify-between p-2 hover:bg-muted rounded cursor-pointer">
                                   <div className="flex items-center gap-2">
                                     {paramName === "momentum" && <SlidersHorizontal className="h-4 w-4 text-blue-500" />}
                                     {paramName === "strength" && <Zap className="h-4 w-4 text-purple-500" />}
                                     {paramName === "noise" && <Atom className="h-4 w-4 text-orange-500" />}
                                     {paramName === "coupling" && <Settings className="h-4 w-4 text-green-500" />}
                                     <span className="font-medium capitalize">ZPE {paramName}</span></div><ChevronDown className="h-4 w-4 text-muted-foreground" /></div></CollapsibleTrigger>
                                 <CollapsibleContent><div className="pl-8 pr-2 pb-2 space-y-2">
                                     {values.map((value, idx) => (<div key={idx} className="flex items-center gap-2">
                                         <div className="w-24 text-xs">Layer {idx + 1}</div><div className="flex-1 bg-muted h-2 rounded-full overflow-hidden">
                                           <div className={`h-full rounded-full ${paramName === "momentum" ? "bg-blue-500" : paramName === "strength" ? "bg-purple-500" : paramName === "noise" ? "bg-orange-500" : paramName === "coupling" ? "bg-green-500" : "bg-gray-500"}`} style={{ width: `${value * 100}%` }}></div></div>
                                         <div className="w-12 text-right font-mono text-xs">{value.toFixed(2)}</div></div>))}</div></CollapsibleContent></Collapsible>
                            );
                         })}
                        </div></div>
                      <div><h3 className="text-sm font-medium mb-4 text-muted-foreground">Relationships</h3><div className="aspect-square bg-muted/50 rounded-lg flex items-center justify-center p-4">
                          <svg width="100%" height="100%" viewBox="0 0 400 400">
                            <circle cx="200" cy="200" r="150" fill="none" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="5,5" />
                            <circle cx="200" cy="200" r="100" fill="none" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="5,5" />
                            <circle cx="200" cy="200" r="50" fill="none" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="5,5" />
                            <line x1="50" y1="200" x2="350" y2="200" stroke="hsl(var(--border))" strokeWidth="1" /><line x1="200" y1="50" x2="200" y2="350" stroke="hsl(var(--border))" strokeWidth="1" />
                            <text x="355" y="205" fill="hsl(var(--muted-foreground))" fontSize="10" textAnchor="start">Strength</text><text x="200" y="45" fill="hsl(var(--muted-foreground))" fontSize="10" textAnchor="middle">Momentum</text>
                            <text x="45" y="205" fill="hsl(var(--muted-foreground))" fontSize="10" textAnchor="end">Noise</text><text x="200" y="360" fill="hsl(var(--muted-foreground))" fontSize="10" textAnchor="middle">Coupling</text>
                            {(selectedConfig.parameters.momentumParams)?.map((momentum, idx) => {
                              const strength = (selectedConfig.parameters.strengthParams)?.[idx] ?? 0;
                              const noise = (selectedConfig.parameters.noiseParams)?.[idx] ?? 0;
                              const coupling = (selectedConfig.parameters.couplingParams)?.[idx] ?? 0.5; {/* Use actual coupling or default */}

                              const x = 200 + (strength - noise) * 140; {/* Scaled to 140 */}
                              const y = 200 + (coupling - momentum) * 140; {/* Scaled to 140 */}
                              return (<g key={idx}><circle cx={x} cy={y} r={6} fill={idx === 3 && selectedConfig.parameters.quantumMode ? "hsl(var(--chart-2))" : "hsl(var(--chart-1))"} fillOpacity="0.7"/><text x={x} y={y - 8} fill="hsl(var(--foreground))" fontSize="10" textAnchor="middle">L{idx+1}</text></g>);
                            })}
                          </svg></div></div></div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card>
                <CardHeader><CardTitle>No Configuration Selected</CardTitle></CardHeader>
                <CardContent><p className="text-muted-foreground">Select a configuration from the list to view its details or create a new one.</p></CardContent>
              </Card>
            )}
             {/* showDeletePrompt Dialog */}
          </div>
        </div>
      </div>
    </div>
  );
}

    