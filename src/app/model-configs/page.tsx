
"use client";
import React, { useState, useEffect, useCallback } from "react";
import type { TrainingParameters } from "@/types/training"; // Use shared type
import {
  Settings, Plus, Trash, Copy, Eye, EyeOff, SlidersHorizontal, ChevronDown, CheckCircle, XCircle, Zap, Atom, Architect
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

// Updated FrontendModelConfig type definition
interface FrontendModelConfig {
  id?: string | null;
  name: string;
  channel_sizes?: number[]; // Optional to align better, but UI provides defaults
  parameters: TrainingParameters; 
  accuracy: number;
  loss: number; // Added loss
  date_created: string;
}


export default function ModelConfigurationsPage() {
  const [configs, setConfigs] = useState<FrontendModelConfig[]>([]); 
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<FrontendModelConfig | null>(null); 
  const [showDeletePrompt, setShowDeletePrompt] = useState(false); // State for delete confirmation
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
      couplingParams: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7],
      quantumCircuitSize: 32,
      labelSmoothing: 0.1,
      quantumMode: true,
      modelName: "ZPE-Sim-V1", 
      baseConfigId: null,
    },
    accuracy: 98.5, 
    loss: 0.0123, // Default loss
    date_created: new Date().toISOString().split('T')[0]
  };
  const [newConfig, setNewConfig] = useState<FrontendModelConfig>(initialNewConfig); 

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Placeholder: Replace with actual API call
      // For now, we'll simulate some data to make the page usable
      const modelConfigsData: FrontendModelConfig[] = [
        { 
          id: "demo-1", 
          name: "AlphaZPE-7b", 
          channel_sizes: [64, 128, 256, 512],
          parameters: { 
            totalEpochs: 50, batchSize: 32, learningRate: 0.0005, weightDecay: 0.0001, 
            momentumParams: [0.9,0.88,0.85,0.8,0.75,0.7], 
            strengthParams: [0.4,0.38,0.35,0.3,0.25,0.2],
            noiseParams: [0.1,0.1,0.1,0.1,0.1,0.1],
            couplingParams: [0.7,0.7,0.7,0.7,0.7,0.7],
            quantumCircuitSize: 32, labelSmoothing: 0.1, quantumMode: true, modelName: "AlphaZPEInternal"
          }, 
          accuracy: 99.12, loss: 0.0089, date_created: "2023-10-15" 
        },
        { 
          id: "demo-2", 
          name: "QuantumLeap-v2", 
          channel_sizes: [32, 64, 128],
          parameters: { 
            totalEpochs: 25, batchSize: 64, learningRate: 0.001, weightDecay: 0.0005, 
            momentumParams: [0.85,0.8,0.75,0.7,0.65,0.6], 
            strengthParams: [0.2,0.2,0.25,0.3,0.3,0.3],
            noiseParams: [0.05,0.05,0.05,0.05,0.05,0.05],
            couplingParams: [0.9,0.85,0.8,0.8,0.8,0.8],
            quantumCircuitSize: 16, labelSmoothing: 0.05, quantumMode: false, modelName: "QuantumLeapInternal"
          }, 
          accuracy: 97.85, loss: 0.0152, date_created: "2023-11-01" 
        },
      ]; 
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
      // Placeholder for API call
      console.log("Simulating create for config:", JSON.stringify(configPayload, null, 2));
      // const response = await fetch('/api/model-configs', { // This API route does not exist yet
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(configPayload),
      // });
      // if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      // const createdConfig: FrontendModelConfig = await response.json();
      const createdConfig: FrontendModelConfig = {...configPayload, id: `demo-new-${Date.now()}`, date_created: new Date().toISOString().split('T')[0]};
      
      setConfigs(prev => [...prev, createdConfig]);
      setSelectedConfig(createdConfig);
      if(createdConfig.id) setShownParameters(prev => ({ ...prev, [createdConfig.id!]: false }));
      setNewConfig(initialNewConfig); 
      setIsCreating(false);
      toast({ title: "Success (Simulated)", description: `Configuration "${createdConfig.name}" created.`});
    } catch (error: any) { 
      console.error("Error creating configuration:", error);
      toast({ title: "Error", description: `Failed to create configuration. ${error.message || ''}`, variant: "destructive"});
    }
    setIsSubmitting(false);
  };

  const handleCloneConfig = async (config: FrontendModelConfig) => { 
    try {
      const clonedPayload = {
        ...config, 
        name: `${config.name}-Clone-${Date.now().toString().slice(-4)}`,
        date_created: new Date().toISOString().split('T')[0],
        id: undefined, 
      };
      // Placeholder for API call
      console.log("Simulating clone for config:", JSON.stringify(clonedPayload, null, 2));
      const createdConfig: FrontendModelConfig = {...clonedPayload, id: `demo-clone-${Date.now()}`};
      setConfigs(prev => [...prev, createdConfig]);
      if(createdConfig.id) setShownParameters(prev => ({ ...prev, [createdConfig.id!]: false }));
      toast({ title: "Success (Simulated)", description: `Configuration "${createdConfig.name}" cloned.`});
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
      toast({ title: "Success (Simulated)", description: `Configuration "${selectedConfig.name}" deleted.`});
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
    <div className="p-4 md:p-6 bg-background text-foreground min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col space-y-2 mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-primary">Model Configurations</h1>
          <p className="text-muted-foreground">Manage and compare different neural network configurations.</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-1/3 space-y-6">
            <Card className="shadow-lg">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Available Configurations</CardTitle>
                <Button variant="outline" size="sm" onClick={() => setIsCreating(!isCreating)}>
                  {isCreating ? "Cancel Create" : <><Plus className="h-4 w-4 mr-2" />Create New</>}
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[calc(100vh-400px)] min-h-[300px]">
                  <Table>
                    <TableHeader className="sticky top-0 bg-card z-10">
                      <TableRow><TableHead>Name</TableHead><TableHead>Accuracy</TableHead><TableHead>Date</TableHead><TableHead className="w-12"></TableHead></TableRow>
                    </TableHeader>
                    <TableBody>
                      {isLoading && <TableRow><TableCell colSpan={4} className="text-center py-6">Loading configurations...</TableCell></TableRow>}
                      {!isLoading && configs.length === 0 && <TableRow><TableCell colSpan={4} className="text-center py-6 text-muted-foreground">No configurations found. Click "Create New" to add one.</TableCell></TableRow>}
                      {configs.map((config) => (
                        <TableRow key={config.id} className={`cursor-pointer hover:bg-muted/50 ${selectedConfig?.id === config.id ? 'bg-primary/10' : ''}`} onClick={() => handleConfigSelect(config)}>
                          <TableCell className="font-medium"><div className="flex items-center gap-2"><Architect className="h-4 w-4 text-muted-foreground" />{config.name}</div></TableCell>
                          <TableCell><Badge variant={config.accuracy > 98 ? "default" : "secondary"} className={config.accuracy > 98 ? "bg-green-500/80 text-primary-foreground" : ""}>{config.accuracy.toFixed(2)}%</Badge></TableCell>
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
              <Card className="shadow-lg">
                <CardHeader><CardTitle>Create New Configuration</CardTitle><CardDescription>Define parameters for a new model configuration.</CardDescription></CardHeader>
                <ScrollArea className="max-h-[calc(100vh-250px)]">
                <CardContent className="space-y-4">
                  <div className="space-y-2"><Label htmlFor="config-name">Configuration Name</Label><Input id="config-name" placeholder="e.g., QuantumZPE_Exp1" value={newConfig.name} onChange={(e) => setNewConfig(prev => ({ ...prev, name: e.target.value }))}/></div>
                  <div className="flex items-center justify-between space-x-2 border p-3 rounded-md"><Label htmlFor="quantum-noise" className="flex flex-col space-y-1"><span>Use Quantum Mode</span><span className="font-normal leading-snug text-muted-foreground text-xs">Enable simulated quantum effects.</span></Label><Switch id="quantum-noise" checked={newConfig.parameters.quantumMode} onCheckedChange={handleQuantumNoiseChange}/></div>
                  <Tabs defaultValue="general" className="mt-6">
                    <TabsList className="grid w-full grid-cols-3 mb-4">
                      <TabsTrigger value="general">General</TabsTrigger>
                      <TabsTrigger value="zpe">ZPE</TabsTrigger>
                      <TabsTrigger value="architecture">Architecture</TabsTrigger>
                    </TabsList>
                    <TabsContent value="general" className="space-y-3 pt-3">
                       <div className="grid grid-cols-2 gap-4">
                         <div className="space-y-1"><Label htmlFor="totalEpochs">Total Epochs</Label><Input id="totalEpochs" type="number" value={newConfig.parameters.totalEpochs} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, totalEpochs: parseInt(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1"><Label htmlFor="batchSize">Batch Size</Label><Input id="batchSize" type="number" value={newConfig.parameters.batchSize} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, batchSize: parseInt(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1"><Label htmlFor="learningRate">Learning Rate</Label><Input id="learningRate" type="number" step="0.00001" value={newConfig.parameters.learningRate} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, learningRate: parseFloat(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1"><Label htmlFor="weightDecay">Weight Decay</Label><Input id="weightDecay" type="number" step="0.00001" value={newConfig.parameters.weightDecay} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, weightDecay: parseFloat(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1"><Label htmlFor="quantumCircuitSize">Quantum Circuit</Label><Input id="quantumCircuitSize" type="number" value={newConfig.parameters.quantumCircuitSize} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, quantumCircuitSize: parseInt(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1"><Label htmlFor="labelSmoothing">Label Smoothing</Label><Input id="labelSmoothing" type="number" step="0.01" value={newConfig.parameters.labelSmoothing} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, labelSmoothing: parseFloat(e.target.value) || 0}}))}/></div>
                         <div className="space-y-1 col-span-2"><Label htmlFor="modelNameParam">Model Name (Internal)</Label><Input id="modelNameParam" type="text" placeholder="e.g., ZPENetV2" value={newConfig.parameters.modelName} onChange={(e) => setNewConfig(prev => ({...prev, parameters: {...prev.parameters, modelName: e.target.value}}))}/></div>
                       </div>
                    </TabsContent>
                    <TabsContent value="zpe" className="space-y-4 pt-3">
                      {(["momentumParams", "strengthParams", "noiseParams", "couplingParams"] as (keyof TrainingParameters)[]).map((param) => {
                         const paramArray = newConfig.parameters[param] as number[] | undefined;
                         if (!paramArray || !Array.isArray(paramArray)) return null;
                         return (
                            <div key={param as string} className="space-y-3">
                              <h4 className="text-sm font-medium capitalize flex items-center gap-1.5">
                                {param === "momentumParams" && <SlidersHorizontal className="h-4 w-4 text-blue-500"/>}
                                {param === "strengthParams" && <Zap className="h-4 w-4 text-purple-500"/>}
                                {param === "noiseParams" && <Atom className="h-4 w-4 text-orange-500"/>}
                                {param === "couplingParams" && <Settings className="h-4 w-4 text-green-500"/>}
                                {param.replace('Params', '')} Parameters
                              </h4>
                              {paramArray.map((value, idx) => (
                                <div key={idx} className="space-y-1">
                                  <div className="flex justify-between text-xs"><Label>Layer {idx + 1}</Label><span className="font-mono">{value.toFixed(2)}</span></div>
                                  <Slider min={0} max={1} step={0.01} defaultValue={[value]} onValueChange={(newValue) => updateNewConfigParam(param, idx, newValue[0])}/>
                                </div>
                              ))}
                            </div>
                         );
                      })}
                    </TabsContent>
                    <TabsContent value="architecture" className="space-y-3 pt-3">
                        <Label>Channel Sizes (e.g., 64,128,256,512)</Label>
                        {(newConfig.channel_sizes || []).map((value, idx) => (
                          <div key={idx} className="flex items-center gap-2">
                            <Label className="w-16 text-xs">Layer {idx + 1}:</Label>
                            <Input type="number" value={value} className="h-8" onChange={(e) => {
                                const updatedChannels = [...(newConfig.channel_sizes || [])];
                                updatedChannels[idx] = parseInt(e.target.value) || 0; 
                                setNewConfig(prev => ({...prev, channel_sizes: updatedChannels}));
                              }} />
                          </div>
                        ))}
                         {(newConfig.channel_sizes || []).length < 6 && (
                           <Button variant="outline" size="sm" onClick={() => setNewConfig(prev => ({...prev, channel_sizes: [...(prev.channel_sizes || []), 0]}))}>Add Channel Layer</Button>
                         )}
                         {(newConfig.channel_sizes || []).length > 1 && (
                            <Button variant="destructive" size="sm" onClick={() => setNewConfig(prev => ({...prev, channel_sizes: (prev.channel_sizes || []).slice(0, -1)}))}>Remove Last</Button>
                         )}
                    </TabsContent>
                  </Tabs>
                </CardContent>
                </ScrollArea>
                <CardFooter className="border-t pt-6"><Button onClick={handleCreateConfig} className="w-full" disabled={isSubmitting}>{isSubmitting ? "Creating..." : <><Plus className="h-4 w-4 mr-2" />Create Config</>}</Button></CardFooter>
              </Card>
            )}
          </div>

          <div className="w-full lg:w-2/3 space-y-6">
            {selectedConfig ? (
              <>
                <Card className="shadow-lg">
                  <CardHeader className="flex flex-row justify-between items-start">
                    <div><CardTitle className="flex items-center gap-2 text-xl"><Settings className="h-5 w-5 text-primary" />{selectedConfig.name}</CardTitle><CardDescription>Created on {format(new Date(selectedConfig.date_created), "MMMM d, yyyy")}</CardDescription></div>
                    <div className="flex gap-2"><Button variant="outline" size="sm" onClick={() => handleCloneConfig(selectedConfig)}><Copy className="h-4 w-4 mr-2" />Clone</Button><Button variant="destructive" size="sm" onClick={() => setShowDeletePrompt(true)}><Trash className="h-4 w-4 mr-2" />Delete</Button></div>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="bg-muted/50 rounded-lg p-4 flex flex-col items-center justify-center ring-1 ring-inset ring-border"><Badge className="mb-2 bg-green-500/20 text-green-400 border-green-500/50">Accuracy</Badge><div className="text-3xl font-bold">{selectedConfig.accuracy.toFixed(2)}%</div></div>
                      <div className="bg-muted/50 rounded-lg p-4 flex flex-col items-center justify-center ring-1 ring-inset ring-border"><Badge className="mb-2 bg-red-500/20 text-red-400 border-red-500/50">Loss</Badge><div className="text-3xl font-bold">{selectedConfig.loss.toFixed(4)}</div></div>
                      <div className="bg-muted/50 rounded-lg p-4 flex flex-col items-center justify-center ring-1 ring-inset ring-border"><Badge className="mb-2 bg-purple-500/20 text-purple-400 border-purple-500/50">Quantum Mode</Badge><div className="text-3xl font-bold">{selectedConfig.parameters.quantumMode ? <CheckCircle className="h-7 w-7 text-green-400" /> : <XCircle className="h-7 w-7 text-red-400" />}</div></div>
                    </div>
                    {selectedConfig.id && shownParameters[selectedConfig.id] && (
                      <div className="border rounded-lg overflow-hidden bg-card">
                        <div className="p-4 border-b bg-muted/30 flex items-center justify-between"><div className="font-semibold text-base">Detailed Parameters</div><Badge variant="outline">{selectedConfig.parameters.momentumParams?.length || 0} ZPE layers</Badge></div>
                        <ScrollArea className="max-h-[400px]">
                        <Table>
                          <TableHeader><TableRow><TableHead>Parameter</TableHead><TableHead>Values / Setting</TableHead></TableRow></TableHeader>
                          <TableBody>
                            <TableRow><TableCell className="font-medium">Internal Model Name</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.modelName}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Total Epochs</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.totalEpochs}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Batch Size</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.batchSize}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Learning Rate</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.learningRate.toPrecision(4)}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Weight Decay</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.weightDecay.toPrecision(4)}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Label Smoothing</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.labelSmoothing.toFixed(2)}</TableCell></TableRow>
                            <TableRow><TableCell className="font-medium">Quantum Circuit Size</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.quantumCircuitSize} qubits</TableCell></TableRow>
                            {selectedConfig.parameters.baseConfigId && <TableRow><TableCell className="font-medium">Base Config ID</TableCell><TableCell className="font-mono text-xs">{selectedConfig.parameters.baseConfigId}</TableCell></TableRow>}
                            <TableRow><TableCell className="font-medium">Channel Sizes</TableCell><TableCell className="font-mono text-xs">{formatArray(selectedConfig.channel_sizes)}</TableCell></TableRow>
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
                                 </TableRow>
                               );
                            })}
                          </TableBody>
                        </Table>
                        </ScrollArea>
                      </div>
                    )}
                  </CardContent>
                </Card>
                <Card className="shadow-lg">
                  <CardHeader><CardTitle className="text-lg">ZPE Parameter Visualization</CardTitle><CardDescription>Visual representation of ZPE parameters across layers.</CardDescription></CardHeader>
                  <CardContent className="pt-2">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <h3 className="text-sm font-medium text-muted-foreground">Values by Layer</h3>
                        {(["momentumParams", "strengthParams", "noiseParams", "couplingParams"] as (keyof TrainingParameters)[]).map((paramKey) => {
                             if (paramKey === "couplingParams" && !selectedConfig.parameters.couplingParams) return null;
                             const paramName = paramKey.replace('Params', '');
                             const values = selectedConfig.parameters[paramKey] as number[] | undefined;
                             if (!values || !Array.isArray(values)) return null;

                             return (<Collapsible key={paramKey as string} defaultOpen={paramKey === "momentumParams" || paramKey === "strengthParams"}>
                               <CollapsibleTrigger asChild><Button variant="ghost" className="w-full justify-between p-2 h-auto hover:bg-muted">
                                   <div className="flex items-center gap-2">
                                     {paramName === "momentum" && <SlidersHorizontal className="h-4 w-4 text-blue-400" />}
                                     {paramName === "strength" && <Zap className="h-4 w-4 text-purple-400" />}
                                     {paramName === "noise" && <Atom className="h-4 w-4 text-orange-400" />}
                                     {paramName === "coupling" && <Settings className="h-4 w-4 text-green-400" />}
                                     <span className="font-medium capitalize">ZPE {paramName}</span></div><ChevronDown className="h-4 w-4 text-muted-foreground transition-transform data-[state=open]:rotate-180" /></Button></CollapsibleTrigger>
                                 <CollapsibleContent><div className="pl-6 pr-2 py-2 space-y-2 border-l-2 ml-3 border-dashed">
                                     {values.map((value, idx) => (<div key={idx} className="flex items-center gap-2">
                                         <div className="w-16 text-xs text-muted-foreground">Layer {idx + 1}</div><div className="flex-1 bg-muted/70 h-2 rounded-full overflow-hidden">
                                           <div className={`h-full rounded-full ${paramName === "momentum" ? "bg-blue-500" : paramName === "strength" ? "bg-purple-500" : paramName === "noise" ? "bg-orange-500" : paramName === "coupling" ? "bg-green-500" : "bg-gray-500"}`} style={{ width: `${value * 100}%` }}></div></div>
                                         <div className="w-10 text-right font-mono text-xs">{value.toFixed(2)}</div></div>))}</div></CollapsibleContent></Collapsible>
                            );
                         })}
                        </div>
                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-muted-foreground">Parameter Relationships</h3>
                        <div className="aspect-square bg-muted/30 rounded-lg flex items-center justify-center p-2 border">
                          <svg width="100%" height="100%" viewBox="0 0 200 200">
                            <circle cx="100" cy="100" r="80" fill="none" stroke="hsl(var(--border))" strokeWidth="0.5" strokeDasharray="2,2" />
                            <circle cx="100" cy="100" r="50" fill="none" stroke="hsl(var(--border))" strokeWidth="0.5" strokeDasharray="2,2" />
                            <line x1="20" y1="100" x2="180" y2="100" stroke="hsl(var(--border))" strokeWidth="0.5" /><line x1="100" y1="20" x2="100" y2="180" stroke="hsl(var(--border))" strokeWidth="0.5" />
                            <text x="182" y="102" fill="hsl(var(--muted-foreground))" fontSize="6" textAnchor="start">Strength</text><text x="100" y="18" fill="hsl(var(--muted-foreground))" fontSize="6" textAnchor="middle">Momentum</text>
                            <text x="18" y="102" fill="hsl(var(--muted-foreground))" fontSize="6" textAnchor="end">Noise</text><text x="100" y="185" fill="hsl(var(--muted-foreground))" fontSize="6" textAnchor="middle">Coupling</text>
                            {(selectedConfig.parameters.momentumParams)?.map((momentum, idx) => {
                              const strength = (selectedConfig.parameters.strengthParams)?.[idx] ?? 0;
                              const noise = (selectedConfig.parameters.noiseParams)?.[idx] ?? 0;
                              const coupling = (selectedConfig.parameters.couplingParams)?.[idx] ?? 0.5; 

                              const x = 100 + (strength - noise) * 70; 
                              const y = 100 + (coupling - momentum) * 70; 
                              const isQuantumLayer = idx === 3 && selectedConfig.parameters.quantumMode;
                              return (<g key={idx}><circle cx={x} cy={y} r="3" fill={isQuantumLayer ? "hsl(var(--accent))" : "hsl(var(--primary))"} fillOpacity="0.8" stroke={isQuantumLayer ? "hsl(var(--accent))" : "hsl(var(--primary))"} strokeWidth="0.5"/><text x={x} y={y - 4} fill="hsl(var(--foreground))" fontSize="5" textAnchor="middle">L{idx+1}</text></g>);
                            })}
                          </svg></div></div></div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="shadow-lg flex items-center justify-center min-h-[400px]">
                <CardContent className="text-center">
                  <Architect className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">Select a configuration from the list to view its details or create a new one.</p></CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
      {showDeletePrompt && selectedConfig && (
         <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <Card className="w-full max-w-md">
                <CardHeader><CardTitle>Confirm Deletion</CardTitle><CardDescription>Are you sure you want to delete configuration "{selectedConfig.name}"? This action cannot be undone.</CardDescription></CardHeader>
                <CardFooter className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowDeletePrompt(false)}>Cancel</Button>
                    <Button variant="destructive" onClick={confirmDeleteConfig}>Delete</Button>
                </CardFooter>
            </Card>
         </div>
      )}
    </div>
  );
}

    