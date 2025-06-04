"use client";
import { useState } from "react";
import { extractHighGainComponents, ExtractHighGainComponentsInput, ExtractHighGainComponentsOutput } from "@/ai/flows/extract-high-gain-components";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "@/hooks/use-toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal, Share2, ListChecks, Brain } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const defaultModelArchitecture = `
ZPEDeepNet based on ResNet principles:
- Input: 1x28x28 (MNIST)
- Conv1: 64 filters, 3x3, BatchNorm, ReLU, MaxPool
- Conv2: 128 filters, 3x3, BatchNorm, ReLU, MaxPool
- Conv3: 256 filters, 3x3, BatchNorm, ReLU, MaxPool
- Conv4: 512 filters, 3x3, BatchNorm, ReLU, MaxPool (Quantum Noise applied here)
- FC Layers: Flatten -> Linear(512, 2048) -> ReLU -> Dropout -> Linear(2048, 512) -> ReLU -> Dropout -> Linear(512, 10)
- ZPE Flows applied after each major block, tuned by momentum, strength, noise, coupling.
- Skip connections present for each convolutional block.
`;

const defaultPerformanceMetrics = `
- Final Validation Accuracy: 99.45%
- Average Training Loss: 0.015
- ZPE Effects (Layer-wise avg deviation from 1.0):
  - Layer 1 (Conv1_out): 0.18
  - Layer 2 (Conv2_out): 0.05
  - Layer 3 (Conv3_out): 0.03
  - Layer 4 (Conv4_out with Quantum Noise): 0.25 (significant perturbation observed)
  - Layer 5 (FC1_out): 0.08
  - Layer 6 (FC2_out): 0.04
- Convergence: Stable after 25 epochs, minor gains up to 30 epochs.
- GPU Memory Usage: ~3.5GB for batch size 32.
`;

const formSchema = z.object({
  modelArchitectureDescription: z.string().min(50, "Model architecture description is too short."),
  performanceMetrics: z.string().min(50, "Performance metrics description is too short."),
  quantumApplicationTarget: z.string().min(20, "Quantum application target description is too short."),
});

export default function ExtractComponentsPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ExtractHighGainComponentsOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { control, handleSubmit, formState: { errors } } = useForm<ExtractHighGainComponentsInput>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      modelArchitectureDescription: defaultModelArchitecture,
      performanceMetrics: defaultPerformanceMetrics,
      quantumApplicationTarget: "Enhance feature extraction in critical layers using a hybrid quantum-classical approach for improved robustness against adversarial attacks.",
    },
  });

  const onSubmit = async (data: ExtractHighGainComponentsInput) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const output = await extractHighGainComponents(data);
      setResult(output);
      toast({ title: "Extraction Successful", description: "High-gain components identified." });
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred.");
      toast({ title: "Extraction Failed", description: e.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Share2 className="h-6 w-6 text-primary" />Extract High-Gain Components</CardTitle>
          <CardDescription>
            Identify model components with high impact for targeted quantum application or optimization.
            Provide model architecture, performance metrics, and your quantum application target.
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="md:col-span-1">
          <CardHeader><CardTitle>Input Descriptions</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div>
                <Label htmlFor="modelArchitectureDescription">Model Architecture Description</Label>
                <Controller name="modelArchitectureDescription" control={control} render={({ field }) => <Textarea {...field} rows={8} placeholder="Describe model layers, connections, parameters..." className="font-mono text-xs"/>} />
                {errors.modelArchitectureDescription && <p className="text-xs text-destructive">{errors.modelArchitectureDescription.message}</p>}
              </div>
              <div>
                <Label htmlFor="performanceMetrics">Performance Metrics</Label>
                <Controller name="performanceMetrics" control={control} render={({ field }) => <Textarea {...field} rows={8} placeholder="Accuracy, loss, ZPE effects, convergence info..." className="font-mono text-xs"/>} />
                {errors.performanceMetrics && <p className="text-xs text-destructive">{errors.performanceMetrics.message}</p>}
              </div>
              <div>
                <Label htmlFor="quantumApplicationTarget">Quantum Application Target</Label>
                <Controller name="quantumApplicationTarget" control={control} render={({ field }) => <Textarea {...field} rows={4} placeholder="e.g., Improve specific layer robustness, explore quantum kernels..." />} />
                {errors.quantumApplicationTarget && <p className="text-xs text-destructive">{errors.quantumApplicationTarget.message}</p>}
              </div>
              <Button type="submit" disabled={isLoading} className="w-full">
                {isLoading ? "Extracting..." : "Identify Components"}
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader><CardTitle>Extraction Results</CardTitle></CardHeader>
          <CardContent>
            {error && (
              <Alert variant="destructive" className="mb-4">
                <Terminal className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            {result ? (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2 flex items-center gap-2"><ListChecks className="h-5 w-5"/>Identified High-Gain Components:</h3>
                  {result.highGainComponents.length > 0 ? (
                    <ScrollArea className="h-[150px] border rounded-md p-3 bg-muted/30">
                      <ul className="list-disc pl-5 space-y-1">
                        {result.highGainComponents.map((component, index) => (
                          <li key={index} className="text-sm">{component}</li>
                        ))}
                      </ul>
                    </ScrollArea>
                  ) : (
                    <p className="text-sm text-muted-foreground">No specific high-gain components were distinctly identified based on the input.</p>
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2 flex items-center gap-2"><Brain className="h-5 w-5"/>Justification:</h3>
                  <Alert>
                    <Terminal className="h-4 w-4"/>
                    <AlertTitle>AI Analysis</AlertTitle>
                    <AlertDescription>
                      <ScrollArea className="h-[250px]">
                         <p className="whitespace-pre-wrap">{result.justification}</p>
                      </ScrollArea>
                    </AlertDescription>
                  </Alert>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-10">Submit model details and target to identify high-gain components.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
