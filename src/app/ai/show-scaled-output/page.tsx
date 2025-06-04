"use client";
import { useState } from "react";
import { showScaledOutput, ShowScaledOutputInput, ShowScaledOutputOutput } from "@/ai/flows/show-scaled-output";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "@/hooks/use-toast";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal, Scaling, SlidersHorizontal } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const formSchema = z.object({
  numQubits: z.coerce.number().int().min(2).max(64),
  zpeStrength: z.coerce.number().min(0.01).max(2.0),
});

export default function ShowScaledOutputPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ShowScaledOutputOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { control, handleSubmit, formState: { errors } } = useForm<ShowScaledOutputInput>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      numQubits: 8,
      zpeStrength: 0.5,
    },
  });

  const onSubmit = async (data: ShowScaledOutputInput) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const output = await showScaledOutput(data);
      setResult(output);
      toast({ title: "Simulation Successful", description: `Generated ${output.scaledOutput.length} scaled output values.` });
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred.");
      toast({ title: "Simulation Failed", description: e.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = result?.scaledOutput.map((value, index) => ({
    name: `Qubit ${index}`,
    value: value,
    fill: `hsl(${200 + index * (160 / (result.scaledOutput.length || 1))}, 70%, 60%)` // Dynamic fill color
  })) || [];

  return (
    <div className="container mx-auto p-4 md:p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Scaling className="h-6 w-6 text-primary" />Pseudo-Quantum Circuit Scaled Output</CardTitle>
          <CardDescription>
            Simulate a pseudo-quantum circuit influenced by Zero-Point Energy (ZPE) strength and visualize its scaled output.
            The output values are normalized to the range [0, 1].
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="md:col-span-1">
          <CardHeader><CardTitle className="flex items-center gap-2"><SlidersHorizontal className="h-5 w-5"/>Input Parameters</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <Label htmlFor="numQubits">Number of Qubits</Label>
                <Controller name="numQubits" control={control} render={({ field }) => <Input {...field} type="number" min="2" max="64" />} />
                {errors.numQubits && <p className="text-xs text-destructive">{errors.numQubits.message}</p>}
              </div>
              <div>
                <Label htmlFor="zpeStrength">ZPE Strength</Label>
                <Controller name="zpeStrength" control={control} render={({ field }) => <Input {...field} type="number" step="0.01" min="0.01" max="2.0" />} />
                {errors.zpeStrength && <p className="text-xs text-destructive">{errors.zpeStrength.message}</p>}
              </div>
              <Button type="submit" disabled={isLoading} className="w-full">
                {isLoading ? "Simulating..." : "Generate Scaled Output"}
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader><CardTitle>Simulation Results</CardTitle></CardHeader>
          <CardContent>
            {error && (
              <Alert variant="destructive" className="mb-4">
                <Terminal className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            {result ? (
              <div className="space-y-4">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border)/0.5)" />
                      <XAxis dataKey="name" fontSize={10} stroke="hsl(var(--muted-foreground))"/>
                      <YAxis domain={[0, 1]} fontSize={10} stroke="hsl(var(--muted-foreground))"/>
                      <Tooltip 
                        contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "var(--radius)"}} 
                        labelStyle={{color: "hsl(var(--popover-foreground))", fontWeight: "bold"}}
                        itemStyle={{color: "hsl(var(--popover-foreground))"}}
                      />
                      <Bar dataKey="value" name="Scaled Output" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h3 className="text-md font-semibold mb-1">Raw Output Values:</h3>
                  <ScrollArea className="h-[100px] border rounded-md p-2 bg-muted/30">
                    <pre className="text-xs font-mono whitespace-pre-wrap">
                      {JSON.stringify(result.scaledOutput.map(v => v.toFixed(4)), null, 2)}
                    </pre>
                  </ScrollArea>
                </div>
                 <Alert>
                  <Scaling className="h-4 w-4" />
                  <AlertTitle>Interpretation</AlertTitle>
                  <AlertDescription>
                    Each bar represents a simulated qubit&apos;s output state, scaled between 0 and 1. The ZPE strength influences the overall distribution and variance of these states.
                  </AlertDescription>
                </Alert>
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-10">Run simulation to see the scaled output.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}