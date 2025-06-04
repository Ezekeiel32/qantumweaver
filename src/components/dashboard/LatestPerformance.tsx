"use client";
import React from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription 
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function LatestPerformance() {
  const generatePerformanceData = () => {
    const result = [];
    for (let epoch = 1; epoch <= 6; epoch++) {
      result.push({
        epoch,
        train_acc: 95 + (epoch / 6) * 4 + Math.random() * 0.5,
        val_acc: 94 + (epoch / 6) * 4.5 + Math.random() * 0.7,
        train_loss: 0.4 - (epoch / 6) * 0.35 + Math.random() * 0.05,
        val_loss: 0.5 - (epoch / 6) * 0.35 + Math.random() * 0.08,
        zpe_effect: 0.2 + (epoch / 6) * 0.3 + Math.random() * 0.05
      });
    }
    return result;
  };

  const performanceData = generatePerformanceData();
  const lastEpoch = performanceData[performanceData.length - 1];

  return (
    <div className="grid gap-4 md:grid-cols-5">
      <Card className="md:col-span-3 bg-card/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="font-headline">Training Metrics</CardTitle>
          <CardDescription>
            Latest performance across epochs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={performanceData.map(p => ({
                  name: `Epoch ${p.epoch}`,
                  "Train Acc": p.train_acc,
                  "Val Acc": p.val_acc
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border)/0.5)" />
                <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[90, 100]} tickFormatter={(val) => `${val}%`}/>
                <Tooltip 
                  contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "var(--radius)"}} 
                  labelStyle={{color: "hsl(var(--popover-foreground))"}}
                  itemStyle={{color: "hsl(var(--popover-foreground))"}}
                />
                <Legend wrapperStyle={{fontSize: "12px"}}/>
                <Line type="monotone" dataKey="Train Acc" stroke="hsl(var(--primary))" strokeWidth={2} dot={{r:3}} activeDot={{r:5}} />
                <Line type="monotone" dataKey="Val Acc" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={{r:3}} activeDot={{r:5}}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      <Card className="md:col-span-2 bg-card/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="font-headline">Latest Results</CardTitle>
          <CardDescription>Epoch {lastEpoch.epoch} performance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  <TableHead className="text-right">Value</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell className="font-medium">Training Accuracy</TableCell>
                  <TableCell className="text-right">
                    <Badge variant="outline" className="bg-primary/10 border-primary/30 text-primary font-semibold">
                      {lastEpoch.train_acc.toFixed(2)}%
                    </Badge>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium">Validation Accuracy</TableCell>
                  <TableCell className="text-right">
                    <Badge variant="outline" className="bg-green-500/10 border-green-500/30 text-green-400 font-semibold">
                      {lastEpoch.val_acc.toFixed(2)}%
                    </Badge>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium">Training Loss</TableCell>
                  <TableCell className="text-right font-code">
                    {lastEpoch.train_loss.toFixed(4)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium">Validation Loss</TableCell>
                  <TableCell className="text-right font-code">
                    {lastEpoch.val_loss.toFixed(4)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium">ZPE Effect (Avg)</TableCell>
                  <TableCell className="text-right">
                    <Badge variant="outline" className="bg-purple-500/10 border-purple-500/30 text-purple-400 font-semibold">
                      {lastEpoch.zpe_effect.toFixed(3)}
                    </Badge>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
            
            <div className="pt-4">
              <h3 className="text-sm font-medium mb-2 text-muted-foreground">ZPE Effects by Layer (Conceptual)</h3>
              <div className="grid grid-cols-6 gap-2">
                {[0.12, 0.18, 0.24, 0.35, 0.11, 0.09].map((value, i) => (
                  <div key={i} className="text-center">
                    <div className="w-full bg-muted rounded-full h-2 mb-1">
                      <div
                        className="bg-purple-500 h-2 rounded-full"
                        style={{ width: `${(value / 0.35) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-muted-foreground">{i+1}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="pt-2">
              <h3 className="text-sm font-medium mb-2 text-muted-foreground">Test Time Augmentation (Conceptual)</h3>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Validation:</span>
                <span className="font-medium">{lastEpoch.val_acc.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">With TTA:</span>
                <span className="font-medium text-green-400">{(lastEpoch.val_acc + 0.6 + Math.random() * 0.2).toFixed(2)}%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
