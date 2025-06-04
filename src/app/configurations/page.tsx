
"use client";
import React, { useState, useEffect, useCallback } from "react";
import type { TrainingJob, TrainingJobSummary } from "@/types/training";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { toast } from "@/hooks/use-toast";
import { List, RefreshCw, ExternalLink, Eye, Settings, Info } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatDistanceToNow } from 'date-fns';
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

export default function ConfigurationsPage() {
  const [jobsList, setJobsList] = useState<TrainingJobSummary[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [selectedJobDetails, setSelectedJobDetails] = useState<TrainingJob | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);

  const fetchJobsList = useCallback(async () => {
    setIsLoadingJobs(true);
    try {
      const response = await fetch(`${API_BASE_URL}/jobs?limit=50`); // Fetch more for history
      if (!response.ok) throw new Error("Failed to fetch jobs list");
      const data = await response.json();
      setJobsList((data.jobs || []).sort((a:TrainingJobSummary,b:TrainingJobSummary) => new Date(b.start_time || 0).getTime() - new Date(a.start_time || 0).getTime()));
    } catch (error: any) {
      toast({ title: "Error fetching jobs", description: error.message, variant: "destructive" });
    } finally {
      setIsLoadingJobs(false);
    }
  }, []);

  useEffect(() => {
    fetchJobsList();
  }, [fetchJobsList]);

  const handleViewJobDetails = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (!response.ok) throw new Error("Failed to fetch job details");
      const data: TrainingJob = await response.json();
      setSelectedJobDetails(data);
      setIsDetailModalOpen(true);
    } catch (error: any) {
      toast({ title: "Error fetching job details", description: error.message, variant: "destructive" });
    }
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


  return (
    <div className="container mx-auto p-4 md:p-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-2 md:space-y-0 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary">Training Job History</h1>
          <p className="text-muted-foreground">Review parameters and outcomes of past training sessions.</p>
        </div>
        <Button onClick={fetchJobsList} variant="outline" disabled={isLoadingJobs}>
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoadingJobs ? 'animate-spin' : ''}`} /> Refresh List
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Job Log</CardTitle>
          <CardDescription>List of initiated training jobs. Click "View" for parameters.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoadingJobs && <div className="flex justify-center py-10"><RefreshCw className="h-8 w-8 animate-spin text-primary" /></div>}
          {!isLoadingJobs && jobsList.length === 0 && (
            <div className="text-center py-10 text-muted-foreground">
              <List className="mx-auto h-12 w-12 mb-4" />
              No training jobs found. Navigate to the "Train Model" page to start one.
            </div>
          )}
          {!isLoadingJobs && jobsList.length > 0 && (
            <ScrollArea className="h-[600px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model Name</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>Epochs</TableHead>
                    <TableHead>Started</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobsList.map(job => (
                    <TableRow key={job.job_id}>
                      <TableCell className="font-medium">{job.model_name}</TableCell>
                      <TableCell>
                        <Badge variant={
                          job.status === "running" ? "default" :
                          job.status === "completed" ? "default" :
                          job.status === "failed" || job.status === "stopped" ? "destructive" : "secondary"
                        }
                        className={job.status === "completed" ? "bg-green-500 hover:bg-green-600 text-primary-foreground" : job.status === "running" ? "bg-blue-500 hover:bg-blue-600 text-primary-foreground animate-pulse": ""}
                        >{job.status}</Badge>
                      </TableCell>
                      <TableCell>{job.accuracy > 0 ? `${job.accuracy.toFixed(2)}%` : '-'}</TableCell>
                      <TableCell>{job.current_epoch}/{job.total_epochs}</TableCell>
                      <TableCell>{job.start_time ? formatDistanceToNow(new Date(job.start_time), { addSuffix: true }) : 'N/A'}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="outline" size="sm" onClick={() => handleViewJobDetails(job.job_id)}>
                          <Eye className="mr-1 h-3 w-3"/>View Params
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {selectedJobDetails && (
        <Dialog open={isDetailModalOpen} onOpenChange={setIsDetailModalOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5 text-primary"/> Job Parameters: {selectedJobDetails.parameters.modelName}
              </DialogTitle>
              <DialogDescription>
                Configuration used for Job ID: <span className="font-mono text-xs">{selectedJobDetails.job_id.replace('zpe_job_','')}</span>
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh] p-1 pr-3">
              <div className="space-y-3 text-sm py-4">
                <h3 className="font-semibold text-base mb-2 border-b pb-1">General Settings</h3>
                <p><strong>Total Epochs:</strong> {selectedJobDetails.parameters.totalEpochs}</p>
                <p><strong>Batch Size:</strong> {selectedJobDetails.parameters.batchSize}</p>
                <p><strong>Learning Rate:</strong> {selectedJobDetails.parameters.learningRate}</p>
                <p><strong>Weight Decay:</strong> {selectedJobDetails.parameters.weightDecay}</p>
                <p><strong>Label Smoothing:</strong> {selectedJobDetails.parameters.labelSmoothing}</p>
                
                <h3 className="font-semibold text-base mt-3 mb-2 border-b pb-1">ZPE Settings</h3>
                <ParamArrayDisplay label="Momentum Params" values={selectedJobDetails.parameters.momentumParams} />
                <ParamArrayDisplay label="Strength Params" values={selectedJobDetails.parameters.strengthParams} />
                <ParamArrayDisplay label="Noise Params" values={selectedJobDetails.parameters.noiseParams} />
                <ParamArrayDisplay label="Coupling Params" values={selectedJobDetails.parameters.couplingParams} />
                
                <h3 className="font-semibold text-base mt-3 mb-2 border-b pb-1">Quantum Settings</h3>
                <p><strong>Quantum Mode:</strong> {selectedJobDetails.parameters.quantumMode ? "Enabled" : "Disabled"}</p>
                <p><strong>Quantum Circuit Size:</strong> {selectedJobDetails.parameters.quantumCircuitSize} Qubits</p>

                {selectedJobDetails.parameters.baseConfigId && <p className="mt-2"><strong>Base Config ID:</strong> <span className="font-mono text-xs">{selectedJobDetails.parameters.baseConfigId}</span></p>}
                 <div className="mt-4 pt-4 border-t">
                    <Button asChild variant="outline" size="sm">
                        <Link href={{ pathname: "/train", query: { prefill: selectedJobDetails.job_id } }}>
                            <RefreshCw className="mr-2 h-4 w-4"/> Retrain with these parameters
                        </Link>
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
       <Card className="mt-6 bg-accent/30 border-accent">
        <CardHeader>
            <CardTitle className="flex items-center gap-2"><Info className="h-5 w-5 text-accent-foreground"/>Understanding Job History</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-accent-foreground/80 space-y-2">
            <p>This page lists all training jobs initiated via the "Train Model" page.</p>
            <p>Each job entry provides a snapshot of its status, key performance metrics, and start time.</p>
            <p>Click "View Params" to see the detailed configuration parameters used for that specific training run.</p>
            <p>The parameters displayed are those submitted when the job was started and are used by the simulated backend training process.</p>
        </CardContent>
      </Card>
    </div>
  );
}

    