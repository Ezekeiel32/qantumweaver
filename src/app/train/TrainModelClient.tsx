"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { toast } from "@/hooks/use-toast";
import { useRouter, useSearchParams } from "next/navigation";
import { useJobStatusPolling } from '../../hooks/useJobStatusPolling';
import ChartPanel from "@/components/ChartPanel";
import TerminalPanel from "@/components/TerminalPanel";
import ControlsPanel from "@/components/ControlsPanel";
import { Button } from "@/components/ui/button";

export default function TrainModelClient() {
  const job = useJobStatusPolling();
  const searchParams = useSearchParams();
  const [advisorParams, setAdvisorParams] = useState<any>(null);
  const [hasStartedFromAdvisor, setHasStartedFromAdvisor] = useState(false);

  useEffect(() => {
    let paramsStr = searchParams.get('advisorParams') || searchParams.get('advisedParams');
    if (paramsStr) {
      let parsedParams: any = null;
      try {
        if (/^[A-Za-z0-9+/=]+$/.test(paramsStr) && paramsStr.length % 4 === 0) {
          parsedParams = JSON.parse(atob(paramsStr));
        } else {
          parsedParams = JSON.parse(decodeURIComponent(paramsStr));
        }
      } catch (e) {
        const err = e as Error;
        toast({ title: "Error loading advisor parameters", description: err.message, variant: "destructive" });
        return;
      }
      // Note: We don't have defaultZPEParams here anymore. The control panel will handle defaults.
      setAdvisorParams(parsedParams);
      if (!hasStartedFromAdvisor) {
        safeStartJob(parsedParams); // Use the parsed params directly
        setHasStartedFromAdvisor(true);
        toast({ title: "Advisor Parameters Loaded", description: `Started training with advisor-suggested parameters.`, variant: "default" });
      }
    } else {
      setAdvisorParams(null);
    }
  }, [searchParams]);

  const safeStartJob = async (params: any) => {
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      let data = null;
      try {
        data = await res.json();
      } catch {
        toast({ title: "Backend Error", description: "Failed to start training: backend did not return valid JSON.", variant: "destructive" });
        return;
      }
      if (!data || !data.job_id) {
        toast({ title: "Backend Error", description: "No job_id returned from backend.", variant: "destructive" });
        return;
      }
      job.setJobId(data.job_id);
      job.resume();
    } catch (e) {
      const err = e as Error;
      toast({ title: "Network Error", description: err.message, variant: "destructive" });
    }
  };
  
  const handleStopJob = () => {
    if (job.jobId) {
      job.stopJob(job.jobId);
    }
  };
  
  const isJobActive = !!job.jobId && (job.isPolling || job.isLoading);

  return (
    <div className="w-full h-full flex flex-col relative items-center bg-[#0a0f1c] min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="instrument-grid-overlay w-full h-full opacity-20" />
      </div>
      
      {/* Any top-level controls can go here if needed */}

      <main className="w-full h-full max-w-screen-2xl mx-auto z-10 dashboard-grid">
        <div className="chart-panel-container panel-3d-flat">
            <ControlsPanel
              startJob={safeStartJob}
              stopJob={handleStopJob}
              isJobActive={isJobActive}
              jobId={job.jobId}
              jobStatus={job.jobStatus}
            />
            <ChartPanel metrics={{ epochs: job.metrics }} />
        </div>
        <div className="terminal-panel-container">
            <TerminalPanel logs={job.logs} />
        </div>
      </main>
    </div>
  );
} 