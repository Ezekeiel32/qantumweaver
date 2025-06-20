import { useState, useEffect, useCallback } from 'react';
import { toast } from './use-toast';
import type { TrainingJob } from '@/types/training';

export function useJobStatusPolling() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [isPolling, setIsPolling] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const startJob = useCallback(async (params: any) => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      if (!data.job_id) {
        throw new Error('No job ID returned');
      }

      setJobId(data.job_id);
      setIsPolling(true);
      toast({
        title: "Training Started",
        description: `Job ID: ${data.job_id}`,
      });
    } catch (error) {
      console.error('Failed to start job:', error);
      toast({
        title: "Error Starting Training",
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopJob = useCallback(async () => {
    if (!jobId) return;

    try {
      const res = await fetch(`/api/stop/${jobId}`, {
        method: 'PUT',
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      setIsPolling(false);
      toast({
        title: "Training Stopped",
        description: `Job ${jobId} has been stopped`,
      });
    } catch (error) {
      console.error('Failed to stop job:', error);
      toast({
        title: "Error Stopping Training",
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: "destructive",
      });
    }
  }, [jobId]);

  const pollJobStatus = useCallback(async () => {
    if (!jobId || !isPolling) return;

    try {
      const res = await fetch(`/api/status/${jobId}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: TrainingJob = await res.json();
      setJobStatus(data.status);
      setLogs(data.log_messages || []);
      
      // Format metrics from the training job data
      if (data.metrics && Array.isArray(data.metrics)) {
        const formattedMetrics = data.metrics.map(metric => ({
          epoch: metric.epoch,
          loss: metric.train_loss,
          accuracy: metric.train_accuracy,
          val_loss: metric.val_loss,
          val_accuracy: metric.val_accuracy
        }));
        setMetrics(formattedMetrics);
      }

      if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
        setIsPolling(false);
        if (data.status === 'completed') {
          toast({
            title: "Training Completed",
            description: `Final accuracy: ${data.metrics?.[data.metrics.length - 1]?.val_accuracy.toFixed(2)}%`,
          });
        }
      }
    } catch (error) {
      console.error('Failed to poll job status:', error);
      setIsPolling(false);
    }
  }, [jobId, isPolling]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (isPolling && jobId) {
      pollJobStatus();
      intervalId = setInterval(pollJobStatus, 2000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isPolling, jobId, pollJobStatus]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setMetrics([]);
  }, []);

  const exportLogs = useCallback(() => {
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${jobId}_logs.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [jobId, logs]);

  const exportMetrics = useCallback(() => {
    const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${jobId}_metrics.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [jobId, metrics]);

  const freeze = useCallback(() => {
    setIsPolling(false);
  }, []);

  const resume = useCallback(() => {
    if (jobId) {
      setIsPolling(true);
    }
  }, [jobId]);

  return {
    jobId,
    jobStatus,
    logs,
    metrics,
    isPolling,
    isLoading,
    startJob,
    stopJob,
    setJobId,
    clearLogs,
    exportLogs,
    exportMetrics,
    freeze,
    resume,
  };
} 