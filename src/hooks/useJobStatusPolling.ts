import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from '@/hooks/use-toast';

export function useJobStatusPolling() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<{ epoch: number; loss: number; accuracy: number; val_loss?: number; val_accuracy?: number; }[]>([]);
  const [isPolling, setIsPolling] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Start a new training job
  const startJob = useCallback(async (params: any) => {
    setIsLoading(true);
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
        setIsLoading(false);
        return;
      }
      if (!data || !data.job_id) {
        toast({ title: "Backend Error", description: "No job_id returned from backend.", variant: "destructive" });
        setIsLoading(false);
        return;
      }
      setJobId(data.job_id);
      setIsPolling(true);
    } catch (e) {
      const err = e as Error;
      toast({ title: "Network Error", description: err.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Stop a running job
  const stopJob = useCallback(async () => {
    if (!jobId) return;
    await fetch(`/api/stop/${jobId}`, { method: 'POST' });
  }, [jobId]);

  // Poll job status
  useEffect(() => {
    if (!jobId || !isPolling) return;
    const poll = async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) return;
        const status = await res.json();
        setJobStatus(status);
        setLogs(status.log_messages || []);
        // Parse metrics from logs or status (customize as needed)
        // Example: status.metrics = [{epoch, loss, accuracy, ...}]
        if (status.metrics) setMetrics(status.metrics);
      } catch {}
    };
    poll();
    pollingRef.current = setInterval(poll, 1500);
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [jobId, isPolling]);

  // Freeze polling
  const freeze = useCallback(() => setIsPolling(false), []);
  // Resume polling
  const resume = useCallback(() => setIsPolling(true), []);
  // Clear logs (UI only)
  const clearLogs = useCallback(() => setLogs([]), []);
  // Export logs/metrics
  const exportLogs = useCallback(() => {
    const blob = new Blob([logs.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job_${jobId}_logs.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [logs, jobId]);
  const exportMetrics = useCallback(() => {
    const csv = [
      'epoch,loss,accuracy,val_loss,val_accuracy',
      ...metrics.map(m => `${m.epoch},${m.loss},${m.accuracy},${m.val_loss ?? ''},${m.val_accuracy ?? ''}`)
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job_${jobId}_metrics.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [metrics, jobId]);

  return {
    jobId,
    jobStatus,
    logs,
    metrics,
    isPolling,
    isLoading,
    startJob,
    stopJob,
    freeze,
    resume,
    clearLogs,
    exportLogs,
    exportMetrics,
    setJobId,
  };
} 