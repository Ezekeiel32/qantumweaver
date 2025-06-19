"use client";

import React, { useState, useEffect, useCallback, useRef, Suspense } from "react";
import { useForm, Controller, Control, FieldPath, FieldValues } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import type { TrainingParameters, TrainingJob, TrainingJobSummary } from "@/types/training";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/hooks/use-toast";
import { Play, StopCircle, List, Zap, Settings, RefreshCw, AlertTriangle, ExternalLink, SlidersHorizontal, Atom, Brain, Waves, BrainCircuit, Wand2, Save, Download, ArrowDownCircle, PlayCircle } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { TooltipProvider, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { defaultZPEParams } from "@/lib/constants";
import NeonAnalyzerChart from "@/components/visualizations/NeonAnalyzerChart";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import InstrumentCard from '../../components/InstrumentCard';
import { useJobStatusPolling } from '../../hooks/useJobStatusPolling';

// ...all the rest of your code from TrainModelPage, unchanged...

export default function TrainModelClient() {
  const job = useJobStatusPolling();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [hasStartedFromAdvisor, setHasStartedFromAdvisor] = useState(false);

  useEffect(() => {
    if (hasStartedFromAdvisor) return;
    let paramsStr = searchParams.get('advisorParams') || searchParams.get('advisedParams');
    if (paramsStr) {
      let parsedParams: any = null;
      try {
        // Try base64 decode first (advisedParams), fallback to URI decode (advisorParams)
        if (/^[A-Za-z0-9+/=]+$/.test(paramsStr) && paramsStr.length % 4 === 0) {
          parsedParams = JSON.parse(atob(paramsStr));
        } else {
          parsedParams = JSON.parse(decodeURIComponent(paramsStr));
        }
      } catch (e) {
        toast({ title: "Error loading advisor parameters", description: e.message, variant: "destructive" });
        return;
      }
      // Merge with defaults and ensure required fields
      const mergedParams = {
        ...defaultZPEParams,
        ...parsedParams,
        couplingParams: parsedParams.couplingParams || defaultZPEParams.couplingParams,
        mixupAlpha: parsedParams.mixupAlpha ?? 0.2,
      };
      job.startJob(mergedParams);
      setHasStartedFromAdvisor(true);
      toast({ title: "Advisor Parameters Loaded", description: `Started training with advisor-suggested parameters for ${mergedParams.modelName}.`, variant: "success" });
    }
  }, [searchParams, job, hasStartedFromAdvisor]);

  return (
    <InstrumentCard
      jobId={job.jobId}
      jobStatus={job.jobStatus}
      logs={job.logs}
      metrics={job.metrics}
      isPolling={job.isPolling}
      isLoading={job.isLoading}
      startJob={job.startJob}
      stopJob={job.stopJob}
      freeze={job.freeze}
      resume={job.resume}
      clearLogs={job.clearLogs}
      exportLogs={job.exportLogs}
      exportMetrics={job.exportMetrics}
      setJobId={job.setJobId}
    />
  );
} 