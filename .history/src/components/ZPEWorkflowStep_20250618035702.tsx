import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { CheckCircle, Clock, AlertCircle, Loader, ArrowRight, Play, Pause, SkipForward } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export interface WorkflowStep {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  progress: number;
  estimatedTime?: number;
  startTime?: Date;
  endTime?: Date;
  error?: string;
  metadata?: {
    [key: string]: any;
  };
}

interface ZPEWorkflowStepProps {
  step: WorkflowStep;
  isActive?: boolean;
  isLast?: boolean;
  onStepClick?: (stepId: string) => void;
  onStepAction?: (stepId: string, action: 'start' | 'pause' | 'skip') => void;
  showDetails?: boolean;
  className?: string;
}

export default function ZPEWorkflowStep({
  step,
  isActive = false,
  isLast = false,
  onStepClick,
  onStepAction,
  showDetails = false,
  className = ""
}: ZPEWorkflowStepProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'running':
        return <Loader className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'skipped':
        return <SkipForward className="w-5 h-5 text-gray-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'skipped':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      default:
        return 'bg-gray-50 text-gray-600 border-gray-200';
    }
  };

  const getProgressColor = () => {
    switch (step.status) {
      case 'completed':
        return 'bg-green-500';
      case 'running':
        return 'bg-blue-500';
      case 'error':
        return 'bg-red-500';
      case 'skipped':
        return 'bg-gray-400';
      default:
        return 'bg-gray-300';
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getElapsedTime = () => {
    if (!step.startTime) return 0;
    const end = step.endTime || new Date();
    return Math.floor((end.getTime() - step.startTime.getTime()) / 1000);
  };

  const getEstimatedTimeRemaining = () => {
    if (step.status !== 'running' || !step.estimatedTime) return null;
    const elapsed = getElapsedTime();
    const remaining = step.estimatedTime - elapsed;
    return remaining > 0 ? remaining : 0;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={className}
    >
      <Card 
        className={`transition-all duration-200 cursor-pointer hover:shadow-md ${
          isActive ? 'ring-2 ring-blue-500 bg-blue-50/50' : ''
        } ${getStatusColor()}`}
        onClick={() => onStepClick?.(step.id)}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3 flex-1">
              <div className="flex-shrink-0 mt-1">
                {getStatusIcon()}
              </div>
              <div className="flex-1 min-w-0">
                <CardTitle className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                  {step.title}
                  <Badge className={`text-xs ${getStatusColor()}`}>
                    {step.status.charAt(0).toUpperCase() + step.status.slice(1)}
                  </Badge>
                </CardTitle>
                <CardDescription className="text-slate-600 mt-1">
                  {step.description}
                </CardDescription>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex items-center gap-1">
              {step.status === 'pending' && (
                <Button
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onStepAction?.(step.id, 'start');
                  }}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <Play className="w-3 h-3 mr-1" />
                  Start
                </Button>
              )}
              
              {step.status === 'running' && (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={(e) => {
                      e.stopPropagation();
                      onStepAction?.(step.id, 'pause');
                    }}
                  >
                    <Pause className="w-3 h-3" />
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={(e) => {
                      e.stopPropagation();
                      onStepAction?.(step.id, 'skip');
                    }}
                  >
                    <SkipForward className="w-3 h-3" />
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Progress</span>
              <span className="font-medium text-slate-800">{step.progress}%</span>
            </div>
            <Progress 
              value={step.progress} 
              className="h-2"
            />
          </div>

          {/* Time Information */}
          <div className="flex items-center justify-between text-sm text-slate-600">
            <div className="flex items-center gap-4">
              {step.startTime && (
                <span>Started: {step.startTime.toLocaleTimeString()}</span>
              )}
              {step.status === 'running' && step.estimatedTime && (
                <span>ETA: {formatDuration(getEstimatedTimeRemaining() || 0)}</span>
              )}
              {step.endTime && (
                <span>Completed: {step.endTime.toLocaleTimeString()}</span>
              )}
            </div>
            {step.status === 'running' && (
              <span className="text-blue-600 font-medium">
                {formatDuration(getElapsedTime())} elapsed
              </span>
            )}
          </div>

          {/* Error Message */}
          {step.error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="bg-red-50 border border-red-200 rounded-lg p-3"
            >
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-red-800">Error occurred</p>
                  <p className="text-sm text-red-700 mt-1">{step.error}</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Detailed Metadata */}
          <AnimatePresence>
            {showDetails && step.metadata && Object.keys(step.metadata).length > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="bg-slate-50 rounded-lg p-3 space-y-2"
              >
                <h4 className="text-sm font-medium text-slate-700">Step Details</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(step.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-slate-600 capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="font-medium text-slate-800">
                        {typeof value === 'number' ? value.toFixed(2) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* Connection Line */}
      {!isLast && (
        <div className="flex justify-center py-2">
          <motion.div
            initial={{ scaleY: 0 }}
            animate={{ scaleY: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
            className="w-0.5 h-8 bg-gradient-to-b from-slate-300 to-slate-200"
          />
        </div>
      )}
    </motion.div>
  );
}

// Workflow Step List Component
interface ZPEWorkflowStepListProps {
  steps: WorkflowStep[];
  activeStepId?: string;
  onStepClick?: (stepId: string) => void;
  onStepAction?: (stepId: string, action: 'start' | 'pause' | 'skip') => void;
  showDetails?: boolean;
  className?: string;
}

export function ZPEWorkflowStepList({
  steps,
  activeStepId,
  onStepClick,
  onStepAction,
  showDetails = false,
  className = ""
}: ZPEWorkflowStepListProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      {steps.map((step, index) => (
        <ZPEWorkflowStep
          key={step.id}
          step={step}
          isActive={step.id === activeStepId}
          isLast={index === steps.length - 1}
          onStepClick={onStepClick}
          onStepAction={onStepAction}
          showDetails={showDetails}
        />
      ))}
    </div>
  );
}

// Workflow Progress Summary Component
interface ZPEWorkflowProgressProps {
  steps: WorkflowStep[];
  className?: string;
}

export function ZPEWorkflowProgress({ steps, className = "" }: ZPEWorkflowProgressProps) {
  const completedSteps = steps.filter(step => step.status === 'completed').length;
  const totalSteps = steps.length;
  const progress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;
  
  const runningStep = steps.find(step => step.status === 'running');
  const errorSteps = steps.filter(step => step.status === 'error').length;

  return (
    <Card className={`bg-white/80 backdrop-blur-sm border-slate-200/50 ${className}`}>
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-slate-800">
          Workflow Progress
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Overall Progress</span>
            <span className="font-medium text-slate-800">
              {completedSteps}/{totalSteps} steps
            </span>
          </div>
          <Progress value={progress} className="h-3" />
        </div>

        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-green-600">{completedSteps}</div>
            <div className="text-sm text-green-700">Completed</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {steps.filter(step => step.status === 'running').length}
            </div>
            <div className="text-sm text-blue-700">Running</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-600">{errorSteps}</div>
            <div className="text-sm text-red-700">Errors</div>
          </div>
        </div>

        {runningStep && (
          <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
            <div className="flex items-center gap-2">
              <Loader className="w-4 h-4 animate-spin text-blue-600" />
              <span className="text-sm font-medium text-blue-800">
                Currently running: {runningStep.title}
              </span>
            </div>
            <Progress value={runningStep.progress} className="mt-2 h-2" />
          </div>
        )}

        {errorSteps > 0 && (
          <div className="bg-red-50 p-3 rounded-lg border border-red-200">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span className="text-sm font-medium text-red-800">
                {errorSteps} step{errorSteps > 1 ? 's' : ''} failed
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}