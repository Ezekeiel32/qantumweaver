import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Play, StopCircle, Settings } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from '@/components/ui/scroll-area';

// Default parameters aligned with the new Pydantic model in app.py
const defaultParams = {
    model_name: "ZPE-DeepNet-Custom",
    total_epochs: 30,
    batch_size: 32,
    learning_rate: 0.001,
    sequence_length: 10,
    label_smoothing: 0.1,
    mixup_alpha: 1.0,
    zpe_regularization_strength: 0.001,
};

interface ControlsPanelProps {
    startJob: (params: any) => void;
    stopJob: () => void;
    isJobActive: boolean;
    jobId: string | null;
    jobStatus: string | null;
}

const ControlsPanel: React.FC<ControlsPanelProps> = ({ startJob, stopJob, isJobActive, jobId, jobStatus }) => {
    const [showConfig, setShowConfig] = useState(false);
    const [pendingParams, setPendingParams] = useState<any>(defaultParams);

    const handleStart = () => {
        startJob(pendingParams);
        setShowConfig(false);
    };
    
    const renderParamInput = (key: string, label: string, type: string = "number", step: string = "0.001") => (
         <div key={key}>
            <Label className="text-sm font-medium">{label}</Label>
            <Input
                type={type}
                step={step}
                value={pendingParams[key]}
                onChange={e => {
                    const value = type === 'number' ? parseFloat(e.target.value) : e.target.value;
                    setPendingParams((p: any) => ({ ...p, [key]: value || 0 }))
                }}
                className="bg-gray-800 border-gray-600"
            />
        </div>
    );

    return (
        <div className="panel-3d-flat flex flex-col h-full">
            <h2 className="panel-title">Controls</h2>
            <div className="mt-4 space-y-4">
                <Button onClick={() => setShowConfig(true)} className="w-full btn-neon">
                    <Settings className="mr-2 h-4 w-4" /> Configure & Start
                </Button>
                <Button onClick={stopJob} disabled={!isJobActive} className="w-full" variant="destructive">
                    <StopCircle className="mr-2 h-4 w-4" /> Stop Training
                </Button>
                
                {isJobActive && (
                    <div className="text-center p-2 rounded-lg bg-black/30">
                        <p className="text-sm font-semibold text-cyan-300">Status: <span className="font-bold text-yellow-400">{jobStatus}</span></p>
                        <p className="text-xs text-gray-400 truncate mt-1">Job ID: {jobId}</p>
                    </div>
                )}
            </div>

            {/* Configuration Dialog */}
            <Dialog open={showConfig} onOpenChange={setShowConfig}>
                <DialogContent className="panel-3d-flat min-w-[600px] max-w-[90vw] max-h-[90vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle className="panel-title text-2xl">Train Configuration</DialogTitle>
                        <DialogDescription>Adjust parameters for the ZPEDeepNet model.</DialogDescription>
                    </DialogHeader>
                    
                    <ScrollArea className="flex-grow minimal-scrollbar pr-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 py-4">
                            {renderParamInput('model_name', 'Model Name', 'text')}
                            {renderParamInput('total_epochs', 'Total Epochs', 'number', '1')}
                            {renderParamInput('batch_size', 'Batch Size', 'number', '1')}
                            {renderParamInput('learning_rate', 'Learning Rate')}
                            {renderParamInput('sequence_length', 'ZPE Sequence Length', 'number', '1')}
                            {renderParamInput('label_smoothing', 'Label Smoothing')}
                            {renderParamInput('mixup_alpha', 'Mixup Alpha')}
                            {renderParamInput('zpe_regularization_strength', 'ZPE Regularization')}
                        </div>
                    </ScrollArea>

                    <div className="flex justify-end space-x-4 pt-4 border-t border-white/10">
                        <Button variant="ghost" onClick={() => setShowConfig(false)}>Cancel</Button>
                        <Button onClick={handleStart} className="btn-neon">
                            <Play className="mr-2 h-4 w-4" /> Start Training
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ControlsPanel; 