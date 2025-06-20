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

    const ParamInput = ({ id, label, type = "number", step = "0.001" }: { id: keyof typeof defaultParams; label: string; type?: string; step?: string }) => (
        <div>
            <Label htmlFor={id} className="text-sm font-medium">{label}</Label>
            <Input
                id={id}
                type={type}
                step={step}
                value={pendingParams[id]}
                onChange={e => {
                    const { value } = e.target;
                    setPendingParams((p: any) => ({
                        ...p,
                        [id]: type === 'number' ? (value === '' ? '' : parseFloat(value)) : value
                    }));
                }}
                className="bg-gray-800 border-gray-600 mt-1"
            />
        </div>
    );

    return (
        <>
            <div className="flex justify-between items-center mb-4 px-4 pt-4">
                <h2 className="panel-title">Training Monitor</h2>
                 <div className="flex items-center gap-4">
                    <Button onClick={() => setShowConfig(true)} className="btn-neon">
                        <Settings className="mr-2 h-4 w-4" /> Configure & Start
                    </Button>
                    <Button onClick={stopJob} disabled={!isJobActive} className="w-full" variant="destructive">
                        <StopCircle className="mr-2 h-4 w-4" /> Stop Training
                    </Button>
                     {isJobActive && (
                        <div className="text-center">
                            <p className="text-sm font-semibold text-cyan-300">Status: <span className="font-bold text-yellow-400">{jobStatus}</span></p>
                        </div>
                    )}
                </div>
            </div>

            <Dialog open={showConfig} onOpenChange={setShowConfig}>
                <DialogContent className="panel-3d-flat min-w-[600px] max-w-[90vw] max-h-[90vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle className="panel-title text-2xl">Train Configuration</DialogTitle>
                        <DialogDescription>Adjust parameters for the ZPEDeepNet model.</DialogDescription>
                    </DialogHeader>
                    
                    <ScrollArea className="flex-grow minimal-scrollbar pr-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 py-4">
                            <ParamInput id="model_name" label="Model Name" type="text" />
                            <ParamInput id="total_epochs" label="Total Epochs" type="number" step="1" />
                            <ParamInput id="batch_size" label="Batch Size" type="number" step="1" />
                            <ParamInput id="learning_rate" label="Learning Rate" />
                            <ParamInput id="sequence_length" label="ZPE Sequence Length" type="number" step="1" />
                            <ParamInput id="label_smoothing" label="Label Smoothing" />
                            <ParamInput id="mixup_alpha" label="Mixup Alpha" />
                            <ParamInput id="zpe_regularization_strength" label="ZPE Regularization" />
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
        </>
    );
};

export default ControlsPanel; 