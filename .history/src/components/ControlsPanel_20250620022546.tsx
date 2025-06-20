import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Play, StopCircle, Settings } from 'lucide-react';
import { defaultZPEParams } from "@/lib/constants";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface ControlsPanelProps {
    startJob: (params: any) => void;
    stopJob: () => void;
    isJobActive: boolean;
    jobId: string | null;
    jobStatus: string | null;
}

const ControlsPanel: React.FC<ControlsPanelProps> = ({ startJob, stopJob, isJobActive, jobId, jobStatus }) => {
    const [showConfig, setShowConfig] = useState(false);
    const [pendingParams, setPendingParams] = useState<any>(defaultZPEParams);

    const handleStart = () => {
        startJob(pendingParams);
        setShowConfig(false);
    };

    return (
        <div className="panel-3d-flat">
            <h2 className="panel-title">Controls</h2>
            <div className="mt-4 space-y-4">
                <Button onClick={() => setShowConfig(true)} className="w-full btn-neon">
                    <Settings className="mr-2 h-4 w-4" /> Configure & Start
                </Button>
                <Button onClick={stopJob} disabled={!isJobActive} className="w-full" variant="destructive">
                    <StopCircle className="mr-2 h-4 w-4" /> Stop Training
                </Button>
                
                {isJobActive && (
                    <div className="text-center p-2 rounded-lg bg-gray-800">
                        <p className="text-sm font-semibold text-white">Status: <span className="font-bold text-yellow-400">{jobStatus}</span></p>
                        <p className="text-xs text-gray-400 truncate">Job ID: {jobId}</p>
                    </div>
                )}
            </div>

            {/* Configuration Dialog */}
            <Dialog open={showConfig} onOpenChange={setShowConfig}>
                <DialogContent className="neon-card min-w-[900px]">
                    <DialogHeader>
                        <DialogTitle className="text-3xl font-bold">Train Configuration</DialogTitle>
                        <DialogDescription>Configure your ZPE-enhanced neural network training job.</DialogDescription>
                    </DialogHeader>
                    
                    {/* Simplified Config for now - will be replaced with the full form */}
                    <div className="space-y-4 py-4">
                        <div>
                          <Label>Model Name</Label>
                          <Input type="text" value={pendingParams.modelName} onChange={e => setPendingParams((p:any) => ({...p, modelName: e.target.value}))} />
                        </div>
                        <div>
                          <Label>Total Epochs</Label>
                          <Input type="number" value={pendingParams.totalEpochs} onChange={e => setPendingParams((p:any) => ({...p, totalEpochs: parseInt(e.target.value,10)||0}))} />
                        </div>
                        <div>
                          <Label>Learning Rate</Label>
                          <Input type="number" step="0.0001" value={pendingParams.learningRate} onChange={e => setPendingParams((p:any) => ({...p, learningRate: parseFloat(e.target.value)||0}))} />
                        </div>
                    </div>

                    <div className="flex justify-end space-x-4">
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