import React, { useState } from 'react';
import NeonAnalyzerChart from "@/components/visualizations/NeonAnalyzerChart";
import { Button } from '@/components/ui/button';
import { ZoomIn, ZoomOut } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush
} from 'recharts';
import { TrainingMetrics } from '@/types/training';

interface ChartPanelProps {
  metrics: TrainingMetrics;
}

const ChartPanel: React.FC<ChartPanelProps> = ({ metrics }) => {
  const [zoom, setZoom] = useState(1);

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 5));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 1));
  
  const lossData = metrics.epochs.map(e => ({ epoch: e.epoch, loss: e.loss, val_loss: e.val_loss }));
  const accData = metrics.epochs.map(e => ({ epoch: e.epoch, accuracy: e.accuracy, val_acc: e.val_acc }));

  // Adjust data based on zoom
  const zoomedDataLength = Math.ceil(lossData.length / zoom);
  const startIndex = Math.max(0, lossData.length - zoomedDataLength);
  const zoomedLossData = lossData.slice(startIndex);
  const zoomedAccData = accData.slice(startIndex);


  return (
    <div className="panel-3d-flat flex flex-col h-full">
      <div className="flex justify-between items-center">
        <h2 className="panel-title">Training Monitor</h2>
        <div className="flex items-center gap-2">
            <Button onClick={handleZoomOut} variant="outline" size="icon" className="h-8 w-8">
                <ZoomOut className="h-4 w-4" />
            </Button>
            <Button onClick={handleZoomIn} variant="outline" size="icon" className="h-8 w-8">
                <ZoomIn className="h-4 w-4" />
            </Button>
        </div>
      </div>

      <div className="flex-grow w-full h-full mt-2 space-y-4">
        <div className="h-1/2 w-full">
            <ResponsiveContainer>
                <LineChart data={zoomedLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis dataKey="epoch" stroke="#888888" />
                    <YAxis stroke="#888888" />
                    <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #555' }} />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
                    <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Validation Loss" />
                    <Brush dataKey="epoch" height={30} stroke="#8884d8" />
                </LineChart>
            </ResponsiveContainer>
        </div>
        <div className="h-1/2 w-full">
            <ResponsiveContainer>
                <LineChart data={zoomedAccData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis dataKey="epoch" stroke="#888888" />
                    <YAxis stroke="#888888" />
                    <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #555' }} />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#ffc658" name="Training Accuracy" />
                    <Line type="monotone" dataKey="val_acc" stroke="#ff8042" name="Validation Accuracy" />
                    <Brush dataKey="epoch" height={30} stroke="#ffc658" />
                </LineChart>
            </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default ChartPanel; 