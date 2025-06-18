'use client';

import React from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';

// Chart Panel Component
export const ChartPanel: React.FC = () => {
  return (
    <Card className="chart-panel h-full w-full bg-black/90 border-neon-blue/30 shadow-neon-blue/20">
      <div className="chart-header p-4 border-b border-neon-blue/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-neon-green rounded-full animate-pulse"></div>
            <h3 className="text-neon-green font-mono text-sm font-medium">SPECTRUM ANALYZER</h3>
            <Badge variant="outline" className="border-neon-blue/50 text-neon-blue text-xs">
              LIVE
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-neon-red rounded-full"></div>
            <span className="text-neon-blue font-mono text-xs">REC</span>
          </div>
        </div>
      </div>
      
      <div className="chart-content p-4 h-full relative">
        {/* Grid Overlay */}
        <div className="absolute inset-0 grid-overlay opacity-20"></div>
        
        {/* Chart Placeholder */}
        <div className="relative z-10 h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 border-2 border-neon-blue/30 rounded-lg flex items-center justify-center">
              <div className="w-8 h-8 border-t-2 border-neon-blue animate-spin rounded-full"></div>
            </div>
            <p className="text-neon-blue/70 font-mono text-sm">CHART DATA LOADING...</p>
            <p className="text-neon-green/50 font-mono text-xs mt-1">QUANTUM SPECTRUM ANALYSIS</p>
          </div>
        </div>
        
        {/* Chart Controls */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="flex items-center justify-between bg-black/50 p-2 rounded border border-neon-blue/20">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Label className="text-neon-blue font-mono text-xs">SCALE</Label>
                <Select defaultValue="auto">
                  <SelectTrigger className="w-20 h-6 bg-black border-neon-blue/30 text-neon-blue text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-black border-neon-blue/30">
                    <SelectItem value="auto">AUTO</SelectItem>
                    <SelectItem value="manual">MANUAL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center space-x-2">
                <Label className="text-neon-blue font-mono text-xs">RANGE</Label>
                <Input 
                  className="w-16 h-6 bg-black border-neon-blue/30 text-neon-blue text-xs text-center"
                  defaultValue="0-100"
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button size="sm" variant="outline" className="h-6 px-2 border-neon-blue/30 text-neon-blue text-xs">
                FREEZE
              </Button>
              <Button size="sm" variant="outline" className="h-6 px-2 border-neon-blue/30 text-neon-blue text-xs">
                EXPORT
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

// Terminal Panel Component
export const TerminalPanel: React.FC = () => {
  const terminalLines = [
    { id: 1, text: '> INITIALIZING QUANTUM NEURAL NETWORK...', type: 'info' },
    { id: 2, text: '> LOADING TRAINING DATA: 100% COMPLETE', type: 'success' },
    { id: 3, text: '> CONFIGURING HYPERPARAMETERS...', type: 'info' },
    { id: 4, text: '> LEARNING RATE: 0.001 | BATCH SIZE: 32', type: 'data' },
    { id: 5, text: '> EPOCH 1/100 | LOSS: 0.8472 | ACC: 0.7234', type: 'progress' },
    { id: 6, text: '> EPOCH 2/100 | LOSS: 0.7234 | ACC: 0.7845', type: 'progress' },
    { id: 7, text: '> EPOCH 3/100 | LOSS: 0.6543 | ACC: 0.8123', type: 'progress' },
    { id: 8, text: '> EPOCH 4/100 | LOSS: 0.5987 | ACC: 0.8345', type: 'progress' },
    { id: 9, text: '> EPOCH 5/100 | LOSS: 0.5432 | ACC: 0.8567', type: 'progress' },
    { id: 10, text: '> VALIDATION LOSS: 0.5213 | VALIDATION ACC: 0.8634', type: 'validation' },
    { id: 11, text: '> EPOCH 6/100 | LOSS: 0.4987 | ACC: 0.8734', type: 'progress' },
    { id: 12, text: '> EPOCH 7/100 | LOSS: 0.4765 | ACC: 0.8845', type: 'progress' },
    { id: 13, text: '> EPOCH 8/100 | LOSS: 0.4543 | ACC: 0.8934', type: 'progress' },
    { id: 14, text: '> EPOCH 9/100 | LOSS: 0.4321 | ACC: 0.9012', type: 'progress' },
    { id: 15, text: '> EPOCH 10/100 | LOSS: 0.4123 | ACC: 0.9087', type: 'progress' },
    { id: 16, text: '> CHECKPOINT SAVED: epoch_10.pt', type: 'success' },
    { id: 17, text: '> CONTINUING TRAINING...', type: 'info' },
  ];

  const getLineColor = (type: string) => {
    switch (type) {
      case 'info': return 'text-neon-blue';
      case 'success': return 'text-neon-green';
      case 'error': return 'text-neon-red';
      case 'warning': return 'text-yellow-400';
      case 'data': return 'text-cyan-400';
      case 'progress': return 'text-neon-green';
      case 'validation': return 'text-purple-400';
      default: return 'text-neon-green';
    }
  };

  return (
    <Card className="terminal-panel h-full w-full bg-black/95 border-neon-green/30 shadow-neon-green/20">
      <div className="terminal-header p-3 border-b border-neon-green/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse"></div>
            <h3 className="text-neon-green font-mono text-sm font-medium">TRAINING LOG</h3>
            <Badge variant="outline" className="border-neon-green/50 text-neon-green text-xs">
              ACTIVE
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-neon-green/70 font-mono text-xs">LINES: {terminalLines.length}</span>
            <Button size="sm" variant="outline" className="h-5 px-2 border-neon-green/30 text-neon-green text-xs">
              CLEAR
            </Button>
          </div>
        </div>
      </div>
      
      <ScrollArea className="terminal-content h-full">
        <div className="p-3 space-y-1">
          {terminalLines.map((line) => (
            <div key={line.id} className={`font-mono text-xs ${getLineColor(line.type)}`}>
              {line.text}
            </div>
          ))}
        </div>
      </ScrollArea>
      
      <div className="terminal-input p-3 border-t border-neon-green/20">
        <div className="flex items-center space-x-2">
          <span className="text-neon-green font-mono text-xs">></span>
          <Input 
            className="flex-1 h-6 bg-black border-neon-green/30 text-neon-green text-xs font-mono"
            placeholder="Enter command..."
          />
          <Button size="sm" variant="outline" className="h-6 px-2 border-neon-green/30 text-neon-green text-xs">
            SEND
          </Button>
        </div>
      </div>
    </Card>
  );
};

// Controls Panel Component
export const ControlsPanel: React.FC = () => {
  return (
    <Card className="controls-panel h-full w-full bg-black/90 border-neon-purple/30 shadow-neon-purple/20">
      <div className="controls-header p-4 border-b border-neon-purple/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-neon-purple rounded-full"></div>
            <h3 className="text-neon-purple font-mono text-sm font-medium">CONTROL CENTER</h3>
            <Badge variant="outline" className="border-neon-purple/50 text-neon-purple text-xs">
              READY
            </Badge>
          </div>
        </div>
      </div>
      
      <ScrollArea className="controls-content h-full">
        <div className="p-4 space-y-6">
          {/* Training Controls */}
          <div className="control-section">
            <h4 className="text-neon-purple font-mono text-sm font-medium mb-3">TRAINING CONTROLS</h4>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-neon-blue font-mono text-xs">AUTO TRAIN</Label>
                <Switch className="data-[state=checked]:bg-neon-blue" />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-neon-blue font-mono text-xs">EARLY STOPPING</Label>
                <Switch className="data-[state=checked]:bg-neon-blue" />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-neon-blue font-mono text-xs">GRADIENT CLIPPING</Label>
                <Switch className="data-[state=checked]:bg-neon-blue" />
              </div>
            </div>
          </div>

          <Separator className="bg-neon-purple/20" />

          {/* Hyperparameters */}
          <div className="control-section">
            <h4 className="text-neon-purple font-mono text-sm font-medium mb-3">HYPERPARAMETERS</h4>
            <div className="space-y-4">
              <div>
                <Label className="text-neon-blue font-mono text-xs">LEARNING RATE</Label>
                <div className="flex items-center space-x-2 mt-1">
                  <Slider defaultValue={[1]} max={100} step={1} className="flex-1" />
                  <Input 
                    className="w-16 h-6 bg-black border-neon-blue/30 text-neon-blue text-xs text-center"
                    defaultValue="0.001"
                  />
                </div>
              </div>
              <div>
                <Label className="text-neon-blue font-mono text-xs">BATCH SIZE</Label>
                <Select defaultValue="32">
                  <SelectTrigger className="w-full h-8 bg-black border-neon-blue/30 text-neon-blue text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-black border-neon-blue/30">
                    <SelectItem value="16">16</SelectItem>
                    <SelectItem value="32">32</SelectItem>
                    <SelectItem value="64">64</SelectItem>
                    <SelectItem value="128">128</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-neon-blue font-mono text-xs">EPOCHS</Label>
                <Input 
                  className="w-full h-8 bg-black border-neon-blue/30 text-neon-blue text-xs"
                  defaultValue="100"
                />
              </div>
            </div>
          </div>

          <Separator className="bg-neon-purple/20" />

          {/* Model Configuration */}
          <div className="control-section">
            <h4 className="text-neon-purple font-mono text-sm font-medium mb-3">MODEL CONFIG</h4>
            <div className="space-y-4">
              <div>
                <Label className="text-neon-blue font-mono text-xs">MODEL TYPE</Label>
                <Select defaultValue="qnn">
                  <SelectTrigger className="w-full h-8 bg-black border-neon-blue/30 text-neon-blue text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-black border-neon-blue/30">
                    <SelectItem value="qnn">QUANTUM NEURAL NET</SelectItem>
                    <SelectItem value="cnn">CONVOLUTIONAL NET</SelectItem>
                    <SelectItem value="rnn">RECURRENT NET</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-neon-blue font-mono text-xs">OPTIMIZER</Label>
                <Select defaultValue="adam">
                  <SelectTrigger className="w-full h-8 bg-black border-neon-blue/30 text-neon-blue text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-black border-neon-blue/30">
                    <SelectItem value="adam">ADAM</SelectItem>
                    <SelectItem value="sgd">SGD</SelectItem>
                    <SelectItem value="rmsprop">RMSPROP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator className="bg-neon-purple/20" />

          {/* Action Buttons */}
          <div className="control-section">
            <h4 className="text-neon-purple font-mono text-sm font-medium mb-3">ACTIONS</h4>
            <div className="grid grid-cols-2 gap-2">
              <Button variant="outline" className="h-8 border-neon-green/30 text-neon-green text-xs">
                START TRAIN
              </Button>
              <Button variant="outline" className="h-8 border-neon-red/30 text-neon-red text-xs">
                STOP TRAIN
              </Button>
              <Button variant="outline" className="h-8 border-neon-blue/30 text-neon-blue text-xs">
                SAVE MODEL
              </Button>
              <Button variant="outline" className="h-8 border-neon-purple/30 text-neon-purple text-xs">
                LOAD MODEL
              </Button>
            </div>
          </div>
        </div>
      </ScrollArea>
    </Card>
  );
}; 