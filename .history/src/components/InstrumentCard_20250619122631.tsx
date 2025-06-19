import React from 'react';
import { ChartPanel, TerminalPanel } from './DashboardPanels';

interface InstrumentCardProps {
  jobId: string | null;
  jobStatus: any;
  logs: string[];
  metrics: { epoch: number; loss: number; accuracy: number; val_loss?: number; val_accuracy?: number }[];
  isPolling: boolean;
  isLoading: boolean;
  startJob: (params: any) => void;
  stopJob: () => void;
  freeze: () => void;
  resume: () => void;
  clearLogs: () => void;
  exportLogs: () => void;
  exportMetrics: () => void;
  setJobId: (id: string | null) => void;
}

const InstrumentCard: React.FC<InstrumentCardProps> = ({
  jobId, jobStatus, logs, metrics, isPolling, isLoading,
  startJob, stopJob, freeze, resume, clearLogs, exportLogs, exportMetrics, setJobId
}) => {
  return (
    <div className="instrument-card bg-black/90 border-4 border-neon-blue rounded-xl shadow-neon-blue/40 w-full h-full flex flex-col" style={{ boxShadow: '0 0 32px 4px #00f0ff55' }}>
      {/* Grid Overlay */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="instrument-grid-overlay w-full h-full" />
      </div>
      <div className="relative z-10 flex-1 flex flex-col h-full">
        <div className="flex-1" style={{ flexBasis: '65%' }}>
          <ChartPanel metrics={metrics} jobStatus={jobStatus} isLoading={isLoading} freeze={freeze} resume={resume} exportMetrics={exportMetrics} />
        </div>
        <div className="border-t-2 border-neon-blue/40" style={{ flexBasis: '35%' }}>
          <TerminalPanel logs={logs} isPolling={isPolling} clearLogs={clearLogs} exportLogs={exportLogs} stopJob={stopJob} />
        </div>
      </div>
    </div>
  );
};

export default InstrumentCard; 