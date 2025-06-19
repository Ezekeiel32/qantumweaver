import React from 'react';
import { ChartPanel, TerminalPanel } from './DashboardPanels';

const InstrumentCard: React.FC = () => {
  return (
    <div className="instrument-card bg-black/90 border-4 border-neon-blue rounded-xl shadow-neon-blue/40 relative flex flex-col w-full max-w-4xl mx-auto my-12" style={{ minHeight: '650px', boxShadow: '0 0 32px 4px #00f0ff55' }}>
      {/* Grid Overlay */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="instrument-grid-overlay w-full h-full" />
      </div>
      <div className="relative z-10 flex-1 flex flex-col h-full">
        <div className="flex-1" style={{ flexBasis: '65%' }}>
          <ChartPanel />
        </div>
        <div className="border-t-2 border-neon-blue/40" style={{ flexBasis: '35%' }}>
          <TerminalPanel />
        </div>
      </div>
    </div>
  );
};

export default InstrumentCard; 