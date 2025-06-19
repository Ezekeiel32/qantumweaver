import { ChartPanel, TerminalPanel, ControlsPanel } from '../components/DashboardPanels';

export default function Home() {
  return (
    <div className="dashboard-grid min-h-screen bg-dashboard">
      <div className="dashboard-chart-panel">
        <ChartPanel />
      </div>
      <div className="dashboard-terminal-panel">
        <TerminalPanel />
      </div>
      <div className="dashboard-controls-panel">
        <ControlsPanel />
      </div>
    </div>
  );
}
