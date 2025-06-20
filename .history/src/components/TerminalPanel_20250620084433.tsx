import React, { useEffect, useRef } from 'react';

interface TerminalPanelProps {
  logs: string[];
}

const TerminalPanel: React.FC<TerminalPanelProps> = ({ logs }) => {
  const terminalRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
        terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="panel-3d-flat flex flex-col h-full">
      <h2 className="panel-title">Terminal</h2>
      <div className="flex-grow bg-black/50 rounded-md mt-2 p-1 overflow-hidden">
        <pre 
            ref={terminalRef} 
            className="h-full w-full minimal-scrollbar overflow-y-auto text-xs text-green-400 font-mono p-4"
        >
          {logs.join('\n')}
        </pre>
      </div>
    </div>
  );
};

export default TerminalPanel; 