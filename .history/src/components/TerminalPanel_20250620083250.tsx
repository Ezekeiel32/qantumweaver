import React, { useEffect, useRef } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";

interface TerminalPanelProps {
  logs: string[];
}

const TerminalPanel: React.FC<TerminalPanelProps> = ({ logs }) => {
  const scrollViewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollViewport.current) {
      scrollViewport.current.scrollTop = scrollViewport.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="panel-3d-flat flex flex-col h-full">
      <h2 className="panel-title">Terminal</h2>
      <ScrollArea className="flex-grow h-full bg-black/50 rounded-md mt-2">
        <div ref={scrollViewport} className="h-full w-full minimal-scrollbar">
          <pre className="text-xs text-green-400 font-mono p-4">
            {logs.join('\n')}
          </pre>
        </div>
      </ScrollArea>
    </div>
  );
};

export default TerminalPanel; 