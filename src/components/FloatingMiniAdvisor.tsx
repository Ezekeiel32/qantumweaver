"use client";
import React, { useState, useRef } from "react";
import { MiniHSQNNAdvisor } from "@/components/mini-hs-qnn-advisor";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Sparkles, Minus, Square, MoveDiagonal } from "lucide-react";

interface FloatingMiniAdvisorProps {
  onApplyParameters: (params: any, previousJobId?: string) => void;
  onSaveConfig: (params: any) => void;
  defaultMinimized?: boolean;
}

export default function FloatingMiniAdvisor({ onApplyParameters, onSaveConfig, defaultMinimized = false }: FloatingMiniAdvisorProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isMinimized, setIsMinimized] = useState(defaultMinimized);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; mouseX: number; mouseY: number } | null>(null);
  const [size, setSize] = useState({ width: 420, height: 520 });
  const resizeStart = useRef<{ width: number; height: number; mouseX: number; mouseY: number } | null>(null);
  const [resizing, setResizing] = useState(false);

  // Calculate initial position: bottom right
  const initialBottom = 32;
  const initialRight = 32;

  // Handlers for drag
  const onMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    dragStart.current = {
      x: position.x,
      y: position.y,
      mouseX: e.clientX,
      mouseY: e.clientY,
    };
    document.body.style.userSelect = "none";
  };
  const onMouseMove = (e: MouseEvent) => {
    if (!dragging || !dragStart.current) return;
    const dx = e.clientX - dragStart.current.mouseX;
    const dy = e.clientY - dragStart.current.mouseY;
    setPosition({ x: dragStart.current.x + dx, y: dragStart.current.y + dy });
  };
  const onMouseUp = () => {
    setDragging(false);
    dragStart.current = null;
    document.body.style.userSelect = "";
  };

  // Attach/detach listeners
  React.useEffect(() => {
    if (dragging) {
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    } else {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    // eslint-disable-next-line
  }, [dragging]);

  // Resize handlers
  const onResizeMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    setResizing(true);
    resizeStart.current = {
      width: size.width,
      height: size.height,
      mouseX: e.clientX,
      mouseY: e.clientY,
    };
    document.body.style.userSelect = "none";
  };
  const onResizeMouseMove = (e: MouseEvent) => {
    if (!resizing || !resizeStart.current) return;
    const dx = e.clientX - resizeStart.current.mouseX;
    const dy = e.clientY - resizeStart.current.mouseY;
    setSize({
      width: Math.max(320, Math.min(700, resizeStart.current.width + dx)),
      height: Math.max(320, Math.min(900, resizeStart.current.height + dy)),
    });
  };
  const onResizeMouseUp = () => {
    setResizing(false);
    resizeStart.current = null;
    document.body.style.userSelect = "";
  };
  React.useEffect(() => {
    if (resizing) {
      window.addEventListener("mousemove", onResizeMouseMove);
      window.addEventListener("mouseup", onResizeMouseUp);
    } else {
      window.removeEventListener("mousemove", onResizeMouseMove);
      window.removeEventListener("mouseup", onResizeMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", onResizeMouseMove);
      window.removeEventListener("mouseup", onResizeMouseUp);
    };
  }, [resizing]);

  // Make window draggable from all margins except interactive elements
  const onWindowMouseDown = (e: React.MouseEvent) => {
    // Only drag if not clicking on a button, input, textarea, select, or the resize handle
    const tag = (e.target as HTMLElement).tagName.toLowerCase();
    if (["button", "input", "textarea", "select", "svg", "path"].includes(tag)) return;
    // Also check for resize handle
    if ((e.target as HTMLElement).classList.contains("resize-handle")) return;
    onMouseDown(e);
  };

  if (isMinimized) {
    return (
      <div
        style={{
          position: "fixed",
          bottom: initialBottom - position.y,
          right: initialRight - position.x,
          zIndex: 1000,
          maxWidth: 420,
        }}
      >
        <Button
          variant="secondary"
          className="rounded-full shadow-lg px-4 py-2 flex items-center gap-2"
          onClick={() => setIsMinimized(false)}
        >
          <Sparkles className="h-5 w-5 text-accent" />
          <span className="font-semibold">HS-QNN Mini Advisor</span>
          <Square className="h-4 w-4 ml-1" />
        </Button>
      </div>
    );
  }

  return (
    <div
      style={{
        position: "fixed",
        bottom: initialBottom - position.y,
        right: initialRight - position.x,
        zIndex: 1000,
        width: size.width,
        height: size.height,
        maxWidth: 700,
        maxHeight: 900,
        minWidth: 320,
        minHeight: 320,
        boxSizing: "border-box",
      }}
      onMouseDown={onWindowMouseDown}
    >
      <div className="rounded-2xl shadow-2xl border border-primary/20 bg-background dark:bg-zinc-900 backdrop-blur-lg overflow-hidden h-full flex flex-col">
        <div className="flex items-center justify-between border-b border-primary/10 bg-transparent select-none">
          <div className="flex-1 flex items-center px-6 py-4 text-lg font-bold text-primary bg-transparent">
            <Sparkles className="h-6 w-6 text-accent" />
            <span className="tracking-tight ml-2">HS-QNN Mini Advisor</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="mx-2"
            aria-label="Minimize Advisor"
            onClick={() => setIsMinimized(true)}
            tabIndex={0}
          >
            <Minus className="h-5 w-5 text-muted-foreground" />
          </Button>
        </div>
        <div className="px-6 pt-4 pb-3 bg-background dark:bg-zinc-900">
          <div className="font-semibold text-base text-primary mb-1">AI-Specific Advice for Your Next Training Run</div>
          <div className="border-t border-border/40 my-2" />
        </div>
        <div className="px-6 pb-6 pt-2 bg-background dark:bg-zinc-900">
          <MiniHSQNNAdvisor
            onApplyParameters={onApplyParameters}
            onSaveConfig={onSaveConfig}
          />
        </div>
        <div
          className="absolute bottom-2 right-2 w-6 h-6 flex items-center justify-center cursor-se-resize resize-handle z-50 bg-background/80 rounded-full hover:bg-accent/30 transition"
          style={{ userSelect: "none" }}
          onMouseDown={onResizeMouseDown}
          title="Resize"
        >
          <MoveDiagonal className="h-4 w-4 text-accent" />
        </div>
      </div>
    </div>
  );
} 