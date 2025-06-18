"use client"

import * as React from "react"
import * as ProgressPrimitive from "@radix-ui/react-progress"

import { cn } from "@/lib/utils"

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, value, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("relative h-2 w-full overflow-hidden rounded-full bg-[var(--bg-panel)] glass", className)}
    {...props}
  >
    <div
      className="absolute left-0 top-0 h-full bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 shadow-inner rounded-full transition-all duration-500"
      style={{ width: `${value || 0}%` }}
    />
  </div>
))
Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }
