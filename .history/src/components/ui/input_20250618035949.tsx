import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-11 w-full rounded-lg border border-white/20 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/50 backdrop-blur-sm transition-all duration-300",
          "focus:border-cyan-400 focus:bg-white/10 focus:outline-none focus:ring-2 focus:ring-cyan-400/20 focus:shadow-lg focus:shadow-cyan-400/20",
          "hover:border-white/30 hover:bg-white/8",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
