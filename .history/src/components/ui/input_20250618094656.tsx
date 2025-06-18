import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<
  React.ElementRef<"input">,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "bg-white/70 backdrop-blur border border-blue-200 focus:ring-2 focus:ring-blue-400 rounded-lg transition-all",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = "Input"

export { Input }
