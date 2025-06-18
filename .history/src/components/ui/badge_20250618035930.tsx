import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg",
        secondary:
          "border-transparent bg-gradient-to-r from-gray-500 to-gray-600 text-white shadow-lg",
        destructive:
          "border-transparent bg-gradient-to-r from-red-500 to-pink-500 text-white shadow-lg",
        outline: "text-white border-white/20 bg-white/5 backdrop-blur-sm hover:bg-white/10",
        success:
          "border-transparent bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg",
        warning:
          "border-transparent bg-gradient-to-r from-yellow-500 to-orange-500 text-white shadow-lg",
        info:
          "border-transparent bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
