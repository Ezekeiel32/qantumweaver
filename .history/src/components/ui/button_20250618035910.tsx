import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "neon-btn inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 relative overflow-hidden",
  {
    variants: {
      variant: {
        default: "bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white shadow-lg hover:shadow-xl hover:scale-105 border border-white/20",
        magenta: "bg-gradient-to-r from-pink-600 to-purple-600 text-white shadow-lg hover:shadow-xl hover:scale-105 border border-white/20",
        blue: "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg hover:shadow-xl hover:scale-105 border border-white/20",
        green: "bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-lg hover:shadow-xl hover:scale-105 border border-white/20",
        outline: "bg-transparent border-2 border-white/20 text-white hover:bg-white/10 hover:border-white/40 backdrop-blur-sm",
        ghost: "bg-transparent text-white hover:bg-white/10 backdrop-blur-sm",
        link: "bg-transparent text-white underline underline-offset-4 hover:text-cyan-400",
        destructive: "bg-gradient-to-r from-red-600 to-pink-600 text-white shadow-lg hover:shadow-xl hover:scale-105 border border-white/20",
      },
      size: {
        default: "h-11 px-6 py-3",
        sm: "h-9 rounded-md px-4 py-2",
        lg: "h-12 rounded-lg px-8 py-4 text-base",
        icon: "h-11 w-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
