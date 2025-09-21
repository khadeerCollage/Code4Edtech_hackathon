import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 transform-gpu cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 active:scale-95",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 active:scale-95",
        outline:
          "border border-input bg-background hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:text-white hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 hover:border-blue-500 active:scale-95",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:text-white hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 active:scale-95",
        ghost:
          "hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:text-white hover:scale-105 hover:-translate-y-0.5 active:scale-95",
        link: "text-primary underline-offset-4 hover:underline hover:text-blue-600 hover:scale-105 active:scale-95",
        gradient:
          "bg-gradient-primary text-primary-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-700 hover:shadow-glow hover:scale-110 hover:-translate-y-1 shadow-md active:scale-95 hover:brightness-110",
        success:
          "bg-gradient-success text-success-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-700 hover:shadow-glow hover:scale-110 hover:-translate-y-1 shadow-md active:scale-95 hover:brightness-110",
        processing:
          "bg-processing text-processing-foreground hover:bg-gradient-to-r hover:from-blue-500 hover:to-blue-600 hover:text-white hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 active:scale-95",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
});
Button.displayName = "Button";

export { Button, buttonVariants };
