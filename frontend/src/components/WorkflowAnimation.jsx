import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Progress } from "@/components/ui/progress.jsx";
import { LogIn, BarChart3, Upload, Clock, CheckCircle, AlertCircle, Eye, Download } from "lucide-react";

const WorkflowAnimation = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  const steps = [
    { id: "login", title: "Login", description: "Evaluator signs in to the platform", icon: LogIn, status: "pending", duration: 2000 },
    { id: "dashboard", title: "Dashboard", description: "View metrics and recent batches", icon: BarChart3, status: "pending", duration: 3000 },
    { id: "upload", title: "Upload Files", description: "Upload OMR sheets and set exam details", icon: Upload, status: "pending", duration: 4000 },
    { id: "processing", title: "Processing", description: "System processes and evaluates sheets", icon: Clock, status: "pending", duration: 5000 },
    { id: "review", title: "Review Flagged", description: "Review and approve flagged sheets", icon: Eye, status: "pending", duration: 3000 },
    { id: "completed", title: "Results Ready", description: "All sheets processed successfully", icon: CheckCircle, status: "pending", duration: 2000 },
    { id: "export", title: "Export Results", description: "Download results as CSV file", icon: Download, status: "pending", duration: 2000 },
  ];

  const [workflowSteps, setWorkflowSteps] = useState(steps);

  useEffect(() => {
    const timer = setInterval(() => {
      setWorkflowSteps((prev) => {
        const updated = [...prev];
        for (let i = 0; i < currentStep; i++) updated[i].status = "completed";
        if (currentStep < updated.length) updated[currentStep].status = "active";
        return updated;
      });

      setProgress(((currentStep + 1) / steps.length) * 100);

      if (currentStep < steps.length - 1) {
        setTimeout(() => setCurrentStep((prev) => prev + 1), workflowSteps[currentStep]?.duration || 2000);
      } else {
        setTimeout(() => {
          setCurrentStep(0);
          setProgress(0);
          setWorkflowSteps(steps.map((step) => ({ ...step, status: "pending" })));
        }, 3000);
      }
    }, workflowSteps[currentStep]?.duration || 2000);

    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  const getStepColor = (status) => {
    switch (status) {
      case "completed":
        return "bg-success text-success-foreground";
      case "active":
        return "bg-processing text-processing-foreground animate-pulse";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getStepIcon = (status) => {
    switch (status) {
      case "completed":
        return CheckCircle;
      case "active":
        return Clock;
      default:
        return AlertCircle;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-foreground mb-2">OMR Evaluation Workflow</h2>
        <p className="text-muted-foreground">Automated journey from login to results export</p>
        <div className="mt-4">
          <Progress value={progress} className="h-2" />
          <p className="text-sm text-muted-foreground mt-2">
            Step {currentStep + 1} of {steps.length}: {workflowSteps[currentStep]?.title}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {workflowSteps.map((step, index) => {
          const IconComponent = step.icon;
          const StatusIcon = getStepIcon(step.status);

          return (
            <Card
              key={step.id}
              className={`transition-all duration-500 ${
                step.status === "active"
                  ? "shadow-glow scale-105 animate-bounce-in"
                  : step.status === "completed"
                  ? "shadow-md animate-scale-in"
                  : "shadow-sm opacity-60"
              }`}
              style={{ animationDelay: step.status === "active" ? "0s" : `${index * 0.1}s` }}
            >
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    <IconComponent className="w-6 h-6 text-primary" />
                  </div>
                  <Badge className={getStepColor(step.status)}>
                    <StatusIcon className="w-3 h-3 mr-1" />
                    {step.status}
                  </Badge>
                </div>
                <h3 className="font-semibold text-foreground mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.description}</p>

                {step.status === "active" && (
                  <div className="mt-3">
                    <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full animate-progress" />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="mt-8 text-center">
        <p className="text-sm text-muted-foreground">
          This animation demonstrates the complete OMR evaluation workflow with smooth transitions and real-time status updates.
        </p>
      </div>
    </div>
  );
};

export default WorkflowAnimation;
