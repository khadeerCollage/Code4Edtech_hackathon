import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button.jsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { ArrowLeft, Play } from "lucide-react";
import WorkflowAnimation from "@/components/WorkflowAnimation.jsx";
import workflowHero from "@/assets/workflow-hero.jpg";

const WorkflowDemo = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted">
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          <Button variant="outline" onClick={() => navigate("/dashboard")} className="mb-4 hover:shadow-md transition-all duration-200">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pb-12">
        <Card className="mb-12 overflow-hidden shadow-lg animate-fade-in">
          <div className="relative">
            <img src={workflowHero} alt="Workflow Hero" className="w-full h-64 object-cover" />
            <div className="absolute inset-0 bg-gradient-to-r from-primary/80 to-primary-glow/60 flex items-center justify-center">
              <div className="text-center text-white">
                <h1 className="text-4xl font-bold mb-4">OMR Workflow Animation</h1>
                <p className="text-xl mb-6 opacity-90">Watch the complete evaluation process from login to export</p>
                <Button variant="gradient" size="lg" className="shadow-glow">
                  <Play className="w-5 h-5 mr-2" />
                  View Live Demo
                </Button>
              </div>
            </div>
          </div>
        </Card>

        <Card className="shadow-lg animate-slide-up">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl bg-gradient-primary bg-clip-text text-transparent">Interactive Workflow Demonstration</CardTitle>
            <p className="text-muted-foreground">This animation shows the complete OMR evaluation workflow with smooth transitions between each stage</p>
          </CardHeader>
          <CardContent>
            <WorkflowAnimation />
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <Card className="animate-scale-in shadow-md" style={{ animationDelay: "0.1s" }}>
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Play className="w-8 h-8 text-primary" />
              </div>
              <h3 className="font-semibold text-lg mb-2">Smooth Animations</h3>
              <p className="text-muted-foreground text-sm">Beautiful transitions between workflow stages with CSS animations and GSAP effects</p>
            </CardContent>
          </Card>

          <Card className="animate-scale-in shadow-md" style={{ animationDelay: "0.2s" }}>
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-success/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <div className="w-8 h-8 bg-success rounded text-success-foreground flex items-center justify-center text-sm font-bold">UI</div>
              </div>
              <h3 className="font-semibold text-lg mb-2">Modern Interface</h3>
              <p className="text-muted-foreground text-sm">Clean, professional UI design with intuitive navigation and responsive layouts</p>
            </CardContent>
          </Card>

          <Card className="animate-scale-in shadow-md" style={{ animationDelay: "0.3s" }}>
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-processing/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <div className="w-8 h-8 border-4 border-processing border-t-transparent rounded-full animate-spin" />
              </div>
              <h3 className="font-semibold text-lg mb-2">Real-time Updates</h3>
              <p className="text-muted-foreground text-sm">Live status updates and progress indicators throughout the evaluation process</p>
            </CardContent>
          </Card>
        </div>

        <div className="text-center mt-12 space-x-4">
          <Button variant="gradient" size="lg" onClick={() => navigate("/login")}>
            Try Live Platform
          </Button>
          <Button variant="outline" size="lg" onClick={() => navigate("/upload")}>
            Start Upload Process
          </Button>
        </div>
      </div>
    </div>
  );
};

export default WorkflowDemo;
