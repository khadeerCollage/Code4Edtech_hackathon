import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to login page
    navigate("/login");
  }, [navigate]);

  return (
    <div className="relative min-h-screen overflow-hidden flex items-center justify-center bg-gradient-to-br from-red-500 via-red-300 to-sky-400">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />
      <div className="text-center animate-fade-in">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-muted-foreground">Loading OMR Evaluation Platform...</p>
      </div>
    </div>
  );
};

export default Index;
