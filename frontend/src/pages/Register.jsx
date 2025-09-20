import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Button } from "@/components/ui/button.jsx";
import { Input } from "@/components/ui/input.jsx";
import { Label } from "@/components/ui/label.jsx";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { useToast } from "@/hooks/use-toast.js";

const Register = () => {
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleRegister = (e) => {
    e.preventDefault();
    setIsLoading(true);
    setTimeout(() => {
      if (form.username && form.email && form.password) {
        toast({ title: "Registration Successful", description: "You can now log in to your account." });
        navigate("/login");
      } else {
        toast({ title: "Registration Failed", description: "Please fill all fields.", variant: "destructive" });
      }
      setIsLoading(false);
    }, 1200);
  };

  return (
    <div className="relative min-h-screen overflow-hidden flex items-center justify-center bg-gradient-to-br from-sky-200 via-indigo-200 to-rose-300 p-6">
      {/* Ambient gradient glows */}
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-pink-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-indigo-400/40 blur-3xl -z-10" />

      {/* Gradient border wrapper for glass card */}
      <div className="w-full max-w-sm p-[1px] rounded-2xl bg-gradient-to-br from-white/80 via-indigo-400/50 to-white/40 shadow-2xl animate-fade-in">
        <Card className="rounded-2xl bg-gradient-to-b from-white/90 to-white/70 backdrop-blur-xl border border-white/60 shadow-[0_10px_40px_rgba(99,102,241,0.15)]">
          <CardHeader className="text-center space-y-2 pb-3">
            <CardTitle className="text-2xl font-bold text-blue-600">Create an account</CardTitle>
            <CardDescription className="text-gray-600">Sign up to start using the platform</CardDescription>
            <div className="flex justify-center gap-2 pt-2">
              <Button asChild variant="outline" className="h-9 px-4">
                <Link to="/login">Login</Link>
              </Button>
              <Button className="h-9 px-4 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white">Register</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <form onSubmit={handleRegister} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-gray-700 font-medium">Username</Label>
                <div className="p-[2px] rounded-md gradient-border-rtl focus-within:animate-gradient-rtl">
                  <Input
                    id="username"
                    type="text"
                    placeholder="Choose a username"
                    value={form.username}
                    onChange={(e) => setForm({ ...form, username: e.target.value })}
                    className="rounded-md bg-white border-transparent shadow-sm focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-700 font-medium">Email</Label>
                <div className="p-[2px] rounded-md gradient-border-rtl focus-within:animate-gradient-rtl">
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={form.email}
                    onChange={(e) => setForm({ ...form, email: e.target.value })}
                    className="rounded-md bg-white border-transparent shadow-sm focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-gray-700 font-medium">Password</Label>
                <div className="p-[2px] rounded-md gradient-border-rtl focus-within:animate-gradient-rtl">
                  <Input
                    id="password"
                    type="password"
                    placeholder="Create a password"
                    value={form.password}
                    onChange={(e) => setForm({ ...form, password: e.target.value })}
                    className="rounded-md bg-white border-transparent shadow-sm focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white font-medium py-3 rounded-md transition-all duration-300 mt-2 hover:shadow-lg hover:shadow-indigo-400/30 focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white active:scale-[.99]"
                disabled={isLoading}
              >
                {isLoading ? "Creating account..." : "Register"}
              </Button>

              <p className="text-center text-sm text-gray-600">
                Already have an account? <Link className="text-blue-600 hover:underline" to="/login">Login</Link>
              </p>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Register;
