import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

const Register = () => {
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleRegister = async (e) => {
    e.preventDefault();
    
    if (!form.username.trim() || !form.email.trim() || !form.password.trim()) {
      setError("All fields are required");
      return;
    }

    if (form.password.length < 6) {
      setError("Password must be at least 6 characters long");
      return;
    }

    console.log('Starting registration process...');
    console.log('Form data:', { username: form.username, email: form.email });

    setIsLoading(true);
    setError("");

    try {
      console.log('Sending registration request to:', 'http://localhost:5000/api/auth/register');
      
      const response = await fetch('http://localhost:5000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: form.username.trim(),
          email: form.email.trim(),
          password: form.password.trim()
        })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Registration failed with status:', response.status, 'Error:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);

      if (data.success) {
        console.log('Registration successful!');
        
        toast({
          title: "Registration successful!",
          description: "Welcome! You can now login to your account.",
        });

        // Clear form
        setForm({ username: "", email: "", password: "" });
        
        // Redirect to login page
        setTimeout(() => {
          navigate('/login');
        }, 1500);
      } else {
        console.error('Registration failed:', data.message);
        setError(data.message || "Registration failed. Please try again.");
      }
    } catch (error) {
      console.error('Registration error:', error);
      
      if (error.message.includes('Failed to fetch')) {
        setError("Cannot connect to server. Please ensure the backend is running.");
      } else if (error.message.includes('HTTP error')) {
        setError("Server error. Please try again later.");
      } else {
        setError(error.message || "Registration failed. Please try again.");
      }

      toast({
        variant: "destructive",
        title: "Registration failed",
        description: error.message || "Please check your connection and try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden flex items-center justify-center bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-6">
      {/* Ambient gradient glows */}
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />

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
            {error && (
              <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded-md text-sm">
                {error}
              </div>
            )}
            
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
                    required
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
                    required
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
                    required
                    minLength={6}
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