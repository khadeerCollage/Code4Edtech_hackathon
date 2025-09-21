import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const Login = () => {
  const [credentials, setCredentials] = useState({ username: "", password: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    
    if (!credentials.username.trim() || !credentials.password.trim()) {
      setError("Both username and password are required");
      return;
    }

    console.log('Starting login process...');
    console.log('Credentials:', { username: credentials.username });

    setIsLoading(true);
    setError("");

    try {
      console.log('Sending login request...');

      const response = await fetch('http://localhost:5000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: credentials.username.trim(),
          password: credentials.password.trim()
        })
      });

      console.log('Login response status:', response.status);

      const data = await response.json();
      console.log('Login response data:', data);

      if (response.ok && data.success) {
        console.log('Login successful!');
        
        // Store token in localStorage
        if (data.token) {
          localStorage.setItem('token', data.token);
          console.log('Token stored successfully');
        }

        // Store user info
        if (data.user) {
          localStorage.setItem('user', JSON.stringify(data.user));
          console.log('User info stored successfully');
        }

        // Clear form
        setCredentials({ username: "", password: "" });
        
        // Navigate to dashboard
        navigate('/dashboard');
      } else {
        console.error('Login failed:', data.message);
        setError(data.message || 'Invalid credentials');
      }
    } catch (error) {
      console.error('Login error:', error);
      
      if (error.message.includes('Failed to fetch')) {
        setError("Cannot connect to server. Please ensure the backend is running.");
      } else {
        setError("Login failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden flex items-center justify-center bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-6">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />

      <div className="w-full max-w-sm p-[1px] rounded-2xl bg-gradient-to-br from-white/80 via-indigo-400/50 to-white/40 shadow-2xl animate-fade-in">
        <Card className="rounded-2xl bg-gradient-to-b from-white/90 to-white/70 backdrop-blur-xl border border-white/60 shadow-[0_10px_40px_rgba(99,102,241,0.15)]">
          <CardHeader className="text-center space-y-2 pb-3">
            <CardTitle className="text-2xl font-bold text-blue-600">OMR Evaluation Platform</CardTitle>
            <CardDescription className="text-gray-600">Sign in to your evaluator account</CardDescription>
            <div className="flex justify-center gap-2 pt-2">
              <Button className="h-9 px-4 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white">Login</Button>
              <Button asChild variant="outline" className="h-9 px-4">
                <Link to="/register">Register</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-gray-700 font-medium">Username</Label>
                <div className="p-[2px] rounded-md gradient-border-rtl focus-within:animate-gradient-rtl">
                  <Input
                    id="username"
                    type="text"
                    placeholder="Enter your username"
                    value={credentials.username}
                    onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
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
                    placeholder="Enter your password"
                    value={credentials.password}
                    onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                    className="rounded-md bg-white border-transparent shadow-sm focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                </div>
              </div>
              {error && <p className="text-red-500 text-sm">{error}</p>}
              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white font-medium py-3 rounded-md transition-all duration-300 mt-6 hover:shadow-lg hover:shadow-indigo-400/30 focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white active:scale-[.99]"
                disabled={isLoading}
              >
                {isLoading ? "Signing in..." : "Login"}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Login;