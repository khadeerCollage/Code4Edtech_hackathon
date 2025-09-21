import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import ResultsTable from "@/components/ResultsTable.jsx";
import Overview from "@/components/Overview.jsx";
import { useToast } from "@/hooks/use-toast.js";
import { FileText, Upload, CheckCircle, AlertCircle, Clock, Play, X, FileSpreadsheet, Camera, Image as ImageIcon, LogOut } from "lucide-react";
import { apiUrl } from "@/lib/api.js";

const Dashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [batches, setBatches] = useState([]);
  const [metrics, setMetrics] = useState({
    totalBatches: 0,
    processingBatches: 0,
    completedBatches: 0,
    flaggedSheets: 0,
  });

  // Upload states
  const [files, setFiles] = useState([]);
  const [questionSet, setQuestionSet] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  // Answer key (Excel) upload states
  const [answerFile, setAnswerFile] = useState(null);
  const [isDragOverAnswer, setIsDragOverAnswer] = useState(false);
  const [isUploadingAnswer, setIsUploadingAnswer] = useState(false);
  // Mock results after uploads
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState([]);

  // Real data fetching functions
  const fetchDashboardStats = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }

      const response = await fetch(apiUrl('/api/dashboard/stats'), {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (error) {
      console.error('Failed to fetch dashboard stats:', error);
    }
  };

  const fetchRecentBatches = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch(apiUrl('/api/dashboard/recent-batches'), {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setBatches(data.batches || []);
      }
    } catch (error) {
      console.error('Failed to fetch recent batches:', error);
    }
  };

  const fetchLatestResults = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      // Get the latest batch with completed results
      const batchesResponse = await fetch(apiUrl('/api/batches'), {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (batchesResponse.ok) {
        const batchesData = await batchesResponse.json();
        const completedBatches = batchesData.batches?.filter(b => b.status === 'Completed') || [];
        
        if (completedBatches.length > 0) {
          const latestBatch = completedBatches[0]; // Most recent completed batch
          
          // Fetch results for the latest batch
          const resultsResponse = await fetch(apiUrl(`/api/batches/${latestBatch.id}/results`), {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });

          if (resultsResponse.ok) {
            const resultsData = await resultsResponse.json();
            
            // Transform database results to match ResultsTable format
            const transformedResults = resultsData.results?.map(result => ({
              name: result.studentName || `Student ${result.studentId}`,
              section: result.detectedSet || 'A',
              math: result.mathScore || 0,
              physics: result.physicsScore || 0,
              chemistry: result.chemistryScore || 0,
              history: result.historyScore || 0
            })) || [];

            setResults(transformedResults);
            setShowResults(transformedResults.length > 0);
          }
        }
      }
    } catch (error) {
      console.error('Failed to fetch latest results:', error);
      setResults([]);
      setShowResults(false);
    }
  };

  useEffect(() => {
    fetchDashboardStats();
    fetchRecentBatches();
    fetchLatestResults();
    
    // Set up periodic refresh for real-time updates
    const interval = setInterval(() => {
      fetchDashboardStats();
      fetchRecentBatches();
    }, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, [navigate]);

  // Upload handlers
  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const acceptImages = useCallback((list) => {
    const arr = Array.from(list || []);
    return arr.filter((f) => f.type && f.type.startsWith("image/"));
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);

      const accepted = acceptImages(e.dataTransfer.files);

      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
      } else {
        toast({ title: "Invalid File Type", description: "Please upload images only (JPG, PNG)", variant: "destructive" });
      }
    },
    [acceptImages, toast]
  );

  // Answer key drag & drop handlers
  const onDragOverAnswer = useCallback((e) => {
    e.preventDefault();
    setIsDragOverAnswer(true);
  }, []);

  const onDragLeaveAnswer = useCallback((e) => {
    e.preventDefault();
    setIsDragOverAnswer(false);
  }, []);

  const onDropAnswer = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOverAnswer(false);

      const droppedFiles = Array.from(e.dataTransfer.files);
      const valid = droppedFiles.find(
        (file) =>
          [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "text/csv",
          ].includes(file.type) ||
          file.name.endsWith(".xlsx") ||
          file.name.endsWith(".xls") ||
          file.name.endsWith(".csv")
      );

      if (valid) {
        setAnswerFile(valid);
      } else {
        toast({
          title: "Invalid File Type",
          description: "Please upload an Excel (.xlsx/.xls) or CSV file",
          variant: "destructive",
        });
      }
    },
    [toast]
  );

  const handleFileSelect = (e) => {
    if (e.target.files) {
      const accepted = acceptImages(e.target.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
      } else {
        toast({ title: "Invalid File Type", description: "Please upload images only (JPG, PNG)", variant: "destructive" });
      }
    }
  };

  const handleCameraSelect = (e) => {
    if (e.target.files) {
      const accepted = acceptImages(e.target.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
        toast({ title: "Captured", description: `${accepted.length} image${accepted.length > 1 ? "s" : ""} added from camera` });
      } else {
        toast({ title: "No image captured", description: "Please try again", variant: "destructive" });
      }
    }
  };

  const handleAnswerFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const valid =
        [
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "application/vnd.ms-excel",
          "text/csv",
        ].includes(file.type) ||
        file.name.endsWith(".xlsx") ||
        file.name.endsWith(".xls") ||
        file.name.endsWith(".csv");

      if (valid) {
        setAnswerFile(file);
      } else {
        toast({ title: "Invalid File Type", description: "Please upload an Excel (.xlsx/.xls) or CSV file", variant: "destructive" });
      }
    }
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const removeAnswerFile = () => setAnswerFile(null);

  const handleLogout = () => {
    // Clear authentication data
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    
    // Show logout success message
    toast({
      title: "Logged Out",
      description: "You have been successfully logged out",
      variant: "default",
    });
    
    // Redirect to login page
    navigate('/login');
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast({ title: "No Files Selected", description: "Please add images to upload.", variant: "destructive" });
      return;
    }

    if (!questionSet) {
      toast({ title: "Question Set Required", description: "Please select a question set before uploading.", variant: "destructive" });
      return;
    }

    setIsUploading(true);

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }

      const formData = new FormData();
      files.forEach((file) => {
        formData.append('images', file);
      });
      formData.append('questionSet', questionSet);
      formData.append('examName', `Quick Upload ${new Date().toLocaleDateString()}`);

      const response = await fetch(apiUrl('/api/upload'), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        toast({ 
          title: "Upload Successful", 
          description: `${files.length} image${files.length !== 1 ? "s" : ""} uploaded successfully. Processing started.` 
        });
        
        // Navigate to real-time results page
        navigate(`/results/${data.batch_id}`);
        
        // Reset form
        setFiles([]);
        setQuestionSet("");
        setIsUploading(false);
        
        // Refresh dashboard data
        fetchDashboardStats();
        fetchRecentBatches();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast({ 
        title: "Upload Failed", 
        description: error.message || "Failed to upload images. Please try again.",
        variant: "destructive" 
      });
      setIsUploading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "Completed":
        return "bg-success text-success-foreground";
      case "Processing":
        return "bg-processing text-processing-foreground";
      case "Flagged":
        return "bg-warning text-warning-foreground";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const handleAnswerUpload = async () => {
    if (!answerFile) {
      toast({ title: "No File Selected", description: "Please choose an Excel/CSV file to upload.", variant: "destructive" });
      return;
    }

    setIsUploadingAnswer(true);

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }

      const formData = new FormData();
      formData.append('answerKey', answerFile);

      const response = await fetch(apiUrl('/api/upload-answer-key'), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        toast({ 
          title: "Answer Key Uploaded", 
          description: `${answerFile.name} uploaded successfully.` 
        });
        setIsUploadingAnswer(false);
        setAnswerFile(null);
        
        // Refresh the latest results
        await fetchLatestResults();
        
        // Refresh dashboard stats
        fetchDashboardStats();
        fetchRecentBatches();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Answer key upload failed');
      }
    } catch (error) {
      console.error('Answer key upload error:', error);
      toast({ 
        title: "Upload Failed", 
        description: error.message || "Failed to upload answer key. Please try again.",
        variant: "destructive" 
      });
      setIsUploadingAnswer(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "Completed":
        return <CheckCircle className="w-4 h-4" />;
      case "Processing":
        return <Clock className="w-4 h-4" />;
      case "Flagged":
        return <AlertCircle className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-3 md:p-6 no-select">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />
      <div className="max-w-7xl mx-auto space-y-4 md:space-y-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 animate-fade-in no-select">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-foreground no-select">Dashboard</h1>
            <p className="text-muted-foreground no-select">OMR Evaluation Platform</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleLogout} variant="destructive" size="sm" className="text-xs md:text-sm button-hover-glow">
              <LogOut className="w-4 h-4 mr-1 md:mr-2" />
              Logout
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
          {/* Quick Upload */}
          <Card className="shadow-lg animate-slide-up w-full no-select">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg md:text-xl no-select">Quick Upload OMR Images</CardTitle>
              <p className="text-sm text-muted-foreground no-select">Upload images (JPG/PNG) and select question set</p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className={`w-full border-2 border-dashed rounded-lg p-6 md:p-8 text-center transition-all duration-300 no-select ${
                  isDragOver ? "border-primary bg-primary/5 shadow-glow" : "border-muted-foreground/25 hover:border-primary/50"
                }`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
              >
                <Upload className="w-8 md:w-12 h-8 md:h-12 text-muted-foreground mx-auto mb-3 md:mb-4" />
                <h3 className="text-base md:text-lg font-semibold mb-2 no-select">Drop images here</h3>
                <p className="text-sm text-muted-foreground mb-3 md:mb-4 no-select">or</p>
                <div className="flex items-center justify-center gap-2 md:gap-3 flex-wrap">
                  <Button variant="outline" size="sm" asChild>
                    <label htmlFor="file-upload" className="cursor-pointer text-xs md:text-sm">
                      Browse Images or Folders
                      <input id="file-upload" type="file" multiple accept="image/*" webkitdirectory="" directory="" onChange={handleFileSelect} className="hidden" />
                    </label>
                  </Button>
                  <Button variant="outline" size="sm" asChild>
                    <label htmlFor="camera-capture" className="cursor-pointer inline-flex items-center gap-1.5 md:gap-2 text-xs md:text-sm">
                      <Camera className="w-4 h-4" /> Use Camera
                      <input id="camera-capture" type="file" accept="image/*" capture="environment" multiple onChange={handleCameraSelect} className="hidden" />
                    </label>
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground no-select">Question Set</label>
                <Select value={questionSet} onValueChange={setQuestionSet}>
                  <SelectTrigger className="transition-all duration-200 focus:shadow-glow">
                    <SelectValue placeholder="Select question set (A/B/C/D)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="set-a">Set A</SelectItem>
                    <SelectItem value="set-b">Set B</SelectItem>
                    <SelectItem value="set-c">Set C</SelectItem>
                    <SelectItem value="set-d">Set D</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {files.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium text-sm text-foreground no-select">Selected Images ({files.length})</h4>
                  <div className="space-y-2 max-h-32 md:max-h-48 overflow-y-auto">
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-2 md:p-3 bg-muted/50 rounded-lg animate-bounce-in text-xs md:text-sm no-select" style={{ animationDelay: `${index * 0.1}s` }}>
                        <div className="flex items-center space-x-2 md:space-x-3 flex-1 min-w-0 no-select">
                          <ImageIcon className="w-4 h-4 text-primary flex-shrink-0" />
                          <div className="min-w-0 flex-1 no-select">
                            <p className="font-medium truncate no-select">{file.name}</p>
                            <p className="text-xs text-muted-foreground no-select">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeFile(index)} className="text-muted-foreground hover:text-destructive p-1 h-auto">
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <Button onClick={handleUpload} variant="gradient" className="w-full transition-all duration-300 button-hover-glow button-ripple" disabled={isUploading || files.length === 0} size="sm">
                {isUploading ? "Processing Upload..." : `Upload ${files.length} Image${files.length !== 1 ? "s" : ""}`}
              </Button>
            </CardContent>
          </Card>

          {/* Answer Key Upload */}
          <Card className="shadow-lg animate-slide-up w-full no-select">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg md:text-xl no-select">Upload Answer Key (Excel/CSV)</CardTitle>
              <p className="text-sm text-muted-foreground no-select">Upload a spreadsheet with correct answers for the selected question set</p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className={`w-full border-2 border-dashed rounded-lg p-6 md:p-8 text-center transition-all duration-300 no-select ${
                  isDragOverAnswer ? "border-primary bg-primary/5 shadow-glow" : "border-muted-foreground/25 hover:border-primary/50"
                }`}
                onDragOver={onDragOverAnswer}
                onDragLeave={onDragLeaveAnswer}
                onDrop={onDropAnswer}
              >
                <FileSpreadsheet className="w-8 md:w-12 h-8 md:h-12 text-muted-foreground mx-auto mb-3 md:mb-4" />
                <h3 className="text-base md:text-lg font-semibold mb-2 no-select">Drop .xlsx/.xls/.csv file here</h3>
                <p className="text-sm text-muted-foreground mb-3 md:mb-4 no-select">or</p>
                <Button variant="outline" size="sm" asChild>
                  <label htmlFor="answer-upload" className="cursor-pointer text-xs md:text-sm">
                    Browse Files
                    <input id="answer-upload" type="file" accept=".xlsx,.xls,.csv" onChange={handleAnswerFileSelect} className="hidden" />
                  </label>
                </Button>
              </div>

              {answerFile && (
                <div className="space-y-2">
                  <h4 className="font-medium text-sm text-foreground no-select">Selected File</h4>
                  <div className="flex items-center justify-between p-2 md:p-3 bg-muted/50 rounded-lg text-xs md:text-sm no-select">
                    <div className="flex items-center space-x-2 md:space-x-3 no-select">
                      <FileSpreadsheet className="w-4 h-4 text-primary" />
                      <span className="font-medium no-select">{answerFile.name}</span>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => removeAnswerFile()} className="text-muted-foreground hover:text-destructive p-1 h-auto">
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              )}

              <Button onClick={handleAnswerUpload} variant="gradient" className="w-full transition-all duration-300 button-hover-glow button-ripple" disabled={isUploadingAnswer || !answerFile} size="sm">
                {isUploadingAnswer ? "Uploading Answer Key..." : "Upload Answer Key"}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="mt-4">
          <ResultsTable title="Results" data={results} />
        </div>

        <div className="mt-6">
          <Overview />
        </div>

        <div className="space-y-4 md:space-y-6 mt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 animate-slide-up">
            <Card 
              className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select cursor-pointer transform hover:scale-105 hover:-translate-y-1" 
              onClick={() => navigate("/total-batches")}
            >
              <CardContent className="p-3 md:p-4 no-select">
                <div className="flex items-center space-x-2 md:space-x-3 no-select">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    <FileText className="w-4 md:w-5 h-4 md:h-5 text-primary" />
                  </div>
                  <div className="no-select">
                    <p className="text-xs text-muted-foreground no-select">Total Batches</p>
                    <p className="text-lg md:text-xl font-bold text-foreground no-select">{metrics.totalBatches}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card 
              className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select cursor-pointer transform hover:scale-105 hover:-translate-y-1" 
              onClick={() => navigate("/processing-batches")}
            >
              <CardContent className="p-3 md:p-4 no-select">
                <div className="flex items-center space-x-2 md:space-x-3 no-select">
                  <div className="p-2 bg-processing/10 rounded-lg">
                    <Clock className="w-4 md:w-5 h-4 md:h-5 text-processing" />
                  </div>
                  <div className="no-select">
                    <p className="text-xs text-muted-foreground no-select">Processing</p>
                    <p className="text-lg md:text-xl font-bold text-foreground no-select">{metrics.processingBatches}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card 
              className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select cursor-pointer transform hover:scale-105 hover:-translate-y-1" 
              onClick={() => navigate("/completed-batches")}
            >
              <CardContent className="p-3 md:p-4 no-select">
                <div className="flex items-center space-x-2 md:space-x-3 no-select">
                  <div className="p-2 bg-success/10 rounded-lg">
                    <CheckCircle className="w-4 md:w-5 h-4 md:h-5 text-success" />
                  </div>
                  <div className="no-select">
                    <p className="text-xs text-muted-foreground no-select">Completed</p>
                    <p className="text-lg md:text-xl font-bold text-foreground no-select">{metrics.completedBatches}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card 
              className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select cursor-pointer transform hover:scale-105 hover:-translate-y-1" 
              onClick={() => navigate("/flagged-sheets")}
            >
              <CardContent className="p-3 md:p-4 no-select">
                <div className="flex items-center space-x-2 md:space-x-3 no-select">
                  <div className="p-2 bg-warning/10 rounded-lg">
                    <AlertCircle className="w-4 md:w-5 h-4 md:h-5 text-warning" />
                  </div>
                  <div className="no-select">
                    <p className="text-xs text-muted-foreground no-select">Flagged Sheets</p>
                    <p className="text-lg md:text-xl font-bold text-foreground no-select">{metrics.flaggedSheets}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Beautiful Footer */}
          <div className="mt-8 animate-fade-in">
            <Card className="bg-gradient-to-r from-red-500/10 via-sky-400/10 to-blue-400/10 border-none shadow-lg">
              <CardContent className="p-6 text-center">
                <div className="flex flex-col items-center justify-center space-y-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">❤️</span>
                    <p className="text-lg font-medium bg-gradient-to-r from-red-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
                      Made with Love
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground font-medium">
                    © 2025 OMR Evaluation Platform
                  </p>
                  <div className="flex items-center space-x-4 mt-3 text-xs text-muted-foreground">
                    <span className="hover:text-primary transition-colors cursor-pointer">Privacy Policy</span>
                    <span className="text-muted-foreground/50">•</span>
                    <span className="hover:text-primary transition-colors cursor-pointer">Terms of Service</span>
                    <span className="text-muted-foreground/50">•</span>
                    <span className="hover:text-primary transition-colors cursor-pointer">Support</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
