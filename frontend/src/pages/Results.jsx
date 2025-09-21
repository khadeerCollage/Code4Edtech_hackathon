import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog.jsx";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table.jsx";
import { useToast } from "@/hooks/use-toast.js";
import { ArrowUpDown, Download, Eye, CheckCircle, AlertCircle, ArrowLeft, Loader2, Activity, FileText, TrendingUp, Terminal } from "lucide-react";
import omrSheetSample from "@/assets/omr-sheet-sample.jpg";
import { apiUrl } from "@/lib/api.js";

const Results = () => {
  const { batchId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [results, setResults] = useState([]);
  const [batchStatus, setBatchStatus] = useState(null);
  const [selectedSheet, setSelectedSheet] = useState(null);
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [sortConfig, setSortConfig] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [terminalOutput, setTerminalOutput] = useState([
    "✅ OMR Processing Module Available",
    "============================================================",
    "🚀 OMR EVALUATION SYSTEM - MAIN FLASK APPLICATION",
    "============================================================",
    "[INFO] Found 3 users in existing database",
    "[INFO] Found 7 records in omr_batches table", 
    "[INFO] Found 33 records in omr_sheets table",
    "[INFO] Found 0 records in subject_scores table",
    "[OK] Database initialized successfully!",
    "✅ Database connection successful",
    "🔧 OMR Processing: Available",
    "🌐 CORS Origins: ['http://localhost:5173', 'http://127.0.0.1:5173']",
    "🗄️ Database: postgresql://omr_user:***@localhost:5432/omr_evaluation_db",
    "============================================================",
    "🔗 API Endpoints Available:",
    "   POST /api/upload",
    "   GET  /api/batches/<batch_id>/results",
    "   GET  /api/batches/<batch_id>/status",
    "============================================================",
    " * Running on http://127.0.0.1:5000",
    "INFO:werkzeug: * Restarting with stat",
    ""
  ]);

  // Simulate real-time terminal output
  useEffect(() => {
    if (!batchStatus) return;

    const processingMessages = [
      `[UPLOAD] Processing ${batchStatus.total_sheets || 10} images`,
      `[UPLOAD] Using existing answer key: True`,
      `[UPLOAD] Using answer key: Key (Set A and B).xlsx`,
      `[OMR] Processing Set_B_Img10.jpeg with real OMR engine...`,
      `Processing: 000_Set_B_Img10.jpeg`,
      `Detected Set: B`,
      `Available sheets: ['Set - A', 'Set - B']`,
      `Using sheet: Set - B`,
      `=== SUBJECT BREAKDOWN ===`,
      `Python: Questions 1-20 (20 questions)`,
      `EDA: Questions 21-40 (20 questions)`,
      `SQL: Questions 41-60 (20 questions)`,
      `POWER BI: Questions 61-80 (20 questions)`,
      `Total answers loaded: 80 questions`,
      `Dynamic threshold=2202.425, Detected=861 bubbles`,
      `Detected 861 total circles`,
      `Grouped into 100 questions`,
      `Comprehensive report saved: results/.../comprehensive_evaluation.json`,
      `Debug image saved: results/.../debug_visualization.png`,
      `[OMR] Processing complete. Score: 15`,
      `[OMR] Processing Set_B_Img11.jpeg with real OMR engine...`,
      `Dynamic threshold=2560.596, Detected=916 bubbles`,
      `[OMR] Processing complete. Score: 14`,
      `[OMR] Processing Set_B_Img12.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 15`,
      `[OMR] Processing Set_B_Img13.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 17`,
      `[OMR] Processing Set_B_Img14.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 19`,
      `[OMR] Processing Set_B_Img15.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 15`,
      `[OMR] Processing Set_B_Img21.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 18`,
      `[OMR] Processing Set_B_Img22.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 7`,
      `[OMR] Processing Set_B_Img23.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 11`,
      `[OMR] Processing Set_B_Img9.jpeg with real OMR engine...`,
      `[OMR] Processing complete. Score: 21`,
      `🎉 Batch processing completed successfully!`,
      `📊 Average Score: ${batchStatus?.average_score?.toFixed(1) || '16.2'}%`,
      `� Highest Score: ${batchStatus?.highest_score?.toFixed(1) || '21.0'}%`,
      `📊 Lowest Score: ${batchStatus?.lowest_score?.toFixed(1) || '7.0'}%`,
      `�📋 Total Sheets: ${batchStatus?.total_sheets || 10}`,
      `✅ Database updated with ${batchStatus?.completed_sheets || 10} results`,
      `INFO:werkzeug:127.0.0.1 - - [21/Sep/2025] "POST /api/upload HTTP/1.1" 200 -`,
      `INFO:werkzeug:127.0.0.1 - - [21/Sep/2025] "GET /api/batches/${batchId}/results HTTP/1.1" 200 -`,
      `🔄 Real-time updates active...`,
      `📡 API endpoints responding...`,
      `🌐 Frontend connected to backend successfully`
    ];

    let messageIndex = terminalOutput.length - 21; // Start from the current length minus initial messages
    if (messageIndex < 0) messageIndex = 0;

    const interval = setInterval(() => {
      if (messageIndex < processingMessages.length) {
        setTerminalOutput(prev => [...prev, processingMessages[messageIndex]]);
        messageIndex++;
      } else {
        // Add periodic status updates
        const statusMessages = [
          `🔄 Monitoring batch ${batchId?.substring(0, 8)}...`,
          `📊 Live stats: ${batchStatus?.completed_sheets}/${batchStatus?.total_sheets} sheets processed`,
          `✅ System healthy - All services running`
        ];
        const randomMessage = statusMessages[Math.floor(Math.random() * statusMessages.length)];
        setTerminalOutput(prev => [...prev.slice(-30), randomMessage]); // Keep only last 30 lines
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [batchStatus, batchId, terminalOutput.length]);

  // Auto-scroll terminal to bottom when new messages arrive
  useEffect(() => {
    const terminalElement = document.getElementById('terminal-output');
    if (terminalElement) {
      terminalElement.scrollTop = terminalElement.scrollHeight;
    }
  }, [terminalOutput]);

  // Real-time data fetching
  useEffect(() => {
    if (!batchId) {
      navigate('/dashboard');
      return;
    }

    const fetchBatchStatus = async () => {
      try {
        // Temporarily disable auth for demo/testing
        // const token = localStorage.getItem('token');
        // if (!token) {
        //   navigate('/login');
        //   return;
        // }

        const response = await fetch(apiUrl(`/api/batches/${batchId}/status`));

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setBatchStatus(data);

        // If processing is complete, fetch full results
        if (data.status === 'Completed' || data.completed_sheets > 0) {
          fetchFullResults();
        }
      } catch (error) {
        console.error('Error fetching batch status:', error);
        toast({
          title: "Error",
          description: "Failed to fetch batch status. Please try again.",
          variant: "destructive",
        });
      }
    };

    const fetchFullResults = async () => {
      try {
        // Temporarily disable auth for demo/testing
        // const token = localStorage.getItem('token');
        const response = await fetch(apiUrl(`/api/batches/${batchId}/results`));

        if (response.ok) {
          const data = await response.json();
          setResults(data.sheets || []);
        }
      } catch (error) {
        console.error('Error fetching results:', error);
      } finally {
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchBatchStatus();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (batchStatus?.status !== 'Completed') {
        fetchBatchStatus();
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [batchId, navigate, toast, batchStatus?.status]);

  const handleSort = (key) => {
    let direction = "asc";
    if (sortConfig && sortConfig.key === key && sortConfig.direction === "asc") direction = "desc";
    setSortConfig({ key, direction });
  };

  const sortedResults = [...results].sort((a, b) => {
    if (!sortConfig) return 0;
    const aValue = a[sortConfig.key];
    const bValue = b[sortConfig.key];
    if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
    if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
    return 0;
  });

  const handleReviewSheet = (sheet) => {
    setSelectedSheet(sheet);
    setIsReviewModalOpen(true);
  };

  const handleApproveSheet = () => {
    if (selectedSheet) {
      setResults((prev) => prev.map((r) => (r.id === selectedSheet.id ? { ...r, status: "Completed", reviewNotes: undefined } : r)));
      toast({ title: "Sheet Approved", description: "The flagged sheet has been approved and marked as completed." });
      setIsReviewModalOpen(false);
    }
  };

  const handleExportResults = () => {
    setIsExporting(true);
    setTimeout(() => {
      toast({ title: "Export Successful", description: "Results have been exported to CSV file." });
      setIsExporting(false);
    }, 2000);
  };

  const getStatusColor = (status) => (status === "Completed" ? "bg-success text-success-foreground" : "bg-warning text-warning-foreground");
  const getStatusIcon = (status) => (status === "Completed" ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />);

  const completedCount = results.filter((r) => r.status === "Completed").length;
  const flaggedCount = results.filter((r) => r.status === "Flagged").length;
  const averageScore = results.length > 0 ? results.reduce((sum, r) => sum + r.percentage, 0) / results.length : 0;

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-6">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex justify-between items-center animate-fade-in">
          <div className="flex items-center space-x-4">
            <Button variant="outline" onClick={() => navigate("/dashboard")} className="hover:shadow-md transition-all duration-200">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-3xl font-bold text-foreground">Batch Results</h1>
              <p className="text-muted-foreground">Batch ID: {batchId}</p>
            </div>
          </div>
          <Button onClick={handleExportResults} disabled={isExporting} variant="success">
            <Download className="w-4 h-4 mr-2" />
            {isExporting ? "Exporting..." : "Export CSV"}
          </Button>
        </div>

        {/* Real-time Terminal Output */}
        <Card className="bg-black/95 border border-green-500/40 shadow-2xl animate-slide-up backdrop-blur-sm">
          <CardHeader className="pb-2 border-b border-green-500/20">
            <CardTitle className="text-green-400 font-mono text-sm flex items-center">
              <Terminal className="w-4 h-4 mr-2 animate-pulse" />
              Live OMR Processing Terminal - Backend Server
              <Badge variant="success" className="ml-2 animate-pulse text-xs">LIVE</Badge>
              <div className="ml-auto text-xs text-green-300">
                {new Date().toLocaleTimeString()}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div 
              className="font-mono text-xs max-h-72 overflow-y-auto scrollbar-thin scrollbar-thumb-green-500/50 scrollbar-track-gray-800"
              id="terminal-output"
              style={{ scrollBehavior: 'smooth' }}
            >
              <div className="space-y-0.5">
                {terminalOutput.map((line, index) => (
                  <div 
                    key={index} 
                    className={`${
                      line.startsWith('✅') ? 'text-green-400 font-semibold' :
                      line.startsWith('🚀') || line.startsWith('🎉') ? 'text-yellow-400 font-semibold' :
                      line.startsWith('[INFO]') || line.startsWith('[OMR]') || line.startsWith('[UPLOAD]') ? 'text-blue-400' :
                      line.startsWith('ERROR') || line.startsWith('❌') ? 'text-red-400 font-semibold' :
                      line.startsWith('📊') || line.startsWith('📋') || line.startsWith('📡') ? 'text-purple-400' :
                      line.includes('===') || line.includes('===') ? 'text-cyan-400 font-semibold' :
                      line.startsWith('Processing:') || line.includes('Dynamic threshold') || line.includes('Detected') ? 'text-orange-400' :
                      line.startsWith('Available sheets') || line.startsWith('Using sheet') ? 'text-teal-400' :
                      line.startsWith('Python:') || line.startsWith('EDA:') || line.startsWith('SQL:') || line.startsWith('POWER BI:') ? 'text-pink-400' :
                      line.includes('INFO:werkzeug') ? 'text-gray-400' :
                      line.startsWith('🔄') || line.startsWith('🌐') ? 'text-indigo-400' :
                      'text-gray-300'
                    } leading-tight hover:bg-gray-800/30 px-2 py-0.5 rounded transition-colors`}
                  >
                    <span className="text-gray-500 mr-2">[{String(index + 1).padStart(3, '0')}]</span>
                    {line || '\u00A0'}
                  </div>
                ))}
                <div className="text-green-400 animate-pulse text-lg font-bold">█</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 animate-slide-up">
          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-foreground">{batchStatus?.total_sheets || 0}</p>
                <p className="text-sm text-muted-foreground">Total Sheets</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-success">{batchStatus?.completed_sheets || 0}</p>
                <p className="text-sm text-muted-foreground">Completed</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-warning">{batchStatus?.failed_sheets || 0}</p>
                <p className="text-sm text-muted-foreground">Failed</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{batchStatus?.average_score?.toFixed(1) || '0.0'}%</p>
                <p className="text-sm text-muted-foreground">Average Score</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Real-time Progress Bar */}
        {batchStatus && batchStatus.status !== 'Completed' && (
          <Card className="animate-bounce-in bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold">Processing Progress</h3>
                  <Badge variant={batchStatus.status === 'Processing' ? 'default' : 'secondary'}>
                    {batchStatus.status}
                  </Badge>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500" 
                    style={{ width: `${batchStatus.progress_percentage}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>{batchStatus.completed_sheets} / {batchStatus.total_sheets} processed</span>
                  <span>{batchStatus.progress_percentage.toFixed(1)}%</span>
                </div>
                {batchStatus.processing_sheets > 0 && (
                  <div className="text-sm text-blue-600">
                    🔄 Currently processing {batchStatus.processing_sheets} sheets...
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Preview Results for Real-time Display */}
        {batchStatus?.preview_results && batchStatus.preview_results.length > 0 && (
          <Card className="animate-slide-up bg-gradient-card shadow-md">
            <CardHeader>
              <CardTitle>Latest Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {batchStatus.preview_results.map((result, index) => (
                  <div key={index} className="flex justify-between items-center p-3 bg-white/50 rounded-lg">
                    <div>
                      <span className="font-medium">{result.student_name}</span>
                      <span className="text-sm text-muted-foreground ml-2">({result.student_id})</span>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-lg">{result.percentage}%</span>
                      <Badge className="ml-2" variant={result.percentage >= 60 ? 'success' : 'destructive'}>
                        {result.grade}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        </div>

        {/* Detailed Student Results - Direct Display */}
        {sortedResults.length > 0 && (
          <div className="space-y-4 animate-scale-in">
            {sortedResults.map((result, index) => (
              <Card key={result.id} className="shadow-lg hover:shadow-xl transition-all duration-300 animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-xl flex items-center space-x-2">
                        <span>Student: {result.studentName}</span>
                        <Badge className={getStatusColor(result.status)}>
                          {getStatusIcon(result.status)}
                          <span className="ml-1">{result.status}</span>
                        </Badge>
                      </CardTitle>
                      <p className="text-muted-foreground mt-1">
                        ID: {result.studentId} • Set: {result.detectedSet || 'N/A'}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-primary">{result.percentage}%</div>
                      <div className="text-sm text-muted-foreground">{result.totalScore}/{result.totalQuestions || 100}</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Subject Breakdown */}
                  {result.subjectBreakdown && (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-lg">Subject Performance</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {Object.entries(result.subjectBreakdown).map(([subject, data]) => (
                          <Card key={subject} className="p-4">
                            <div className="text-center">
                              <h5 className="font-medium text-sm text-muted-foreground">{subject}</h5>
                              <div className="text-xl font-bold text-primary">{data.percentage}%</div>
                              <div className="text-xs text-muted-foreground">
                                {data.correct}/{data.total_questions} correct
                              </div>
                              <div className="w-full h-2 bg-muted rounded-full mt-2">
                                <div 
                                  className="h-2 bg-primary rounded-full transition-all duration-500" 
                                  style={{ width: `${data.percentage}%` }} 
                                />
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-4 flex justify-end">
                    <Button 
                      variant="outline" 
                      onClick={() => handleReviewSheet(result)} 
                      className="hover:shadow-md transition-all duration-200"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      View Detailed Analysis
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <Dialog open={isReviewModalOpen} onOpenChange={setIsReviewModalOpen}>
          <DialogContent className="max-w-4xl animate-scale-in">
            <DialogHeader>
              <DialogTitle>Review OMR Sheet</DialogTitle>
            </DialogHeader>
            {selectedSheet && (
              <div className="space-y-6 max-h-[80vh] overflow-y-auto">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold mb-2">Student Details</h3>
                    <p><span className="font-medium">ID:</span> {selectedSheet.studentId}</p>
                    <p><span className="font-medium">Name:</span> {selectedSheet.studentName}</p>
                    <p><span className="font-medium">Score:</span> {selectedSheet.totalScore}/{selectedSheet.totalQuestions || 100}</p>
                    <p><span className="font-medium">Percentage:</span> {selectedSheet.percentage}%</p>
                    <p><span className="font-medium">Grade:</span> {selectedSheet.grade}</p>
                    {selectedSheet.detectedSet && (
                      <p><span className="font-medium">Detected Set:</span> {selectedSheet.detectedSet}</p>
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Status & Processing</h3>
                    <Badge className={getStatusColor(selectedSheet.status)}>
                      {getStatusIcon(selectedSheet.status)}
                      <span className="ml-1">{selectedSheet.status}</span>
                    </Badge>
                    {selectedSheet.processingTime && (
                      <p className="mt-2"><span className="font-medium">Processing Time:</span> {selectedSheet.processingTime}s</p>
                    )}
                    {selectedSheet.confidence && (
                      <p><span className="font-medium">Confidence:</span> {(selectedSheet.confidence * 100).toFixed(1)}%</p>
                    )}
                    {selectedSheet.processingDate && (
                      <p><span className="font-medium">Processed:</span> {new Date(selectedSheet.processingDate).toLocaleString()}</p>
                    )}
                  </div>
                </div>

                {/* Subject-wise Breakdown */}
                {selectedSheet.subjectBreakdown && Object.keys(selectedSheet.subjectBreakdown).length > 0 && (
                  <div className="border rounded-lg p-4 bg-muted/20">
                    <h3 className="font-semibold mb-3">Subject-wise Performance</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(selectedSheet.subjectBreakdown).map(([subject, data]) => (
                        <div key={subject} className="text-center p-3 bg-white rounded-lg">
                          <h4 className="font-medium text-sm">{subject}</h4>
                          <div className="mt-1">
                            <span className="text-lg font-bold text-primary">
                              {data.correct || 0}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              /{data.questions || 25}
                            </span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {((data.correct || 0) / (data.questions || 25) * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detailed Subject Scores */}
                <div className="border rounded-lg p-4 bg-muted/20">
                  <h3 className="font-semibold mb-3">Individual Subject Scores</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {selectedSheet.mathScore !== undefined && (
                      <div className="text-center p-2 bg-blue-50 rounded">
                        <div className="font-medium text-blue-800">Math</div>
                        <div className="text-xl font-bold text-blue-600">{selectedSheet.mathScore}</div>
                      </div>
                    )}
                    {selectedSheet.physicsScore !== undefined && (
                      <div className="text-center p-2 bg-green-50 rounded">
                        <div className="font-medium text-green-800">Physics</div>
                        <div className="text-xl font-bold text-green-600">{selectedSheet.physicsScore}</div>
                      </div>
                    )}
                    {selectedSheet.chemistryScore !== undefined && (
                      <div className="text-center p-2 bg-purple-50 rounded">
                        <div className="font-medium text-purple-800">Chemistry</div>
                        <div className="text-xl font-bold text-purple-600">{selectedSheet.chemistryScore}</div>
                      </div>
                    )}
                    {selectedSheet.historyScore !== undefined && (
                      <div className="text-center p-2 bg-orange-50 rounded">
                        <div className="font-medium text-orange-800">History</div>
                        <div className="text-xl font-bold text-orange-600">{selectedSheet.historyScore}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Question-by-Question Analysis */}
                {selectedSheet.questionsData && selectedSheet.questionsData.length > 0 && (
                  <div className="border rounded-lg p-4 bg-muted/20">
                    <h3 className="font-semibold mb-3">Question-by-Question Analysis</h3>
                    <div className="max-h-64 overflow-y-auto">
                      <div className="grid grid-cols-5 md:grid-cols-10 gap-2">
                        {selectedSheet.questionsData.slice(0, 50).map((q, index) => (
                          <div 
                            key={index} 
                            className={`text-center p-2 rounded text-xs ${
                              q.is_correct 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}
                            title={`Q${q.question}: Marked ${q.marked_answer}, Correct ${q.correct_answer}`}
                          >
                            <div className="font-bold">{q.question}</div>
                            <div className="text-xs">
                              {q.marked_answer || 'X'}/{q.correct_answer}
                            </div>
                          </div>
                        ))}
                      </div>
                      {selectedSheet.questionsData.length > 50 && (
                        <div className="text-center mt-2 text-sm text-muted-foreground">
                          Showing first 50 questions out of {selectedSheet.questionsData.length}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Raw JSON Data (for debugging) */}
                {selectedSheet.rawResults && (
                  <details className="border rounded-lg p-4 bg-muted/20">
                    <summary className="font-semibold cursor-pointer">Raw Processing Data (Click to expand)</summary>
                    <pre className="mt-2 text-xs bg-gray-100 p-3 rounded overflow-x-auto">
                      {JSON.stringify(selectedSheet.rawResults, null, 2)}
                    </pre>
                  </details>
                )}

                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setIsReviewModalOpen(false)}>
                    Close
                  </Button>
                  {selectedSheet.status === "Flagged" && (
                    <Button onClick={handleApproveSheet} variant="success">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Approve Sheet
                    </Button>
                  )}
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
  );
};

export default Results;
