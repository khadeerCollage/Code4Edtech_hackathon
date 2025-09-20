import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import ResultsTable from "@/components/ResultsTable.jsx";
import Overview from "@/components/Overview.jsx";
import { useToast } from "@/hooks/use-toast.js";
import { FileText, Upload, CheckCircle, AlertCircle, Clock, Play, X, FileSpreadsheet, Camera, Image as ImageIcon } from "lucide-react";

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

  useEffect(() => {
    const mockBatches = [
      {
        id: "batch-001",
        examName: "Mathematics Final Exam 2024",
        uploadDate: "2024-03-15",
        status: "Completed",
        totalSheets: 120,
        completedSheets: 120,
      },
      {
        id: "batch-002",
        examName: "Physics Midterm Test",
        uploadDate: "2024-03-14",
        status: "Processing",
        totalSheets: 85,
        completedSheets: 45,
      },
      {
        id: "batch-003",
        examName: "Chemistry Quiz Set A",
        uploadDate: "2024-03-13",
        status: "Flagged",
        totalSheets: 60,
        completedSheets: 58,
      },
    ];

    setTimeout(() => {
      setBatches(mockBatches);
      setMetrics({
        totalBatches: mockBatches.length,
        processingBatches: mockBatches.filter((b) => b.status === "Processing").length,
        completedBatches: mockBatches.filter((b) => b.status === "Completed").length,
        flaggedSheets: mockBatches.reduce(
          (sum, b) => sum + (b.status === "Flagged" ? b.totalSheets - b.completedSheets : 0),
          0
        ),
      });
    }, 800);
  }, []);

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

  const handleUpload = async () => {
    if (!questionSet || files.length === 0) {
      toast({ title: "Missing Information", description: "Please select question set and upload files", variant: "destructive" });
      return;
    }

    setIsUploading(true);
    setTimeout(() => {
      toast({ title: "Upload Successful", description: `${files.length} image${files.length !== 1 ? "s" : ""} uploaded for processing` });
      setFiles([]);
      setQuestionSet("");
      setIsUploading(false);
    }, 3000);
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
    if (!questionSet) {
      toast({ title: "Select Question Set", description: "Please choose a question set before uploading the answer key.", variant: "destructive" });
      return;
    }
    if (!answerFile) {
      toast({ title: "No File Selected", description: "Please choose an Excel/CSV file to upload.", variant: "destructive" });
      return;
    }

    setIsUploadingAnswer(true);
    setTimeout(() => {
      toast({ title: "Answer Key Uploaded", description: `${answerFile.name} uploaded for the selected question set.` });
      setIsUploadingAnswer(false);
      setAnswerFile(null);
      setShowResults(true);
      setResults([
        { name: "Alice Smith", section: "A", math: 85, physics: 78, chemistry: 92, history: 87 },
        { name: "Bob Johnson", section: "A", math: 72, physics: 65, chemistry: 88, history: 90 },
        { name: "Eva Jensen", section: "B", math: 79, physics: 82, chemistry: 76, history: 89 },
        { name: "Charlie Brown", section: "A", math: 78, physics: 80, chemistry: 75, history: 82 },
        { name: "Diana Miller", section: "B", math: 90, physics: 88, chemistry: 94, history: 89 },
        { name: "Melvin Bale", section: "A", math: 60, physics: 95, chemistry: 99, history: 82 },
        { name: "Henna Malik", section: "B", math: 83, physics: 70, chemistry: 70, history: 92 },
        { name: "Diane Small", section: "A", math: 88, physics: 95, chemistry: 61, history: 91 },
      ]);
    }, 2000);
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
    <div className="min-h-screen bg-gradient-to-br from-background to-muted p-3 md:p-6 no-select">
      <div className="max-w-7xl mx-auto space-y-4 md:space-y-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 animate-fade-in no-select">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-foreground no-select">Dashboard</h1>
            <p className="text-muted-foreground no-select">OMR Evaluation Platform</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button onClick={() => navigate("/workflow-demo")} variant="outline" size="sm" className="text-xs md:text-sm button-hover-pulse button-icon-rotate">
              <Play className="w-4 h-4 mr-1 md:mr-2" />
              View Workflow
            </Button>
            <Button onClick={() => navigate("/upload")} variant="gradient" size="sm" className="text-xs md:text-sm button-hover-glow button-shimmer">
              <Upload className="w-4 h-4 mr-1 md:mr-2" />
              Full Upload Page
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

              <Button onClick={handleUpload} variant="gradient" className="w-full transition-all duration-300 button-hover-glow button-ripple" disabled={isUploading || files.length === 0 || !questionSet} size="sm">
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

              <Button onClick={handleAnswerUpload} variant="gradient" className="w-full transition-all duration-300 button-hover-glow button-ripple" disabled={isUploadingAnswer || !answerFile || !questionSet} size="sm">
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
            <Card className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select">
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

            <Card className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select">
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

            <Card className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select">
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

            <Card className="bg-gradient-card shadow-md hover:shadow-lg transition-all duration-300 no-select">
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

          <Card className="animate-scale-in shadow-lg no-select">
            <CardHeader className="pb-3 md:pb-4">
              <CardTitle className="text-base md:text-lg no-select">Recent Batches</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {batches.map((batch, index) => (
                  <div key={batch.id} className="flex items-center justify-between p-3 md:p-4 border rounded-lg hover:bg-muted/50 transition-colors duration-200 cursor-pointer animate-fade-in no-select" style={{ animationDelay: `${index * 0.1}s` }} onClick={() => navigate(`/results/${batch.id}`)}>
                    <div className="flex items-center space-x-2 md:space-x-3 flex-1 min-w-0 no-select">
                      <div className="p-1.5 md:p-2 bg-primary/10 rounded">
                        <FileText className="w-3 md:w-4 h-3 md:h-4 text-primary" />
                      </div>
                      <div className="min-w-0 flex-1 no-select">
                        <h3 className="font-medium text-sm md:text-base text-foreground truncate no-select">{batch.examName}</h3>
                        <p className="text-xs text-muted-foreground no-select">{new Date(batch.uploadDate).toLocaleDateString()}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2 md:space-x-4 flex-shrink-0">
                      <div className="text-right hidden sm:block">
                        <p className="text-xs font-medium">{batch.completedSheets}/{batch.totalSheets}</p>
                        <div className="w-16 md:w-20 h-1.5 md:h-2 bg-muted rounded-full mt-1">
                          <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${(batch.completedSheets / batch.totalSheets) * 100}%` }} />
                        </div>
                      </div>
                      <Badge className={`${getStatusColor(batch.status)} text-xs`}>
                        {getStatusIcon(batch.status)}
                        <span className="ml-1 hidden sm:inline">{batch.status}</span>
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
