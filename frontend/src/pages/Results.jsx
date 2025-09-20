import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog.jsx";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table.jsx";
import { useToast } from "@/hooks/use-toast.js";
import { ArrowUpDown, Download, Eye, CheckCircle, AlertCircle, ArrowLeft } from "lucide-react";
import omrSheetSample from "@/assets/omr-sheet-sample.jpg";

const Results = () => {
  const { batchId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [results, setResults] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState(null);
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [sortConfig, setSortConfig] = useState(null);
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    const mockResults = [
      { id: "sheet-001", studentId: "ST001", studentName: "Alice Johnson", score: 85, totalMarks: 100, percentage: 85, status: "Completed" },
      { id: "sheet-002", studentId: "ST002", studentName: "Bob Smith", score: 92, totalMarks: 100, percentage: 92, status: "Completed" },
      { id: "sheet-003", studentId: "ST003", studentName: "Carol Davis", score: 78, totalMarks: 100, percentage: 78, status: "Flagged", reviewNotes: "Ambiguous markings in questions 15-17" },
      { id: "sheet-004", studentId: "ST004", studentName: "David Wilson", score: 88, totalMarks: 100, percentage: 88, status: "Completed" },
      { id: "sheet-005", studentId: "ST005", studentName: "Eva Brown", score: 95, totalMarks: 100, percentage: 95, status: "Completed" },
    ];
    setTimeout(() => setResults(mockResults), 800);
  }, []);

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
    <div className="min-h-screen bg-gradient-to-br from-background to-muted p-6">
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

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 animate-slide-up">
          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-foreground">{results.length}</p>
                <p className="text-sm text-muted-foreground">Total Sheets</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-success">{completedCount}</p>
                <p className="text-sm text-muted-foreground">Completed</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-warning">{flaggedCount}</p>
                <p className="text-sm text-muted-foreground">Flagged</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-md">
            <CardContent className="p-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{averageScore.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">Average Score</p>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="animate-scale-in shadow-lg">
          <CardHeader>
            <CardTitle>Individual Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="cursor-pointer hover:text-primary transition-colors" onClick={() => handleSort("studentId")}>
                      <div className="flex items-center">
                        Student ID
                        <ArrowUpDown className="w-4 h-4 ml-1" />
                      </div>
                    </TableHead>
                    <TableHead className="cursor-pointer hover:text-primary transition-colors" onClick={() => handleSort("studentName")}>
                      <div className="flex items-center">
                        Student Name
                        <ArrowUpDown className="w-4 h-4 ml-1" />
                      </div>
                    </TableHead>
                    <TableHead className="cursor-pointer hover:text-primary transition-colors" onClick={() => handleSort("score")}>
                      <div className="flex items-center">
                        Score
                        <ArrowUpDown className="w-4 h-4 ml-1" />
                      </div>
                    </TableHead>
                    <TableHead className="cursor-pointer hover:text-primary transition-colors" onClick={() => handleSort("percentage")}>
                      <div className="flex items-center">
                        Percentage
                        <ArrowUpDown className="w-4 h-4 ml-1" />
                      </div>
                    </TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedResults.map((result, index) => (
                    <TableRow key={result.id} className="animate-fade-in hover:bg-muted/50 transition-colors" style={{ animationDelay: `${index * 0.05}s` }}>
                      <TableCell className="font-medium">{result.studentId}</TableCell>
                      <TableCell>{result.studentName}</TableCell>
                      <TableCell>
                        {result.score}/{result.totalMarks}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{result.percentage}%</span>
                          <div className="w-16 h-2 bg-muted rounded-full">
                            <div className="h-2 bg-primary rounded-full transition-all duration-500" style={{ width: `${result.percentage}%` }} />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge className={getStatusColor(result.status)}>
                          {getStatusIcon(result.status)}
                          <span className="ml-1">{result.status}</span>
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Button variant="ghost" size="sm" onClick={() => handleReviewSheet(result)} className="hover:shadow-md transition-all duration-200">
                          <Eye className="w-4 h-4 mr-1" />
                          Review
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>

      <Dialog open={isReviewModalOpen} onOpenChange={setIsReviewModalOpen}>
        <DialogContent className="max-w-4xl animate-scale-in">
          <DialogHeader>
            <DialogTitle>Review OMR Sheet</DialogTitle>
          </DialogHeader>
          {selectedSheet && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold mb-2">Student Details</h3>
                  <p>
                    <span className="font-medium">ID:</span> {selectedSheet.studentId}
                  </p>
                  <p>
                    <span className="font-medium">Name:</span> {selectedSheet.studentName}
                  </p>
                  <p>
                    <span className="font-medium">Score:</span> {selectedSheet.score}/{selectedSheet.totalMarks}
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Status</h3>
                  <Badge className={getStatusColor(selectedSheet.status)}>
                    {getStatusIcon(selectedSheet.status)}
                    <span className="ml-1">{selectedSheet.status}</span>
                  </Badge>
                  {selectedSheet.reviewNotes && (
                    <div className="mt-2">
                      <p className="text-sm text-muted-foreground">Notes:</p>
                      <p className="text-sm">{selectedSheet.reviewNotes}</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="border rounded-lg p-4 bg-muted/20">
                <h3 className="font-semibold mb-2">OMR Sheet Preview</h3>
                <div className="bg-white rounded border p-4">
                  <img src={omrSheetSample} alt="OMR Sheet Sample" className="w-full max-w-md mx-auto rounded shadow-md" />
                </div>
              </div>

              {selectedSheet.status === "Flagged" && (
                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setIsReviewModalOpen(false)}>
                    Close
                  </Button>
                  <Button onClick={handleApproveSheet} variant="success">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Approve Sheet
                  </Button>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Results;
