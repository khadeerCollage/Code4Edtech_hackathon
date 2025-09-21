import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table.jsx";
import { Input } from "@/components/ui/input.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import { ArrowLeft, Filter, Search, FileText, Clock, CheckCircle, AlertCircle } from "lucide-react";

const useQuery = () => new URLSearchParams(useLocation().search);

const demoBatches = [
  { id: "batch-001", examName: "Mathematics Final Exam 2024", school: "Central High School", uploadDate: "2024-03-15", status: "Completed", totalSheets: 120, completedSheets: 120, grade: "Grade 12", subject: "Mathematics" },
  { id: "batch-002", examName: "Physics Midterm Test", school: "Science Academy", uploadDate: "2024-03-14", status: "Processing", totalSheets: 85, completedSheets: 65, grade: "Grade 11", subject: "Physics" },
  { id: "batch-003", examName: "Chemistry Unit Test", school: "Modern School", uploadDate: "2024-03-13", status: "Completed", totalSheets: 95, completedSheets: 95, grade: "Grade 10", subject: "Chemistry" },
  { id: "batch-004", examName: "Biology Quiz Assessment", school: "Green Valley School", uploadDate: "2024-03-12", status: "Flagged", totalSheets: 75, completedSheets: 70, grade: "Grade 9", subject: "Biology", flaggedSheets: 12 },
  { id: "batch-005", examName: "English Literature Exam", school: "Royal Academy", uploadDate: "2024-03-11", status: "Completed", totalSheets: 110, completedSheets: 110, grade: "Grade 12", subject: "English" },
  { id: "batch-006", examName: "History Annual Test", school: "Heritage School", uploadDate: "2024-03-10", status: "Processing", totalSheets: 90, completedSheets: 45, grade: "Grade 11", subject: "History" },
  { id: "batch-011", examName: "Mathematics Practice Test", school: "Excellence Academy", uploadDate: "2024-03-05", status: "Flagged", totalSheets: 88, completedSheets: 82, grade: "Grade 10", subject: "Mathematics", flaggedSheets: 8 },
];

const statusMeta = {
  All: { icon: FileText, color: "bg-blue-600/90", label: "All Batches" },
  Processing: { icon: Clock, color: "bg-yellow-500/90", label: "Processing" },
  Completed: { icon: CheckCircle, color: "bg-green-600/90", label: "Completed" },
  Flagged: { icon: AlertCircle, color: "bg-red-600/90", label: "Flagged" },
};

const Batches = () => {
  const navigate = useNavigate();
  const query = useQuery();

  const initialFilter = query.get("status") || "All";
  const [statusFilter, setStatusFilter] = useState(initialFilter);
  const [search, setSearch] = useState("");

  useEffect(() => {
    // sync URL when filter changes
    const params = new URLSearchParams();
    if (statusFilter && statusFilter !== "All") params.set("status", statusFilter);
    navigate({ pathname: "/batches", search: params.toString() }, { replace: true });
  }, [statusFilter, navigate]);

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    return demoBatches.filter((b) => {
      const matchesStatus = statusFilter === "All" || b.status === statusFilter;
      if (!s) return matchesStatus;
      return (
        matchesStatus && (
          b.examName.toLowerCase().includes(s) ||
          b.school.toLowerCase().includes(s) ||
          b.subject.toLowerCase().includes(s) ||
          b.grade.toLowerCase().includes(s) ||
          String(b.totalSheets).includes(s)
        )
      );
    });
  }, [statusFilter, search]);

  const StatusButton = ({ name }) => {
    const active = statusFilter === name;
    return (
      <Button
        variant={active ? "gradient" : "outline"}
        size="sm"
        className="min-w-[120px]"
        onClick={() => setStatusFilter(name)}
      >
        {(() => {
          const Icon = statusMeta[name].icon; return <Icon className="w-4 h-4 mr-2"/>;
        })()}
        {statusMeta[name].label}
      </Button>
    );
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-6">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header with back + title */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={() => navigate("/dashboard")}> 
              <ArrowLeft className="w-4 h-4 mr-2" /> Back
            </Button>
            <h1 className="text-2xl md:text-3xl font-bold text-white">Batches</h1>
          </div>
        </div>

        {/* Status buttons */}
        <div className="flex flex-wrap gap-3">
          {Object.keys(statusMeta).map((name) => <StatusButton key={name} name={name} />)}
        </div>

        {/* Search and simple filter */}
        <Card className="bg-white/90 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Search and Filter</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="relative md:col-span-2">
                <Input placeholder="Search by exam, school, subject, grade or sheets..." value={search} onChange={(e)=>setSearch(e.target.value)} className="h-10 pr-9"/>
                <Search className="w-4 h-4 absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"/>
              </div>
              <div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="h-10 w-full">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="All">All</SelectItem>
                    <SelectItem value="Processing">Processing</SelectItem>
                    <SelectItem value="Completed">Completed</SelectItem>
                    <SelectItem value="Flagged">Flagged</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Table */}
        <Card className="shadow-lg overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-base md:text-lg">{statusMeta[statusFilter]?.label || "Batches"}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-xl border bg-white/90 overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="bg-blue-600/90 text-white">
                    <TableHead className="text-white">Exam</TableHead>
                    <TableHead className="text-white">School</TableHead>
                    <TableHead className="text-white">Grade</TableHead>
                    <TableHead className="text-white">Subject</TableHead>
                    <TableHead className="text-white">Status</TableHead>
                    <TableHead className="text-white">Sheets</TableHead>
                    <TableHead className="text-white">Uploaded</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((b) => (
                    <TableRow key={b.id} className="hover:bg-muted/50 cursor-pointer" onClick={()=>navigate(`/results/${b.id}`)}>
                      <TableCell className="text-foreground">{b.examName}</TableCell>
                      <TableCell className="text-foreground">{b.school}</TableCell>
                      <TableCell className="text-foreground">{b.grade}</TableCell>
                      <TableCell className="text-foreground">{b.subject}</TableCell>
                      <TableCell>
                        <Badge variant={b.status === 'Completed' ? 'success' : b.status === 'Processing' ? 'processing' : b.status === 'Flagged' ? 'destructive' : 'outline'}>
                          {b.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-foreground">{b.completedSheets}/{b.totalSheets}</TableCell>
                      <TableCell className="text-foreground">{new Date(b.uploadDate).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
                {filtered.length === 0 && <TableCaption className="text-xs">No batches match your search</TableCaption>}
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Batches;