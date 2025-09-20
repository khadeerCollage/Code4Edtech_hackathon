import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Button } from "@/components/ui/button.jsx";
import { Input } from "@/components/ui/input.jsx";
import { Label } from "@/components/ui/label.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import { useToast } from "@/hooks/use-toast.js";
import { Upload as UploadIcon, Image as ImageIcon, X, ClipboardList, Camera } from "lucide-react";

const Upload = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [examName, setExamName] = useState("");
  const [answerKeyVersion, setAnswerKeyVersion] = useState("");
  const [files, setFiles] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const acceptFiles = useCallback((incoming) => {
    const arr = Array.from(incoming || []);
    const accepted = arr.filter((file) => file.type && file.type.startsWith("image/"));
    return accepted;
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);

      const accepted = acceptFiles(e.dataTransfer.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
      } else {
        toast({ title: "Invalid File Type", description: "Please upload images only (JPG, PNG)", variant: "destructive" });
      }
    },
    [acceptFiles, toast]
  );

  const handleFileSelect = (e) => {
    if (e.target.files) {
      const accepted = acceptFiles(e.target.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
      } else {
        toast({ title: "Invalid File Type", description: "Please upload images only (JPG, PNG)", variant: "destructive" });
      }
    }
  };

  const handleCameraSelect = (e) => {
    if (e.target.files) {
      const accepted = acceptFiles(e.target.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
        toast({ title: "Captured", description: `${accepted.length} image${accepted.length > 1 ? "s" : ""} added from camera` });
      } else {
        toast({ title: "No image captured", description: "Please try again", variant: "destructive" });
      }
    }
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!examName || !answerKeyVersion || files.length === 0) {
      toast({ title: "Missing Information", description: "Please fill in all fields and select files", variant: "destructive" });
      return;
    }

    setIsUploading(true);

    setTimeout(() => {
      toast({ title: "Upload Successful", description: `${files.length} image${files.length !== 1 ? "s" : ""} uploaded for processing` });
      navigate("/dashboard");
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 animate-fade-in">
          <Button variant="outline" onClick={() => navigate("/dashboard")} className="mb-4 hover:shadow-md transition-all duration-200">
            ← Back to Dashboard
          </Button>
          <h1 className="text-3xl font-bold text-foreground">Upload OMR Sheets</h1>
          <p className="text-muted-foreground">Upload images (JPG/PNG) containing OMR sheets for evaluation</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="animate-scale-in shadow-lg lg:col-span-2 w-full">
            <CardHeader>
              <CardTitle>Upload OMR Images</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
                  isDragOver ? "border-primary bg-primary/5 shadow-glow" : "border-muted-foreground/25 hover:border-primary/50"
                }`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
              >
                <UploadIcon className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Drag and drop images here</h3>
                <p className="text-muted-foreground mb-4">or</p>
                <div className="flex items-center justify-center gap-3 flex-wrap">
                  <Button variant="outline" asChild>
                    <label htmlFor="file-upload" className="cursor-pointer">
                      Browse Images or Folders
                      <input id="file-upload" type="file" multiple accept="image/*" webkitdirectory="" directory="" onChange={handleFileSelect} className="hidden" />
                    </label>
                  </Button>
                  <Button variant="outline" asChild>
                    <label htmlFor="camera-capture" className="cursor-pointer inline-flex items-center gap-2">
                      <Camera className="w-4 h-4" /> Use Camera
                      <input id="camera-capture" type="file" accept="image/*" capture="environment" multiple onChange={handleCameraSelect} className="hidden" />
                    </label>
                  </Button>
                </div>
              </div>

              {files.length > 0 && (
                <div className="mt-6 space-y-2">
                  <h4 className="font-medium text-foreground">Selected Images ({files.length})</h4>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg animate-bounce-in" style={{ animationDelay: `${index * 0.1}s` }}>
                        <div className="flex items-center space-x-3">
                          <ImageIcon className="w-5 h-5 text-primary" />
                          <div>
                            <p className="font-medium text-sm">{file.name}</p>
                            <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeFile(index)} className="text-muted-foreground hover:text-destructive">
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="animate-slide-up shadow-lg lg:col-span-2 w-full">
            <CardHeader>
              <CardTitle>Exam Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className="border-2 border-dashed rounded-lg p-6 transition-all duration-300 border-muted-foreground/25 hover:border-primary/50 focus-within:border-primary focus-within:bg-primary/5 focus-within:shadow-glow"
              >
                <div className="text-center mb-4">
                  <ClipboardList className="w-10 h-10 text-muted-foreground mx-auto mb-2" />
                  <h3 className="text-lg font-semibold">Provide Exam Details</h3>
                  <p className="text-sm text-muted-foreground">Fill in the exam info and choose the answer key version</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="examName">Exam Name</Label>
                    <Input
                      id="examName"
                      type="text"
                      placeholder="e.g., Mathematics Final Exam 2024"
                      value={examName}
                      onChange={(e) => setExamName(e.target.value)}
                      className="transition-all duration-200 focus:shadow-glow"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="answerKey">Answer Key Version</Label>
                    <Select value={answerKeyVersion} onValueChange={setAnswerKeyVersion}>
                      <SelectTrigger className="transition-all duration-200 focus:shadow-glow">
                        <SelectValue placeholder="Select answer key version" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="version-a">Version A</SelectItem>
                        <SelectItem value="version-b">Version B</SelectItem>
                        <SelectItem value="version-c">Version C</SelectItem>
                        <SelectItem value="version-d">Version D</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button type="submit" variant="gradient" className="w-full transition-all duration-300" disabled={isUploading || files.length === 0}>
                    {isUploading ? "Processing Upload..." : `Upload ${files.length} Image${files.length !== 1 ? "s" : ""}`}
                  </Button>
                </form>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Upload;
