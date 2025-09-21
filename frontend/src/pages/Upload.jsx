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
    // Accept all image types including common formats
    const imageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'];
    const accepted = arr.filter((file) => {
      // Check by MIME type first
      if (file.type && imageTypes.some(type => file.type.toLowerCase().includes(type.split('/')[1]))) {
        return true;
      }
      // Check by file extension as fallback
      const fileName = file.name.toLowerCase();
      return fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || 
             fileName.endsWith('.png') || fileName.endsWith('.gif') || 
             fileName.endsWith('.webp') || fileName.endsWith('.bmp');
    });
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
        const folderCount = e.target.files.length;
        const imageCount = accepted.length;
        if (imageCount < folderCount) {
          toast({ 
            title: "Files Filtered", 
            description: `Found ${imageCount} images out of ${folderCount} files. Non-image files were automatically excluded.`,
            duration: 4000
          });
        } else {
          toast({ 
            title: "Images Added", 
            description: `${imageCount} image${imageCount > 1 ? 's' : ''} selected successfully`
          });
        }
      } else {
        toast({ 
          title: "No Images Found", 
          description: "Please select a folder containing image files (JPG, PNG, GIF, WebP, BMP)", 
          variant: "destructive" 
        });
      }
    }
  };

  const handleCameraCapture = async () => {
    try {
      // Check if camera is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast({ 
          title: "Camera Not Supported", 
          description: "Your browser doesn't support camera access", 
          variant: "destructive" 
        });
        return;
      }

      // For mobile devices, prefer back camera
      const constraints = {
        video: { 
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Create a video element to capture the frame
      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();

      // Wait for video to load
      video.addEventListener('loadedmetadata', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0);
        
        // Convert to blob
        canvas.toBlob((blob) => {
          const file = new File([blob], `camera-capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
          setFiles((prev) => [...prev, file]);
          toast({ 
            title: "Photo Captured", 
            description: "Image captured from camera successfully" 
          });
          
          // Stop the stream
          stream.getTracks().forEach(track => track.stop());
        }, 'image/jpeg', 0.9);
      });

    } catch (error) {
      console.error('Camera access error:', error);
      toast({ 
        title: "Camera Access Failed", 
        description: "Unable to access camera. Please check permissions.", 
        variant: "destructive" 
      });
    }
  };

  const handleCameraSelect = (e) => {
    if (e.target.files) {
      const accepted = acceptFiles(e.target.files);
      if (accepted.length > 0) {
        setFiles((prev) => [...prev, ...accepted]);
        toast({ 
          title: "Camera Image Added", 
          description: `${accepted.length} image${accepted.length > 1 ? "s" : ""} added from camera` 
        });
      } else {
        toast({ 
          title: "No valid image", 
          description: "Please capture a valid image", 
          variant: "destructive" 
        });
      }
    }
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate files
    if (files.length === 0) {
      toast({ 
        title: "No Files Selected", 
        description: "Please select images to upload", 
        variant: "destructive" 
      });
      return;
    }

    // For now, we'll create a simple mock answer key since it's not required for basic OMR processing
    if (!answerKeyVersion) {
      setAnswerKeyVersion("A"); // Default to version A
    }

    setIsUploading(true);

    try {
      // Prepare form data
      const formData = new FormData();
      
      // Add metadata
      formData.append('examName', examName || `OMR Upload ${new Date().toLocaleDateString()}`);
      formData.append('answerKeyVersion', answerKeyVersion);
      formData.append('schoolName', 'Default School');
      formData.append('grade', '10');
      formData.append('subject', 'General');
      
      // Add image files
      files.forEach((file, index) => {
        formData.append('images', file);
      });
      
      // Use the existing answer key from dataset
      const answerKeyPath = '../../dataset/data/Key (Set A and B).xlsx';
      
      // Create a proper answer key reference
      formData.append('useExistingAnswerKey', 'true');
      formData.append('answerKeyPath', answerKeyPath);

      console.log('Uploading files to backend...');

      // Get token from localStorage
      const token = localStorage.getItem('token');
      
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Upload failed: ${response.status} - ${errorData}`);
      }

      const result = await response.json();
      console.log('Upload successful:', result);

      toast({ 
        title: "Upload Successful", 
        description: `${files.length} image${files.length !== 1 ? "s" : ""} uploaded and processing started`,
        duration: 5000
      });
      
      // Reset form
      setFiles([]);
      setExamName("");
      
      // Navigate to real-time results page
      setTimeout(() => {
        navigate(`/results/${result.batch_id}`);
      }, 1500);

    } catch (error) {
      console.error('Upload error:', error);
      
      let errorMessage = "Upload failed. Please try again.";
      
      if (error.message.includes('Failed to fetch')) {
        errorMessage = "Cannot connect to server. Please ensure you're logged in and the backend is running.";
      } else if (error.message.includes('401')) {
        errorMessage = "Authentication failed. Please login again.";
        // Clear token and redirect to login
        localStorage.removeItem('token');
        navigate('/login');
        return;
      } else if (error.message.includes('400')) {
        errorMessage = "Invalid upload data. Please check your files and try again.";
      }

      toast({ 
        title: "Upload Failed", 
        description: errorMessage,
        variant: "destructive",
        duration: 6000
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-red-500 via-red-300 to-sky-400 p-6">
      <div className="pointer-events-none absolute -top-24 -left-24 h-[38rem] w-[38rem] rounded-full bg-red-500/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-[36rem] w-[36rem] rounded-full bg-sky-400/45 blur-3xl -z-10" />
      <div className="pointer-events-none absolute top-1/3 -right-24 h-[30rem] w-[30rem] rounded-full bg-blue-400/40 blur-3xl -z-10" />
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
                <h3 className="text-lg font-semibold mb-2">Drop images here</h3>
                <p className="text-muted-foreground mb-4">Supports JPG, PNG, GIF, WebP, BMP formats</p>
                <p className="text-muted-foreground mb-4">or</p>
                <div className="flex items-center justify-center gap-3 flex-wrap">
                  <Button variant="outline" asChild>
                    <label htmlFor="file-upload" className="cursor-pointer">
                      Browse Images
                      <input 
                        id="file-upload" 
                        type="file" 
                        multiple 
                        accept="image/*,.jpg,.jpeg,.png,.gif,.webp,.bmp" 
                        onChange={handleFileSelect} 
                        className="hidden" 
                      />
                    </label>
                  </Button>
                  <Button variant="outline" asChild>
                    <label htmlFor="folder-upload" className="cursor-pointer">
                      Select Folder
                      <input 
                        id="folder-upload" 
                        type="file" 
                        webkitdirectory="" 
                        directory="" 
                        multiple 
                        onChange={handleFileSelect} 
                        className="hidden" 
                      />
                    </label>
                  </Button>
                  <Button variant="outline" onClick={handleCameraCapture} className="inline-flex items-center gap-2">
                    <Camera className="w-4 h-4" /> Open Camera
                  </Button>
                  <Button variant="outline" asChild>
                    <label htmlFor="camera-capture" className="cursor-pointer inline-flex items-center gap-2">
                      <Camera className="w-4 h-4" /> Upload from Camera
                      <input 
                        id="camera-capture" 
                        type="file" 
                        accept="image/*" 
                        capture="environment" 
                        multiple 
                        onChange={handleCameraSelect} 
                        className="hidden" 
                      />
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
