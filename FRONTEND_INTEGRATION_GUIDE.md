# COMPLETE FRONTEND INTEGRATION GUIDE
# OMR Evaluation System - React Frontend to Flask Backend

## 🚀 OVERVIEW
This guide provides the exact API calls and modifications needed to integrate your React frontend with the Flask backend. All endpoints match your component requirements exactly.

## 📊 API BASE CONFIGURATION

### 1. Update Your API Configuration
```javascript
// src/lib/api.js (create this file)
const API_BASE_URL = 'http://127.0.0.1:5000/api';

// Helper function for authenticated requests
export const apiRequest = async (endpoint, options = {}) => {
  const token = localStorage.getItem('auth_token');
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options.headers,
    },
    ...options,
  };

  const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'API request failed');
  }
  
  return response.json();
};

export default API_BASE_URL;
```

## 🔐 1. LOGIN.JSX INTEGRATION

### Current Login.jsx Issues:
- Uses mock setTimeout instead of real API
- Hardcoded credentials check

### Replace Login Function:
```javascript
// In Login.jsx, replace the mock handleSubmit with:
const handleSubmit = async (e) => {
  e.preventDefault();
  setIsLoading(true);
  setError('');

  try {
    const response = await fetch('http://127.0.0.1:5000/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: formData.email,    // Your form uses 'email' but backend accepts email/username
        password: formData.password,
      }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Login failed');
    }

    // Store auth token and user data
    localStorage.setItem('auth_token', data.token);
    localStorage.setItem('user_data', JSON.stringify(data.user));
    
    // Redirect to dashboard
    navigate('/dashboard');
    
  } catch (err) {
    setError(err.message);
  } finally {
    setIsLoading(false);
  }
};
```

### Test Credentials Available:
```
Email: test@example.com | Password: password123
Email: admin@omr.com | Password: password123
Username: testuser | Password: password123
```

## 📝 2. REGISTER.JSX INTEGRATION

### Current Register.jsx Issues:
- Uses mock setTimeout
- No real user creation

### Replace Register Function:
```javascript
// In Register.jsx, replace the mock handleSubmit with:
const handleSubmit = async (e) => {
  e.preventDefault();
  setIsLoading(true);
  setError('');

  // Frontend validation
  if (formData.password !== formData.confirmPassword) {
    setError('Passwords do not match');
    setIsLoading(false);
    return;
  }

  try {
    const response = await fetch('http://127.0.0.1:5000/api/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: formData.username,
        email: formData.email,
        password: formData.password,
        fullName: formData.fullName,
        schoolName: formData.schoolName || '',
        phone: formData.phone || '',
        role: 'teacher'
      }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Registration failed');
    }

    // Auto-login after successful registration
    localStorage.setItem('auth_token', data.token);
    localStorage.setItem('user_data', JSON.stringify(data.user));
    
    // Redirect to dashboard
    navigate('/dashboard');
    
  } catch (err) {
    setError(err.message);
  } finally {
    setIsLoading(false);
  }
};
```

## 📊 3. DASHBOARD.JSX INTEGRATION

### Current Dashboard.jsx Issues:
- Mock data with static values
- No real batch statistics

### Update Dashboard Data Fetching:
```javascript
// In Dashboard.jsx, replace mock data with real API calls:
import { useEffect, useState } from 'react';
import { apiRequest } from '../lib/api';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [recentBatches, setRecentBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch dashboard statistics
        const [statsResponse, batchesResponse] = await Promise.all([
          apiRequest('/dashboard/stats'),
          apiRequest('/dashboard/recent-batches?limit=5')
        ]);
        
        setStats(statsResponse.stats);
        setRecentBatches(batchesResponse.batches);
        
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Update your JSX to use real data:
  const metrics = [
    {
      title: "Total Batches",
      value: stats?.totalBatches || 0,
      icon: FileText,
      color: "text-blue-600"
    },
    {
      title: "Processing",
      value: stats?.processingBatches || 0,
      icon: Clock,
      color: "text-yellow-600"
    },
    {
      title: "Completed",
      value: stats?.completedBatches || 0,
      icon: CheckCircle,
      color: "text-green-600"
    },
    {
      title: "Flagged",
      value: stats?.flaggedBatches || 0,
      icon: AlertTriangle,
      color: "text-red-600"
    }
  ];

  // Use recentBatches array for the recent activity section
  return (
    // Your existing JSX, but replace mock data with {stats.X} and map over {recentBatches}
  );
};
```

## 📁 4. BATCHES.JSX INTEGRATION

### Current Batches.jsx Issues:
- Hardcoded mock batch data
- No filtering functionality

### Update Batches Component:
```javascript
// In Batches.jsx, replace with real data fetching:
import { useEffect, useState } from 'react';
import { apiRequest } from '../lib/api';

const Batches = () => {
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    status: '',
    subject: '',
    school: ''
  });
  const [pagination, setPagination] = useState({
    page: 1,
    total: 0,
    pages: 0
  });

  const fetchBatches = async () => {
    try {
      setLoading(true);
      
      const queryParams = new URLSearchParams({
        page: pagination.page,
        per_page: 10,
        ...(filters.status && { status: filters.status }),
        ...(filters.subject && { subject: filters.subject }),
        ...(filters.school && { school: filters.school })
      });
      
      const response = await apiRequest(`/batches?${queryParams}`);
      
      setBatches(response.batches);
      setPagination(response.pagination);
      
    } catch (err) {
      console.error('Failed to fetch batches:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBatches();
  }, [filters, pagination.page]);

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({ ...prev, [filterType]: value }));
    setPagination(prev => ({ ...prev, page: 1 })); // Reset to first page
  };

  // Your existing JSX, but map over {batches} array
  return (
    // Replace hardcoded batches with {batches.map(batch => ...)}
  );
};
```

## 📤 5. UPLOAD.JSX INTEGRATION

### Current Upload.jsx Issues:
- Mock file upload with setTimeout
- No real file processing

### Replace Upload Function:
```javascript
// In Upload.jsx, replace the mock handleUpload with:
const handleUpload = async () => {
  if (!selectedFiles.length || !answerKey) {
    alert('Please select both OMR sheet images and answer key');
    return;
  }

  if (!examName.trim() || !answerKeyVersion.trim()) {
    alert('Please provide exam name and answer key version');
    return;
  }

  setUploading(true);
  setUploadProgress(0);

  try {
    const formData = new FormData();
    
    // Add form fields (matching backend expectations)
    formData.append('examName', examName);
    formData.append('answerKeyVersion', answerKeyVersion);
    formData.append('schoolName', schoolName || '');
    formData.append('grade', grade || '');
    formData.append('subject', subject || '');
    
    // Add answer key file
    formData.append('answerKey', answerKey);
    
    // Add image files
    selectedFiles.forEach(file => {
      formData.append('images', file);
    });

    const token = localStorage.getItem('auth_token');
    
    const response = await fetch('http://127.0.0.1:5000/api/upload', {
      method: 'POST',
      headers: {
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: formData,
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Upload failed');
    }

    // Success - redirect to batch results
    alert('Upload successful! Processing started.');
    navigate(`/results/${data.batch_id}`);
    
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  } finally {
    setUploading(false);
    setUploadProgress(0);
  }
};

// Add real progress tracking if needed:
const uploadWithProgress = async (formData) => {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const progress = Math.round((e.loaded / e.total) * 100);
        setUploadProgress(progress);
      }
    });
    
    xhr.onload = () => {
      if (xhr.status === 201) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(JSON.parse(xhr.responseText).error));
      }
    };
    
    xhr.onerror = () => reject(new Error('Upload failed'));
    
    xhr.open('POST', 'http://127.0.0.1:5000/api/upload');
    xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_token')}`);
    xhr.send(formData);
  });
};
```

## 📊 6. RESULTS.JSX INTEGRATION

### Current Results.jsx Issues:
- Mock student data
- No real OMR processing results

### Update Results Component:
```javascript
// In Results.jsx, replace with real data fetching:
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { apiRequest } from '../lib/api';

const Results = () => {
  const { batchId } = useParams();
  const [batch, setBatch] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    subject: '',
    status: '',
    min_score: '',
    max_score: ''
  });
  const [pagination, setPagination] = useState({
    page: 1,
    total: 0,
    pages: 0
  });

  const fetchResults = async () => {
    try {
      setLoading(true);
      
      const queryParams = new URLSearchParams({
        page: pagination.page,
        per_page: 50,
        ...(filters.subject && { subject: filters.subject }),
        ...(filters.status && { status: filters.status }),
        ...(filters.min_score && { min_score: filters.min_score }),
        ...(filters.max_score && { max_score: filters.max_score })
      });
      
      const response = await apiRequest(`/batches/${batchId}/results?${queryParams}`);
      
      setBatch(response.batch);
      setResults(response.results);
      setPagination(response.pagination);
      
    } catch (err) {
      console.error('Failed to fetch results:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (batchId) {
      fetchResults();
    }
  }, [batchId, filters, pagination.page]);

  const handleExport = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      const response = await fetch(`http://127.0.0.1:5000/api/batches/${batchId}/export`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `results_${batch?.exam_name}_${batchId}.csv`;
        a.click();
      }
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  // Your existing JSX, but map over {results} array
  return (
    // Replace mock data with real {batch} and {results}
  );
};
```

## 📋 7. RESULTS TABLE COMPONENT INTEGRATION

### Update ResultsTable.jsx for Subject Filtering:
```javascript
// In ResultsTable.jsx, add real subject filtering:
const ResultsTable = ({ results, onSubjectFilter, selectedSubject }) => {
  const getSubjectScore = (result, subject) => {
    const subjectScores = result.subject_scores || {};
    return subjectScores[subject] || 0;
  };

  const getFilteredResults = () => {
    if (!selectedSubject) return results;
    
    return results.filter(result => {
      const score = getSubjectScore(result, selectedSubject);
      return score > 0; // Show only students who attempted this subject
    });
  };

  return (
    <div className="overflow-x-auto">
      {/* Subject filter buttons */}
      <div className="mb-4 flex gap-2">
        <button 
          onClick={() => onSubjectFilter('')}
          className={`px-4 py-2 rounded ${!selectedSubject ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          All Subjects
        </button>
        {['math', 'physics', 'chemistry', 'history'].map(subject => (
          <button
            key={subject}
            onClick={() => onSubjectFilter(subject)}
            className={`px-4 py-2 rounded capitalize ${selectedSubject === subject ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          >
            {subject}
          </button>
        ))}
      </div>

      {/* Table with real data */}
      <table className="min-w-full bg-white">
        <thead>
          <tr>
            <th>Student ID</th>
            <th>Student Name</th>
            <th>Score</th>
            <th>Percentage</th>
            <th>Grade</th>
            <th>Status</th>
            {selectedSubject && <th>{selectedSubject} Score</th>}
          </tr>
        </thead>
        <tbody>
          {getFilteredResults().map(result => (
            <tr key={result.id}>
              <td>{result.student_id}</td>
              <td>{result.student_name}</td>
              <td>{result.score}/{result.total_marks}</td>
              <td>{result.percentage.toFixed(1)}%</td>
              <td>{result.grade}</td>
              <td>
                <span className={`px-2 py-1 rounded text-sm ${
                  result.status === 'Completed' ? 'bg-green-100 text-green-800' :
                  result.status === 'Flagged' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {result.status}
                </span>
              </td>
              {selectedSubject && (
                <td>{getSubjectScore(result, selectedSubject)}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

## 🔐 8. AUTHENTICATION CONTEXT (OPTIONAL)

### Create Auth Context for Better State Management:
```javascript
// src/contexts/AuthContext.jsx
import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('auth_token');
    const savedUser = localStorage.getItem('user_data');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
    
    setLoading(false);
  }, []);

  const login = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
    localStorage.setItem('auth_token', authToken);
    localStorage.setItem('user_data', JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
  };

  const value = {
    user,
    token,
    login,
    logout,
    isAuthenticated: !!token
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
```

## 🚀 9. ENVIRONMENT SETUP

### Create .env file in your frontend:
```env
# Frontend .env file
VITE_API_BASE_URL=http://127.0.0.1:5000/api
VITE_FILE_BASE_URL=http://127.0.0.1:5000/api/files
```

## 🧪 10. TESTING CHECKLIST

### Phase 1: Authentication Testing
- [ ] Login with test credentials
- [ ] Register new user
- [ ] Token storage and retrieval
- [ ] Protected route access

### Phase 2: Dashboard Testing
- [ ] Dashboard stats loading
- [ ] Recent batches display
- [ ] Navigation to other pages

### Phase 3: Upload Testing
- [ ] File selection (images + Excel)
- [ ] Form validation
- [ ] Upload progress
- [ ] Batch creation

### Phase 4: Results Testing
- [ ] Batch results loading
- [ ] Student data display
- [ ] Subject filtering
- [ ] Export functionality

### Phase 5: Complete Flow Testing
- [ ] Upload → Process → Results
- [ ] Batch management
- [ ] User session management

## 🔧 11. DEBUGGING TIPS

### Common Issues:
1. **CORS Errors**: Backend has CORS enabled for localhost:3000 and localhost:5173
2. **Token Issues**: Check localStorage for 'auth_token'
3. **API Errors**: Check browser Network tab for detailed error messages
4. **File Upload**: Ensure FormData is used, not JSON
5. **Database Connection**: Run the SQL schema first

### Debug Console Commands:
```javascript
// Check authentication
console.log('Token:', localStorage.getItem('auth_token'));
console.log('User:', JSON.parse(localStorage.getItem('user_data') || '{}'));

// Test API connection
fetch('http://127.0.0.1:5000/api/dashboard/stats', {
  headers: { Authorization: `Bearer ${localStorage.getItem('auth_token')}` }
})
.then(r => r.json())
.then(console.log);
```

## ✅ FINAL STEPS

1. **Start Backend**: Run `python complete_frontend_integrated_app.py`
2. **Run SQL Schema**: Execute `frontend_based_complete_schema.sql`
3. **Start Frontend**: Run `npm run dev` or `yarn dev`
4. **Test Login**: Use `test@example.com` / `password123`
5. **Upload Files**: Test with sample OMR images
6. **View Results**: Check processed results

Your React frontend will now be fully connected to the Flask backend with real data flow!