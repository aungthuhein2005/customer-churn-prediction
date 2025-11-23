import { useState, useRef } from 'react';
import { useNavigate } from 'react-router';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';
import { dataStore } from '../utils/dataStore';
import { BACKEND_API } from '../utils/config';

export function UploadScreen() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (!validTypes.includes(file.type) && !file.name.endsWith('.csv') && !file.name.endsWith('.xlsx')) {
      setError('Please upload a valid CSV or Excel file.');
      setSelectedFile(null);
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setError('File size must be less than 10MB.');
      setSelectedFile(null);
      return;
    }

    setError('');
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      // Create FormData to send file to backend
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Upload file to backend
      const response = await fetch(`${BACKEND_API}/upload_csv/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const csvData = await response.json();
      
      // Check for error response from backend
      if (csvData.error) {
        setError(csvData.error);
        setIsUploading(false);
        return;
      }
      
      if (!csvData.headers || csvData.headers.length === 0) {
        setError('The uploaded file appears to be empty.');
        setIsUploading(false);
        return;
      }

      // Store the data
      dataStore.csvData = csvData;
      dataStore.fileName = selectedFile.name;

      // Navigate to mapping screen
      navigate('/mapping');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to upload the file. Please ensure it\'s a valid CSV file and the backend is running.';
      setError(errorMessage);
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card className="shadow-xl border-2">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center">
            <Upload className="w-8 h-8 text-indigo-600" />
          </div>
          <CardTitle>Welcome to Churn Predictor! 🚀</CardTitle>
          <CardDescription>
            Upload your customer data to analyze churn risk.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="flex-1"
              >
                <FileText className="mr-2 h-4 w-4" />
                {selectedFile ? selectedFile.name : 'Choose File'}
              </Button>
              <Button 
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="px-8"
              >
                {isUploading ? 'Uploading...' : 'Upload'}
              </Button>
            </div>

            <p className="text-sm text-gray-500 text-center">
              Supported formats: CSV, Excel (.xlsx)
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4 space-y-3">
            <h4 className="text-sm text-gray-700">📋 Before uploading:</h4>
            <ul className="text-sm text-gray-600 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 mt-0.5">•</span>
                <span><strong>Max file size:</strong> 10MB</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 mt-0.5">•</span>
                <span><strong>Required columns:</strong> A column indicating churn (Yes/No) and a unique customer ID</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-indigo-600 mt-0.5">•</span>
                <span><strong>Optional columns:</strong> Support tickets, last login date, usage metrics, etc.</span>
              </li>
            </ul>
          </div>

          <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
            <p className="text-sm text-indigo-900">
              💡 <strong>Tip:</strong> Make sure your file contains a column indicating churn (Yes/No) 
              and a unique customer ID. The more data columns you include, the better the predictions!
            </p>
          </div>

          <div className="text-center">
            <a 
              href="/example-customer-churn.csv" 
              download="example-customer-churn.csv"
              className="text-sm text-indigo-600 hover:underline"
            >
              Download example CSV template
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
