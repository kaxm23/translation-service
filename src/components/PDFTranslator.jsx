import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './PDFTranslator.css';

const PDFTranslator = () => {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [translationProgress, setTranslationProgress] = useState(0);
  const [taskId, setTaskId] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    
    if (!file || !file.type === 'application/pdf') {
      setError('Please upload a valid PDF file');
      return;
    }

    try {
      setError(null);
      setUploadStatus('uploading');
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Upload file
      const response = await axios.post('http://localhost:8000/translate/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTaskId(response.data.task_id);
      setUploadStatus('processing');
      
      // Start progress polling
      pollTranslationProgress(response.data.task_id);
      
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload file');
      setUploadStatus('error');
    }
  }, []);

  const pollTranslationProgress = async (tid) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/status/${tid}`);
        const { status, progress, error: taskError } = response.data;

        setTranslationProgress(progress * 100);

        if (status === 'completed') {
          clearInterval(pollInterval);
          setUploadStatus('completed');
        } else if (status === 'failed') {
          clearInterval(pollInterval);
          setError(taskError || 'Translation failed');
          setUploadStatus('error');
        }
      } catch (err) {
        clearInterval(pollInterval);
        setError('Failed to check translation status');
        setUploadStatus('error');
      }
    }, 1000);
  };

  const downloadTranslation = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/download/${taskId}`, {
        responseType: 'blob',
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'translated.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download translation');
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: 'application/pdf',
    multiple: false,
  });

  return (
    <div className="translator-container">
      <h1>PDF Translator</h1>
      <p className="subtitle">English to Arabic Translation</p>

      {!uploadStatus && (
        <div 
          {...getRootProps()} 
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the PDF here</p>
          ) : (
            <p>Drag and drop a PDF here, or click to select</p>
          )}
        </div>
      )}

      {uploadStatus === 'uploading' && (
        <div className="status-container">
          <div className="spinner" />
          <p>Uploading PDF...</p>
        </div>
      )}

      {uploadStatus === 'processing' && (
        <div className="status-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${translationProgress}%` }}
            />
          </div>
          <p>Translating... {translationProgress.toFixed(1)}%</p>
        </div>
      )}

      {uploadStatus === 'completed' && (
        <div className="status-container success">
          <p>Translation completed!</p>
          <button 
            className="download-button"
            onClick={downloadTranslation}
          >
            Download Translation
          </button>
        </div>
      )}

      {error && (
        <div className="error-container">
          <p className="error-message">{error}</p>
          <button 
            className="retry-button"
            onClick={() => {
              setError(null);
              setUploadStatus(null);
              setTranslationProgress(0);
              setTaskId(null);
            }}
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

export default PDFTranslator;