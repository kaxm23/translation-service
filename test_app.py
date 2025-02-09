import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
from app import app, settings

client = TestClient(app)

def test_translate_pdf_invalid_file():
    """Test uploading an invalid file type."""
    response = client.post(
        "/translate/",
        files={"file": ("test.txt", "content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

def test_translate_pdf_success():
    """Test successful PDF upload."""
    # Create a test PDF file
    test_pdf = Path("test.pdf")
    test_pdf.write_bytes(b"%PDF-1.4\n%...")  # Minimal PDF content
    
    with open(test_pdf, "rb") as f:
        response = client.post(
            "/translate/",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    
    # Clean up
    test_pdf.unlink()

def test_get_status_invalid_task():
    """Test getting status for invalid task ID."""
    response = client.get("/status/invalid-task-id")
    assert response.status_code == 404

def test_download_invalid_task():
    """Test downloading for invalid task ID."""
    response = client.get("/download/invalid-task-id")
    assert response.status_code == 404