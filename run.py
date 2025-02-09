import subprocess
import os
import webbrowser
from time import sleep

def run_streamlit():
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Start Streamlit
    process = subprocess.Popen([
        'streamlit', 'run',
        'app.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ])
    
    # Wait for server to start
    sleep(2)
    
    # Open browser
    webbrowser.open('http://localhost:8501')
    
    return process

if __name__ == "__main__":
    process = run_streamlit()
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()