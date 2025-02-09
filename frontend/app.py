import streamlit as st
import requests
import time
from datetime import datetime
import json
from pathlib import Path
import base64

class PDFTranslatorApp:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.configure_page()
        self.initialize_session_state()
        
    def configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="PDF Translator",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stProgress .st-bo {
            background-color: #00a67d;
        }
        .success-message {
            padding: 1rem;
            background-color: #dcffe4;
            border-radius: 0.5rem;
            color: #00a67d;
        }
        .error-message {
            padding: 1rem;
            background-color: #ffe4e4;
            border-radius: 0.5rem;
            color: #d32f2f;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'task_id' not in st.session_state:
            st.session_state.task_id = None
        if 'translation_complete' not in st.session_state:
            st.session_state.translation_complete = False
        if 'extracted_text' not in st.session_state:
            st.session_state.extracted_text = None
        if 'translated_text' not in st.session_state:
            st.session_state.translated_text = None

    def render_header(self):
        """Render the application header."""
        st.title("PDF Translator")
        st.markdown("Upload a PDF file to extract text and translate it.")
        
        # Show current time in sidebar
        st.sidebar.markdown("### System Information")
        st.sidebar.text(f"Current Time (UTC):")
        st.sidebar.code(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        st.sidebar.text("User:")
        st.sidebar.code("kaxm23")

    def upload_file(self):
        """Handle file upload."""
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to translate"
        )
        
        if uploaded_file:
            with st.spinner("Uploading file..."):
                try:
                    files = {'file': uploaded_file}
                    response = requests.post(
                        f"{self.api_url}/translate/",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.task_id = data['task_id']
                        st.success("File uploaded successfully!")
                        return True
                        
                except requests.RequestException as e:
                    st.error(f"Error uploading file: {str(e)}")
                    
        return False

    def check_translation_status(self):
        """Check translation progress."""
        if not st.session_state.task_id:
            return False
            
        try:
            response = requests.get(
                f"{self.api_url}/status/{st.session_state.task_id}"
            )
            
            if response.status_code == 200:
                status_data = response.json()
                progress = status_data['progress']
                status = status_data['status']
                message = status_data['message']
                
                # Show progress bar
                progress_bar = st.progress(progress)
                st.text(message)
                
                if status == 'completed':
                    progress_bar.progress(1.0)
                    st.session_state.translation_complete = True
                    return True
                elif status == 'failed':
                    st.error(f"Translation failed: {status_data.get('error', 'Unknown error')}")
                    return False
                    
                time.sleep(1)  # Prevent too frequent updates
                
        except requests.RequestException as e:
            st.error(f"Error checking status: {str(e)}")
            return False
            
        return False

    def show_translation_results(self):
        """Display translation results."""
        if not st.session_state.translation_complete:
            return
            
        try:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs([
                "Original Text",
                "Translated Text",
                "Download Options"
            ])
            
            # Original Text Tab
            with tab1:
                if st.session_state.extracted_text:
                    st.markdown("### Original Text")
                    st.text_area(
                        "Extracted text",
                        st.session_state.extracted_text,
                        height=300,
                        disabled=True
                    )
                    
            # Translated Text Tab
            with tab2:
                if st.session_state.translated_text:
                    st.markdown("### Translated Text")
                    st.text_area(
                        "Arabic translation",
                        st.session_state.translated_text,
                        height=300,
                        disabled=True
                    )
                    
            # Download Options Tab
            with tab3:
                st.markdown("### Download Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download PDF"):
                        self.download_file('pdf')
                        
                with col2:
                    if st.button("Download DOCX"):
                        self.download_file('docx')
                        
                with col3:
                    if st.button("Download TXT"):
                        self.download_file('txt')
                        
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")

    def download_file(self, format_type: str):
        """Generate download link for translated file."""
        try:
            response = requests.get(
                f"{self.api_url}/download/{st.session_state.task_id}",
                params={'format': format_type}
            )
            
            if response.status_code == 200:
                # Create download button
                filename = f"translated_document.{format_type}"
                b64 = base64.b64encode(response.content).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" ' \
                       f'download="{filename}">Click to download {format_type.upper()}</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except requests.RequestException as e:
            st.error(f"Error downloading file: {str(e)}")

    def show_statistics(self):
        """Display translation statistics."""
        if st.session_state.translation_complete:
            try:
                response = requests.get(
                    f"{self.api_url}/stats/{st.session_state.task_id}"
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    st.sidebar.markdown("### Translation Statistics")
                    st.sidebar.markdown(f"""
                    - Processing Time: {stats['processing_time']:.2f}s
                    - Characters Translated: {stats['characters_translated']}
                    - Words Translated: {stats['words_translated']}
                    - Pages Processed: {stats['pages_processed']}
                    """)
                    
            except requests.RequestException:
                pass

    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        
        # File upload section
        if not st.session_state.task_id:
            if self.upload_file():
                st.experimental_rerun()
                
        # Translation progress section
        if st.session_state.task_id and not st.session_state.translation_complete:
            if self.check_translation_status():
                st.experimental_rerun()
                
        # Results section
        if st.session_state.translation_complete:
            self.show_translation_results()
            self.show_statistics()
            
        # Reset button
        if st.sidebar.button("Start New Translation"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == "__main__":
    app = PDFTranslatorApp()
    app.run()