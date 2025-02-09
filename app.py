    def add_to_history(self,
                      source_text: str,
                      translated_text: str,
                      source_lang: str,
                      target_lang: str,
                      document_type: str,
                      result: Dict):
        """Add translation to history."""
        st.session_state.history.append({
            'timestamp': "2025-02-09 09:05:24",
            'source_text': source_text[:100] + "..." if len(source_text) > 100 else source_text,
            'translated_text': translated_text[:100] + "..." if len(translated_text) > 100 else translated_text,
            'source_lang': self.languages[source_lang],
            'target_lang': self.languages[target_lang],
            'document_type': self.document_types[document_type]['name'],
            'tokens': result['tokens'],
            'cost': result['cost'],
            'user': "kaxm23"
        })

    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file."""
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.txt':
                # Handle text files
                text = uploaded_file.getvalue().decode('utf-8')
                
            elif file_extension == '.pdf':
                # Handle PDF files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                text = self.extract_text_from_pdf(tmp_file_path)
                Path(tmp_file_path).unlink()  # Clean up temp file
                
            elif file_extension in ['.doc', '.docx']:
                # Handle Word documents
                import docx2txt
                text = docx2txt.process(uploaded_file)
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Failed to extract text from file: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            text = []
            
            for page in doc:
                text.append(page.get_text())
            
            return "\n".join(text).strip()
            
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {str(e)}")
            return ""

    def download_translation(self, translated_text: str):
        """Create downloadable file with translation."""
        try:
            # Create filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"translation_{timestamp}.txt"
            
            # Add metadata
            metadata = (
                f"Translation Details:\n"
                f"Date: 2025-02-09 09:05:24\n"
                f"Document Type: {self.document_types[st.session_state.doc_type_select]['name']}\n"
                f"Source Language: {self.languages[st.session_state.source_lang]}\n"
                f"Target Language: {self.languages[st.session_state.target_lang]}\n"
                f"Translated by: kaxm23\n"
                f"\n{'='*50}\n\n"
            )
            
            # Combine metadata and translation
            file_content = metadata + translated_text
            
            # Create download button
            st.download_button(
                "Download Translation",
                file_content,
                filename,
                "text/plain",
                key="download_button"
            )
            
        except Exception as e:
            st.error(f"Failed to create download file: {str(e)}")

    def export_history(self, format: str = 'csv'):
        """Export translation history."""
        try:
            if not st.session_state.history:
                st.warning("No translation history to export")
                return
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            if format == 'csv':
                # Export as CSV
                df = pd.DataFrame(st.session_state.history)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    "Download CSV",
                    csv,
                    f"translation_history_{timestamp}.csv",
                    "text/csv"
                )
                
            elif format == 'json':
                # Export as JSON
                json_str = json.dumps(
                    st.session_state.history,
                    indent=2,
                    ensure_ascii=False
                )
                
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"translation_history_{timestamp}.json",
                    "application/json"
                )
                
        except Exception as e:
            st.error(f"Failed to export history: {str(e)}")

    def display_help(self):
        """Display help information."""
        with st.expander("ℹ️ Help & Information"):
            st.markdown("""
                ### Document Types
                
                #### 🏛️ Legal Translation
                - Contracts, regulations, and legal documents
                - Maintains legal terminology and formal tone
                - Preserves citations and formatting
                
                #### 🏥 Medical Translation
                - Clinical records and medical reports
                - Accurate medical terminology
                - Handles abbreviations and technical terms
                
                #### 📚 Academic Translation
                - Research papers and scholarly articles
                - Technical precision
                - Citation preservation
                
                #### 📝 General Translation
                - Business and marketing content
                - Natural language flow
                - Cultural adaptation
                
                ### Tips
                1. Choose the appropriate document type
                2. Provide context in Advanced Settings
                3. Review the translation confidence score
                4. Use the history feature for reference
                
                ### Support
                For assistance, contact: support@example.com
            """)

if __name__ == "__main__":
    app = TranslationUI()
    app.run()