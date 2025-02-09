from enhanced_pdf_extractor import EnhancedPDFExtractor
import logging

def main():
    # Initialize the enhanced extractor
    extractor = EnhancedPDFExtractor(
        tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Adjust path for your system
        ocr_lang='eng',  # Use 'eng+ara' for English and Arabic
        dpi=300,
        log_level=logging.INFO
    )
    
    try:
        # Extract text from a PDF file
        result = extractor.extract_text(
            pdf_path="example.pdf",
            pages=None,  # Extract all pages
            include_metadata=True,
            force_ocr=False  # Set to True to force OCR on all pages
        )
        
        # Save the extracted text
        if result['status'] == 'success':
            extractor.save_extracted_text(
                extracted_data=result,
                output_path="extracted_text.txt",
                include_processing_info=True
            )
            
            # Print processing statistics
            info = result['processing_info']
            print(f"\nProcessing Complete:")
            print(f"Total pages: {info['total_pages']}")
            print(f"OCR applied to {info['ocr_pages']} pages")
            print(f"Direct text extraction from {info['text_pages']} pages")
            print(f"Failed pages: {info['failed_pages']}")
            print(f"Processing time: {info['processing_time']:.2f} seconds")
            
        else:
            print(f"Error occurred: {result['error']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()