"""
Test script for test report processing
Use this to test if your test reports are being processed correctly
"""

from test_report_processor import TestReportProcessor
from document_processor import DocumentProcessor
from pathlib import Path

def test_test_report_processor():
    """Test the test report processor with sample data"""
    
    processor = TestReportProcessor()
    
    # Sample lab report text
    sample_lab_report = """
    LABORATORY REPORT
    Date: 01/15/2024
    
    Test Name          Value    Unit    Reference Range    Status
    Hemoglobin         12.5     g/dL    12.0-16.0          Normal
    Glucose            95       mg/dL   70-100             Normal
    Cholesterol        220      mg/dL   <200               High
    WBC                7.2      x10^3/uL 4.0-11.0         Normal
    Creatinine         1.2      mg/dL   0.6-1.2           Normal
    """
    
    print("=" * 60)
    print("Testing Lab Report Processing")
    print("=" * 60)
    
    # Detect report type
    report_type = processor.detect_report_type(sample_lab_report)
    print(f"\nDetected Report Type: {report_type}")
    
    # Process the report
    processed = processor.process_test_report(sample_lab_report)
    
    print(f"\nReport Type: {processed['report_type']}")
    print(f"Has Abnormal Results: {processed['has_abnormal_results']}")
    print(f"\nNumber of Lab Results: {len(processed['lab_results'])}")
    
    # Display lab results
    print("\n" + processed['lab_summary'])
    
    # Sample imaging report
    sample_imaging_report = """
    RADIOLOGY REPORT
    Date: 01/15/2024
    Study: Chest X-ray
    
    FINDINGS:
    The lungs are clear bilaterally with no acute cardiopulmonary process.
    The heart size is within normal limits. No pleural effusion or pneumothorax.
    
    IMPRESSION:
    Normal chest X-ray. No acute abnormalities.
    
    RECOMMENDATIONS:
    None. Routine follow-up as clinically indicated.
    """
    
    print("\n" + "=" * 60)
    print("Testing Imaging Report Processing")
    print("=" * 60)
    
    # Process imaging report
    processed_img = processor.process_test_report(sample_imaging_report)
    
    print(f"\nReport Type: {processed_img['report_type']}")
    print(f"Number of Imaging Results: {len(processed_img['imaging_results'])}")
    
    # Display imaging results
    print("\n" + processed_img['imaging_summary'])

def test_with_uploaded_files():
    """Test with actual uploaded files"""
    
    processor = TestReportProcessor()
    doc_processor = DocumentProcessor()
    
    upload_dir = Path("uploads")
    
    if not upload_dir.exists():
        print("No uploads directory found. Upload some test reports first.")
        return
    
    print("=" * 60)
    print("Testing with Uploaded Files")
    print("=" * 60)
    
    for file_path in upload_dir.glob("*"):
        if file_path.suffix.lower() in ['.txt', '.pdf', '.docx']:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"{'='*60}")
            
            try:
                # Process document
                text = doc_processor.process_document(str(file_path))
                
                # Detect report type
                report_type = processor.detect_report_type(text)
                print(f"Detected Type: {report_type}")
                
                if report_type != 'unknown':
                    # Process test report
                    processed = processor.process_test_report(text)
                    
                    print(f"Report Type: {processed['report_type']}")
                    print(f"Lab Results: {len(processed['lab_results'])}")
                    print(f"Imaging Results: {len(processed['imaging_results'])}")
                    print(f"Has Abnormal: {processed['has_abnormal_results']}")
                    
                    if processed['lab_summary']:
                        print("\nLab Summary:")
                        print(processed['lab_summary'][:500] + "...")
                    
                    if processed['imaging_summary']:
                        print("\nImaging Summary:")
                        print(processed['imaging_summary'][:500] + "...")
                else:
                    print("Not detected as a test report (might be a discharge summary)")
                    
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    print("\n🧪 Test Report Processor Test Suite\n")
    
    # Test with sample data
    test_test_report_processor()
    
    print("\n" + "=" * 60)
    print("\n")
    
    # Test with uploaded files
    test_with_uploaded_files()
    
    print("\n" + "=" * 60)
    print("\n✅ Testing Complete!")
    print("\nNext steps:")
    print("1. Upload test reports via the web interface")
    print("2. Ask questions like 'What are my lab results?'")
    print("3. Check if the system correctly interprets test reports")

