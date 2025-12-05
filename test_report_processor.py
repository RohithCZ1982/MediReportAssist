"""
Test Report Processor
Specialized processor for medical test reports (lab results, imaging, diagnostics)
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LabResult:
    """Represents a single lab test result"""
    test_name: str
    value: Optional[str]
    unit: Optional[str]
    reference_range: Optional[str]
    status: Optional[str]  # Normal, High, Low, Abnormal, etc.
    date: Optional[str]

@dataclass
class ImagingResult:
    """Represents an imaging test result"""
    test_type: str  # X-ray, CT, MRI, Ultrasound, etc.
    body_part: Optional[str]
    findings: str
    impression: Optional[str]
    recommendations: Optional[str]
    date: Optional[str]

class TestReportProcessor:
    """Processes medical test reports and extracts structured information"""
    
    def __init__(self):
        # Common lab test patterns
        self.lab_patterns = {
            'cbc': ['complete blood count', 'cbc', 'hemoglobin', 'hematocrit', 'wbc', 'rbc', 'platelet'],
            'metabolic': ['glucose', 'creatinine', 'bun', 'sodium', 'potassium', 'chloride', 'co2', 'calcium'],
            'lipid': ['cholesterol', 'triglyceride', 'hdl', 'ldl', 'lipid panel'],
            'liver': ['alt', 'ast', 'alkaline phosphatase', 'bilirubin', 'albumin', 'liver function'],
            'thyroid': ['tsh', 't3', 't4', 'thyroid', 'free t4'],
            'cardiac': ['troponin', 'ck-mb', 'bnp', 'nt-probnp', 'cardiac enzymes'],
            'infection': ['covid', 'flu', 'strep', 'culture', 'sensitivity'],
            'urine': ['urinalysis', 'urine', 'protein', 'glucose urine', 'ketones'],
            'vitamin': ['vitamin d', 'b12', 'folate', 'vitamin'],
            'hormone': ['testosterone', 'estrogen', 'progesterone', 'cortisol']
        }
        
        # Imaging test patterns
        self.imaging_patterns = {
            'xray': ['x-ray', 'xray', 'radiograph', 'chest x-ray'],
            'ct': ['ct scan', 'computed tomography', 'cat scan'],
            'mri': ['mri', 'magnetic resonance', 'magnetic resonance imaging'],
            'ultrasound': ['ultrasound', 'sonogram', 'echocardiogram', 'echo'],
            'mammogram': ['mammogram', 'mammography'],
            'bone_scan': ['bone scan', 'bone density'],
            'pet': ['pet scan', 'pet-ct', 'positron emission']
        }
        
        # Status indicators
        self.status_keywords = {
            'normal': ['normal', 'within normal limits', 'wnl', 'negative', 'nl'],
            'high': ['high', 'elevated', 'increased', 'above normal', '↑', '>'],
            'low': ['low', 'decreased', 'below normal', '↓', '<'],
            'abnormal': ['abnormal', 'abnl', 'positive', 'detected', 'present']
        }
    
    def detect_report_type(self, text: str) -> str:
        """
        Detect the type of test report
        
        Returns:
            'lab', 'imaging', 'mixed', or 'unknown'
        """
        text_lower = text.lower()
        
        has_lab = any(
            any(pattern in text_lower for pattern in patterns)
            for patterns in self.lab_patterns.values()
        )
        
        has_imaging = any(
            any(pattern in text_lower for pattern in patterns)
            for patterns in self.imaging_patterns.values()
        )
        
        if has_lab and has_imaging:
            return 'mixed'
        elif has_lab:
            return 'lab'
        elif has_imaging:
            return 'imaging'
        else:
            return 'unknown'
    
    def extract_lab_results(self, text: str) -> List[LabResult]:
        """
        Extract lab test results from text
        
        Args:
            text: Text content of the test report
            
        Returns:
            List of LabResult objects
        """
        results = []
        lines = text.split('\n')
        
        # Pattern to match: Test Name | Value | Unit | Reference Range | Status
        # Common formats:
        # "Glucose: 95 mg/dL (70-100) Normal"
        # "Hemoglobin    12.5    g/dL    12.0-16.0    Normal"
        # "WBC 7.2 x10^3/uL (4.0-11.0) Normal"
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Try to extract test result
            result = self._parse_lab_line(line)
            if result:
                results.append(result)
        
        return results
    
    def _parse_lab_line(self, line: str) -> Optional[LabResult]:
        """Parse a single line for lab result"""
        
        # Pattern 1: "Test Name: Value Unit (Range) Status"
        pattern1 = r'([A-Za-z\s/]+(?:\([^)]+\))?)\s*:?\s*([\d\.]+)\s*([a-zA-Z/%\^]+)?\s*\(?([^)]+)\)?\s*([A-Za-z\s]+)?'
        
        # Pattern 2: Tab/space separated columns
        # "Test Name    Value    Unit    Range    Status"
        pattern2 = r'([A-Za-z\s/]+(?:\([^)]+\))?)\s+([\d\.]+)\s+([a-zA-Z/%\^]+)?\s+([^\s]+)?\s+([A-Za-z\s]+)?'
        
        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                test_name = match.group(1).strip()
                value = match.group(2).strip() if match.lastindex >= 2 else None
                unit = match.group(3).strip() if match.lastindex >= 3 and match.group(3) else None
                ref_range = match.group(4).strip() if match.lastindex >= 4 and match.group(4) else None
                status = match.group(5).strip() if match.lastindex >= 5 and match.group(5) else None
                
                # Clean up test name
                test_name = re.sub(r'\s+', ' ', test_name)
                
                # Determine status if not explicitly stated
                if not status and value and ref_range:
                    status = self._determine_status(value, ref_range)
                
                return LabResult(
                    test_name=test_name,
                    value=value,
                    unit=unit,
                    reference_range=ref_range,
                    status=status,
                    date=None
                )
        
        return None
    
    def _determine_status(self, value: str, ref_range: str) -> Optional[str]:
        """Determine if value is normal, high, or low based on reference range"""
        try:
            value_num = float(value)
            
            # Extract range (e.g., "70-100" or "12.0-16.0")
            range_match = re.search(r'([\d\.]+)\s*-\s*([\d\.]+)', ref_range)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                
                if low <= value_num <= high:
                    return 'Normal'
                elif value_num > high:
                    return 'High'
                else:
                    return 'Low'
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def extract_imaging_results(self, text: str) -> List[ImagingResult]:
        """
        Extract imaging test results from text
        
        Args:
            text: Text content of the imaging report
            
        Returns:
            List of ImagingResult objects
        """
        results = []
        
        # Find sections: Technique, Findings, Impression, Recommendations
        findings_match = re.search(r'(?:FINDINGS|Findings|RESULTS|Results)[:\s]*(.+?)(?=(?:IMPRESSION|Impression|CONCLUSION|Conclusion)|$)', 
                                  text, re.IGNORECASE | re.DOTALL)
        impression_match = re.search(r'(?:IMPRESSION|Impression|CONCLUSION|Conclusion)[:\s]*(.+?)(?=(?:RECOMMENDATIONS|Recommendations)|$)', 
                                     text, re.IGNORECASE | re.DOTALL)
        recommendations_match = re.search(r'(?:RECOMMENDATIONS|Recommendations)[:\s]*(.+?)$', 
                                          text, re.IGNORECASE | re.DOTALL)
        
        findings = findings_match.group(1).strip() if findings_match else ""
        impression = impression_match.group(1).strip() if impression_match else None
        recommendations = recommendations_match.group(1).strip() if recommendations_match else None
        
        # Detect test type
        test_type = self._detect_imaging_type(text)
        
        # Detect body part
        body_part = self._detect_body_part(text)
        
        # Extract date
        date = self._extract_date(text)
        
        if findings or impression:
            results.append(ImagingResult(
                test_type=test_type,
                body_part=body_part,
                findings=findings,
                impression=impression,
                recommendations=recommendations,
                date=date
            ))
        
        return results
    
    def _detect_imaging_type(self, text: str) -> str:
        """Detect the type of imaging test"""
        text_lower = text.lower()
        
        for img_type, patterns in self.imaging_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return img_type.replace('_', ' ').title()
        
        return "Imaging Study"
    
    def _detect_body_part(self, text: str) -> Optional[str]:
        """Detect the body part being imaged"""
        body_parts = [
            'chest', 'abdomen', 'pelvis', 'head', 'brain', 'spine', 'neck',
            'knee', 'shoulder', 'hip', 'wrist', 'ankle', 'foot', 'hand',
            'heart', 'lung', 'liver', 'kidney', 'pancreas', 'spleen'
        ]
        
        text_lower = text.lower()
        for part in body_parts:
            if part in text_lower:
                return part.title()
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}'  # January 1, 2024
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def format_lab_summary(self, results: List[LabResult]) -> str:
        """Format lab results into a readable summary"""
        if not results:
            return "No lab results found."
        
        summary = "LAB TEST RESULTS:\n\n"
        
        for result in results:
            summary += f"• {result.test_name}"
            if result.value:
                summary += f": {result.value}"
            if result.unit:
                summary += f" {result.unit}"
            if result.reference_range:
                summary += f" (Reference: {result.reference_range})"
            if result.status:
                status_emoji = "✅" if result.status.lower() == "normal" else "⚠️"
                summary += f" - {status_emoji} {result.status}"
            summary += "\n"
        
        return summary
    
    def format_imaging_summary(self, results: List[ImagingResult]) -> str:
        """Format imaging results into a readable summary"""
        if not results:
            return "No imaging results found."
        
        summary = "IMAGING TEST RESULTS:\n\n"
        
        for result in results:
            summary += f"Test Type: {result.test_type}\n"
            if result.body_part:
                summary += f"Body Part: {result.body_part}\n"
            if result.date:
                summary += f"Date: {result.date}\n"
            summary += f"\nFindings:\n{result.findings}\n"
            if result.impression:
                summary += f"\nImpression:\n{result.impression}\n"
            if result.recommendations:
                summary += f"\nRecommendations:\n{result.recommendations}\n"
            summary += "\n" + "-"*50 + "\n\n"
        
        return summary
    
    def process_test_report(self, text: str) -> Dict:
        """
        Process a test report and extract all information
        
        Returns:
            Dictionary with report type, lab results, imaging results, and formatted summaries
        """
        report_type = self.detect_report_type(text)
        
        lab_results = []
        imaging_results = []
        
        if report_type in ['lab', 'mixed']:
            lab_results = self.extract_lab_results(text)
        
        if report_type in ['imaging', 'mixed']:
            imaging_results = self.extract_imaging_results(text)
        
        # Create formatted summaries
        lab_summary = self.format_lab_summary(lab_results) if lab_results else ""
        imaging_summary = self.format_imaging_summary(imaging_results) if imaging_results else ""
        
        return {
            'report_type': report_type,
            'lab_results': lab_results,
            'imaging_results': imaging_results,
            'lab_summary': lab_summary,
            'imaging_summary': imaging_summary,
            'has_abnormal_results': any(
                r.status and r.status.lower() not in ['normal', 'negative', 'nl', 'wnl']
                for r in lab_results
            ) or any(
                'abnormal' in r.findings.lower() or 'abnormal' in (r.impression or '').lower()
                for r in imaging_results
            )
        }

