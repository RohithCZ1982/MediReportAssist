# Training Guide for Medical Test Reports

This guide explains how to train and optimize your system to read and understand medical test reports (lab results, imaging studies, diagnostic tests).

## What Are Medical Test Reports?

Medical test reports include:
- **Lab Results**: Blood tests, urine tests, cultures, etc. with values and reference ranges
- **Imaging Studies**: X-rays, CT scans, MRIs, ultrasounds with findings and impressions
- **Diagnostic Tests**: EKGs, stress tests, biopsies, etc.

## System Enhancements

Your system has been enhanced with:

1. **Test Report Processor** (`test_report_processor.py`)
   - Automatically detects test report types
   - Extracts structured data (lab values, imaging findings)
   - Formats results for better understanding

2. **Enhanced Document Processing**
   - Automatically detects and enhances test reports
   - Adds structured summaries to improve retrieval

3. **Updated Query Manager**
   - Recognizes test report-related queries
   - Understands lab result and imaging terminology

4. **Improved RAG Prompts**
   - Better explanations of test results
   - Patient-friendly interpretation of medical terminology

## How It Works

### Automatic Detection

When you upload a document, the system:
1. Extracts text from the document
2. Detects if it's a test report (lab, imaging, or mixed)
3. Extracts structured information:
   - Lab: Test names, values, units, reference ranges, status
   - Imaging: Test type, body part, findings, impression, recommendations
4. Enhances the text with formatted summaries
5. Stores everything in the vector database

### Query Examples

You can now ask questions like:

**Lab Results:**
- "What are my lab results?"
- "Is my glucose level normal?"
- "What does my hemoglobin value mean?"
- "Are there any abnormal results?"
- "What is my cholesterol level?"

**Imaging:**
- "What did my X-ray show?"
- "What are the findings from my CT scan?"
- "What does my MRI report say?"
- "Are there any abnormalities in my imaging?"

**General:**
- "Summarize my test results"
- "What tests were performed?"
- "Do I need to follow up on any results?"

## Training Options

### Option 1: Improve Test Report Detection (Quick)

The system automatically detects test reports, but you can improve accuracy by:

1. **Adding more test patterns** in `test_report_processor.py`:

```python
# In TestReportProcessor.__init__()
self.lab_patterns = {
    # Add your specific lab tests
    'custom': ['your test name', 'another test'],
    # ... existing patterns
}
```

2. **Improving extraction patterns** for your specific report format

### Option 2: Fine-tune Embeddings for Test Reports (Recommended)

Train the embedding model on test report data to improve retrieval:

1. **Create training data** from your test reports:

```python
# create_test_report_training_data.py
from test_report_processor import TestReportProcessor
from pathlib import Path
import json

processor = TestReportProcessor()
training_data = []

for file_path in Path("uploads").glob("*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process the report
    processed = processor.process_test_report(text)
    
    # Create Q&A pairs
    if processed['lab_results']:
        for result in processed['lab_results']:
            question = f"What is my {result.test_name} result?"
            answer = f"Your {result.test_name} is {result.value} {result.unit or ''}. "
            if result.reference_range:
                answer += f"Normal range: {result.reference_range}. "
            if result.status:
                answer += f"Status: {result.status}."
            
            training_data.append({
                "instruction": "Answer questions about lab test results",
                "input": question,
                "output": answer
            })
    
    # Similar for imaging results...

# Save training data
with open("test_report_training.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
```

2. **Fine-tune the embedding model** (see `TRAINING_GUIDE.md` Option 2)

### Option 3: Create Custom Ollama Model for Test Reports

Create a specialized model for test report interpretation:

1. **Create a Modelfile for test reports**:

```dockerfile
FROM llama3.2

SYSTEM """You are a specialized medical assistant focused on interpreting test reports for patients.

Your expertise includes:
1. Explaining lab test results in simple, patient-friendly language
2. Interpreting imaging study findings
3. Comparing test values to reference ranges
4. Identifying normal vs. abnormal results
5. Explaining what test results mean for patient health
6. Recommending when to follow up with healthcare providers

Guidelines:
- Always explain test names in simple terms (e.g., "hemoglobin" = "protein in red blood cells that carries oxygen")
- Compare values to reference ranges and explain if they're normal, high, or low
- Use visual indicators: ✅ for normal, ⚠️ for abnormal
- Explain what abnormal results might indicate (but never diagnose)
- Always recommend consulting healthcare provider for abnormal results
- Use patient-friendly language instead of medical jargon
- Format results clearly with bullet points or tables when helpful

When explaining:
- Lab values: Include the number, unit, reference range, and what it means
- Imaging findings: Describe in simple terms, explain significance
- Test status: Clearly state if results are normal, abnormal, or need attention
"""
```

2. **Create the model**:

```bash
ollama create test-report-assistant -f Modelfile
```

3. **Use it**:

```bash
export LLM_MODEL=test-report-assistant
```

### Option 4: Improve PDF Extraction for Test Reports

Test reports in PDFs often have tables that need special handling:

1. **Install better PDF library**:

```bash
pip install pdfplumber  # Better table extraction
```

2. **Update document processor**:

```python
# In document_processor.py
import pdfplumber

def _process_pdf(self, file_path: Path) -> str:
    """Extract text from PDF with table support"""
    text_content = []
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                text_content.append(text)
            
            # Extract tables (important for lab results)
            tables = page.extract_tables()
            for table in tables:
                # Convert table to readable format
                table_text = self._format_table(table)
                text_content.append(table_text)
    
    return '\n\n'.join(text_content)

def _format_table(self, table):
    """Format table as text"""
    formatted = []
    for row in table:
        if row:
            formatted.append(' | '.join(str(cell) if cell else '' for cell in row))
    return '\n'.join(formatted)
```

## Testing Your Training

### Test with Sample Queries

```python
# test_test_reports.py
from rag_system import RAGSystem
from test_report_processor import TestReportProcessor

rag = RAGSystem()
processor = TestReportProcessor()

# Upload a test report first, then test queries
test_queries = [
    "What are my lab results?",
    "Is my glucose normal?",
    "What did my X-ray show?",
    "Are there any abnormal results?",
    "What does my hemoglobin value mean?",
]

# Test each query
for query in test_queries:
    context = rag.retrieve_context("your_doc_id", query)
    answer = rag.generate_answer(query, context)
    print(f"Q: {query}")
    print(f"A: {answer}\n")
```

## Common Test Report Formats

### Lab Report Format

```
Test Name          Value    Unit    Reference Range    Status
Hemoglobin         12.5     g/dL    12.0-16.0          Normal
Glucose            95       mg/dL   70-100             Normal
Cholesterol        220      mg/dL   <200               High ⚠️
```

### Imaging Report Format

```
STUDY: Chest X-ray
DATE: 01/15/2024

FINDINGS:
The lungs are clear bilaterally. No acute cardiopulmonary process.

IMPRESSION:
Normal chest X-ray.

RECOMMENDATIONS:
None.
```

## Best Practices

1. **Upload Clean Documents**: Ensure PDFs are text-based (not scanned images)
2. **Use Structured Formats**: Well-formatted reports are easier to parse
3. **Train on Your Data**: Use your actual test reports for training
4. **Iterate on Prompts**: Adjust prompts based on user feedback
5. **Monitor Accuracy**: Test with real queries and improve

## Troubleshooting

### Problem: Test results not being extracted

**Solution**: 
- Check if the report format matches expected patterns
- Add custom patterns in `test_report_processor.py`
- Improve PDF extraction (use pdfplumber for tables)

### Problem: Values not being recognized

**Solution**:
- Update regex patterns in `_parse_lab_line()` method
- Handle different number formats (commas, decimals, etc.)

### Problem: Imaging findings not extracted

**Solution**:
- Check if report has standard sections (FINDINGS, IMPRESSION)
- Update section detection patterns
- Handle variations in section names

## Next Steps

1. ✅ Upload test reports to see automatic detection
2. ✅ Test with sample queries
3. ⏭️ Fine-tune embeddings if you have many reports
4. ⏭️ Create custom Ollama model for better interpretation
5. ⏭️ Improve PDF extraction if needed

## Resources

- **Lab Test Reference**: https://labtestsonline.org/
- **Medical Terminology**: https://www.medicinenet.com/script/main/hp.asp
- **Test Report Examples**: Check your healthcare provider's portal

---

**Note**: This system helps patients understand their test reports but should never replace professional medical interpretation. Always encourage patients to discuss results with their healthcare providers.

