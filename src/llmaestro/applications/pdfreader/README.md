# PDFReader

Extract structured data from PDF documents using vision capabilities and LLMs.

## Features

- Process single and multi-page PDFs
- Extract data according to custom schemas
- Confidence scoring for extractions
- Warning system for ambiguous content
- Customizable result combination logic

## Installation

The PDFReader requires `poppler` for PDF processing:

```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils

# Windows
# Download and install from: http://blog.alivate.com.au/poppler-windows/
```

## Usage

1. Define your output schema:
```python
from pydantic import BaseModel
from datetime import datetime

class InvoiceData(BaseModel):
    invoice_number: str
    date: datetime
    total: float
    items: Dict[str, float]
```

2. Create a reader instance:
```python
from llmaestro.applications.pdfreader import PDFReader

reader = PDFReader(
    output_model=InvoiceData,
    api_key="your-api-key",  # Optional, defaults to config.yaml
)
```

3. Process PDFs:
```python
result = await reader.process_pdf("path/to/invoice.pdf")

print(f"Extracted data: {result.extracted_data}")
print(f"Confidence: {result.confidence}")
if result.warnings:
    print(f"Warnings: {result.warnings}")
```

## Configuration

The PDFReader can be configured in several ways:

1. Using config.yaml:
```yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-latest
  api_key: your-api-key
```

2. Using environment variables:
```bash
export ANTHROPIC_API_KEY=your-api-key
```

3. Direct initialization:
```python
reader = PDFReader(
    output_model=YourModel,
    api_key="your-api-key",
    config_path="path/to/config.yaml"
)
```

## Customization

### Custom Result Combination

For multi-page PDFs, you can customize how results are combined:

```python
class CustomPDFReader(PDFReader):
    def _combine_results(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom logic for combining results from multiple pages."""
        combined_data = {}
        max_confidence = 0.0
        warnings = []

        for result in results:
            # Merge data from each page
            combined_data.update(result["extracted_data"])
            max_confidence = max(max_confidence, result["confidence"])
            if result.get("warnings"):
                warnings.extend(result["warnings"])

        return {
            "extracted_data": combined_data,
            "confidence": max_confidence,
            "warnings": warnings if warnings else None
        }
```

### Custom Prompt

You can modify the prompt template in `prompt.yaml` to:
- Add specific instructions
- Change the system behavior
- Customize response format
- Add example interactions

## Error Handling

The PDFReader handles several types of errors:

1. File errors:
```python
try:
    result = await reader.process_pdf("nonexistent.pdf")
except FileNotFoundError:
    print("PDF file not found")
```

2. Processing errors:
```python
try:
    result = await reader.process_pdf("document.pdf")
except ValueError as e:
    print(f"Processing error: {e}")
```

3. Validation errors:
```python
try:
    result = await reader.process_pdf("document.pdf")
except ValidationError as e:
    print(f"Schema validation failed: {e}")
```

## Testing

Run the tests with:
```bash
# All tests
pytest tests/test_integration/test_invoice_processor.py

# Only integration tests
pytest tests/test_integration/test_invoice_processor.py -m integration

# With real LLM
pytest tests/test_integration/test_invoice_processor.py --use-llm-tokens
```
