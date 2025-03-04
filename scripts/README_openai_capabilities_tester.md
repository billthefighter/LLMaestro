# OpenAI Model Capabilities Tester

This script tests OpenAI models for their capabilities and updates the models.py file with the results.

## Overview

The script:
1. Gets a list of OpenAI models
2. Scrapes pricing information from the OpenAI pricing page
3. For each model, attempts to make API calls that test various LLMCapabilities and VisionCapabilities
4. Updates/adds the results to the models.py file to ensure we have consistent properties that properly reflect the actual capabilities of each model

## Requirements

- Python 3.8+
- OpenAI Python SDK (v1.0.0+)
- Beautiful Soup 4 (for pricing scraping)
- HTTPX (for async HTTP requests)
- An OpenAI API key

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install openai beautifulsoup4 httpx
   ```

## Usage

### Basic Usage

```bash
# Test all available models and print results
python test_openai_capabilities.py --api-key YOUR_API_KEY

# Test specific models
python test_openai_capabilities.py --api-key YOUR_API_KEY --models gpt-4 gpt-3.5-turbo

# Test models and update the models.py file
python test_openai_capabilities.py --api-key YOUR_API_KEY --update-file

# Skip pricing data scraping
python test_openai_capabilities.py --api-key YOUR_API_KEY --skip-pricing
```

You can also set your API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key
python test_openai_capabilities.py
```

### Command Line Arguments

- `--api-key`: Your OpenAI API key (optional if OPENAI_API_KEY environment variable is set)
- `--models`: Specific models to test (optional, tests all available models if not specified)
- `--update-file`: Update the models.py file with the test results
- `--skip-pricing`: Skip fetching pricing data from the OpenAI website

## How It Works

### Capability Testing

The script tests the following capabilities for each model:

- Basic chat completion
- Context window size
- Streaming support
- Function calling
- Vision capabilities
- JSON mode
- System prompt support
- Tools support
- Frequency penalty
- Presence penalty
- Stop sequences

For models that support vision, it also tests vision-specific capabilities.

### Pricing Data Scraping

The script automatically scrapes the OpenAI pricing page (https://platform.openai.com/docs/pricing) to get the latest pricing information for each model. It:

1. Fetches the HTML content of the pricing page
2. Parses the tables to extract model names and their associated pricing
3. Normalizes model names to match the API model names
4. Extracts input and output costs per 1K tokens
5. Falls back to hardcoded values if scraping fails or for models not found on the pricing page

This ensures that the generated model definitions have up-to-date pricing information.

## Output

The script outputs the test results for each model in JSON format. If the `--update-file` flag is used, it also updates the models.py file with new model definitions.

Example output:

```
2023-07-01 12:34:56,789 - __main__ - INFO - Fetching pricing data from OpenAI pricing page...
2023-07-01 12:34:57,123 - __main__ - INFO - Found pricing for gpt-4: input=$0.03, output=$0.06 per 1K tokens
2023-07-01 12:34:57,456 - __main__ - INFO - Found pricing for gpt-3.5-turbo: input=$0.0015, output=$0.002 per 1K tokens
2023-07-01 12:34:58,789 - __main__ - INFO - Testing capabilities for model: gpt-4
2023-07-01 12:34:59,890 - __main__ - INFO - Results for gpt-4: {
  "capabilities": {
    "max_context_window": 8192,
    "supports_streaming": true,
    "supports_function_calling": true,
    "supports_vision": false,
    "input_cost_per_1k_tokens": 0.03,
    "output_cost_per_1k_tokens": 0.06,
    ...
  },
  "vision_capabilities": null,
  "created": 1687882410,
  "owned_by": "openai"
}
```

## Notes

- The script uses a simplified approach to determine the context window size based on model name patterns. For more accurate results, you may need to update the context_windows dictionary in the _test_context_window method.
- The script does not test all possible capabilities exhaustively. Some capabilities, like typical_speed, would require more extensive testing.
- The pricing scraper may need updates if OpenAI changes the structure of their pricing page.
- Fallback pricing is provided for common models in case the scraping fails or for models not found on the pricing page.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
