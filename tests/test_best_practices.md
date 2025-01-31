# Test Best Practices for LLM Orchestrator

## Core Principles

1. **Use pytest Fixtures Over unittest-style Setup**
   ```python
   # ✅ Good - pytest fixture
   @pytest.fixture
   def mock_llm_client():
       return MockLLMClient()

   # ❌ Avoid - unittest style
   def setUp(self):
       self.mock_llm = MockLLMClient()
   ```

2. **Prefer pytest.mark for Test Organization**
   ```python
   @pytest.mark.integration
   @pytest.mark.slow
   def test_complex_chain():
       pass
   ```

## Mocking Best Practices

1. **Use pytest-style Monkeypatch Over unittest.mock**
   ```python
   # ✅ Good
   def test_api_call(monkeypatch):
       monkeypatch.setattr("module.api_function", lambda x: "mocked_response")

   # ❌ Avoid
   @mock.patch("module.api_function")
   def test_api_call(mock_api):
       mock_api.return_value = "mocked_response"
   ```

2. **Use Fixtures for Complex Mock Objects**
   ```python
   @pytest.fixture
   def mock_anthropic_response():
       return {
           "completion": "test response",
           "model": "claude-3",
           "usage": {"prompt_tokens": 10, "completion_tokens": 20}
       }
   ```

## Async Testing

1. **Use pytest-asyncio for Async Tests**
   ```python
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result == expected_value
   ```

2. **Handle Async Fixtures**
   ```python
   @pytest.fixture
   async def async_client():
       client = AsyncClient()
       yield client
       await client.close()
   ```

## Test Data Management

1. **Use Fixtures for Test Data**
   ```python
   @pytest.fixture
   def sample_prompt_data():
       return {
           "name": "test_prompt",
           "description": "Test prompt for unit tests",
           "system_prompt": "You are a test assistant",
           "user_prompt": "Hello {name}",
           "metadata": {...}
       }
   ```

2. **Use YAML Files for Complex Test Data**
   ```python
   @pytest.fixture
   def model_config():
       with open("src/llm/models/claude.yaml") as f:
           return yaml.safe_load(f)
   ```

## Test Structure

1. **Follow Arrange-Act-Assert Pattern**
   ```python
   def test_prompt_rendering():
       # Arrange
       prompt = BasePrompt(...)
       variables = {"name": "test"}

       # Act
       system, user = prompt.render(**variables)

       # Assert
       assert "test" in user
   ```

2. **Use Descriptive Test Names**
   ```python
   # ✅ Good
   def test_prompt_renders_with_valid_variables():
       pass

   # ❌ Avoid
   def test_prompt_1():
       pass
   ```

## Error Testing

1. **Test Exception Cases with pytest.raises**
   ```python
   def test_invalid_prompt_variables():
       with pytest.raises(ValueError) as exc_info:
           prompt.render()
       assert "Missing required variables" in str(exc_info.value)
   ```

## Test Categories

1. **Unit Tests**
   - Test individual components in isolation
   - Use appropriate markers: `@pytest.mark.unit`
   - Keep fast and focused

2. **Integration Tests**
   - Test component interactions
   - Use `@pytest.mark.integration`
   - Can be slower than unit tests

3. **End-to-End Tests**
   - Test complete workflows
   - Use `@pytest.mark.e2e`
   - Consider running in separate CI/CD stages

## Performance Considerations

1. **Use Parametrized Tests**
   ```python
   @pytest.mark.parametrize("input,expected", [
       ("test1", "result1"),
       ("test2", "result2"),
   ])
   def test_multiple_scenarios(input, expected):
       assert process(input) == expected
   ```

2. **Skip Slow Tests When Appropriate**
   ```python
   @pytest.mark.slow
   def test_time_consuming_operation():
       pass
   ```

## Project-Specific Guidelines

1. **LLM Response Testing**
   - Mock LLM API calls in unit tests
   - Use recorded responses for integration tests
   - Consider token usage in test scenarios

2. **Prompt Testing**
   - Test template variable validation
   - Verify prompt rendering
   - Test version control features

3. **Rate Limiter Testing**
   - Test rate limiting behavior
   - Mock time in tests
   - Verify concurrent request handling

## Running Tests

1. **Command Line Options**
   ```bash
   # Run all tests
   pytest

   # Run specific test categories
   pytest -m unit
   pytest -m integration
   pytest -m e2e

   # Run with coverage
   pytest --cov=src
   ```

2. **Configuration**
   - Use `pyproject.toml` for pytest configuration
   - Define custom markers
   - Set up coverage settings

## Test Coverage

1. **Coverage Goals**
   - Aim for high coverage of core functionality
   - Focus on critical paths
   - Don't pursue 100% coverage at the expense of test quality

2. **Coverage Reporting**
   ```bash
   pytest --cov=src --cov-report=html
   ```

## Common Pitfalls to Avoid

1. ❌ Using unittest-style assertions
2. ❌ Creating complex inheritance hierarchies in test classes
3. ❌ Using mutable fixtures without proper cleanup
4. ❌ Writing tests that depend on execution order
5. ❌ Over-mocking or creating brittle tests

## Additional Resources

1. [Pytest Documentation](https://docs.pytest.org/)
2. [Pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
3. [Python Testing with pytest (Book)](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)
