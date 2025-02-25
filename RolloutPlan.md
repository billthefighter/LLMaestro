# LLM State Management Refactoring Plan

## Overview
This refactoring aims to improve separation of concerns and reduce redundancy in the LLM state management system. The key changes involve separating pure state from lifecycle management and behavior.

## New Files

### `src/llmaestro/llm/state.py`
New module containing state management classes:
- `LLMRuntimeConfig`: Runtime configuration for LLM instances
- `LLMState`: Pure state container
- `LLMLifecycle`: Lifecycle information

## Modified Files

### `src/llmaestro/llm/llm_registry.py`
- Refactor `LLMRuntime` to use new state classes
- Remove direct state management
- Update initialization flow

### `src/llmaestro/llm/interfaces/base.py`
- Remove state management from `BaseLLMInterface`
- Use `LLMState` for configuration
- Simplify initialization

### `src/llmaestro/llm/interfaces/factory.py`
- Update factory methods to use `LLMState`
- Simplify interface creation

## Code to Remove

### In `BaseLLMInterface`:
```python
# Remove these fields (now in LLMState)
self.provider = provider
self.model = model
self.api_key = api_key
self.max_tokens = max_tokens
self.temperature = temperature
self.max_context_tokens = max_context_tokens
self.stream = stream
```

### In `LLMRuntime`:
```python
# Remove direct state fields (now in LLMState)
registered_at: datetime
provider: Provider
profile: LLMProfile
is_deprecated: bool
last_capability_update: Optional[datetime]
recommended_replacement: Optional[str]
initialized_at: Optional[datetime]
```

### In `Provider`:
```python
# Remove capability-related fields (now only in LLMProfile)
capabilities: ProviderCapabilities
capabilities_detector: Optional[Type[BaseCapabilityDetector]]
```

## Changes to __init__.py

### Add New Exports
```python
from .state import LLMState, LLMRuntimeConfig, LLMLifecycle

__all__ += [
    "LLMState",
    "LLMRuntimeConfig",
    "LLMLifecycle"
]
```

### Remove Obsolete Exports
```python
# Remove these as they're now internal implementation details
"OpenAILLMInterface",
"AnthropicLLMInterface",
"GoogleLLMInterface",
"HuggingFaceLLMInterface",
```

## Migration Steps

1. **Phase 1: New Implementation**
   - Create `state.py` with new classes
   - Update `LLMRuntime` to use new state structure
   - Add tests for new state classes

2. **Phase 2: Interface Updates**
   - Refactor `BaseLLMInterface` to use `LLMState`
   - Update factory methods
   - Add tests for new interface initialization

3. **Phase 3: Registry Updates**
   - Update `LLMRegistry` to manage `LLMState`
   - Add state validation
   - Update provider initialization

4. **Phase 4: Cleanup**
   - Remove deprecated code
   - Update documentation
   - Add migration guides

## Testing Strategy

1. **Unit Tests**
   - Test state class initialization
   - Test state validation
   - Test interface creation with new state
   - Test registry operations with new state

2. **Integration Tests**
   - Test complete initialization flow
   - Test provider operations
   - Test interface operations with state

3. **Migration Tests**
   - Test backward compatibility
   - Test state conversion
   - Test error handling

## Backward Compatibility

The refactoring will maintain backward compatibility through:
1. Factory methods that accept both old and new configurations
2. State conversion utilities
3. Deprecation warnings for old patterns

## Timeline

1. Week 1: Implementation of new state classes
2. Week 2: Interface and registry updates
3. Week 3: Testing and documentation
4. Week 4: Cleanup and migration support

## Risks and Mitigation

1. **Risk**: Breaking changes in interface initialization
   - **Mitigation**: Provide factory methods supporting both old and new patterns

2. **Risk**: State conversion errors
   - **Mitigation**: Add comprehensive validation and error handling

3. **Risk**: Performance impact
   - **Mitigation**: Benchmark before and after, optimize if needed

## Success Criteria

1. All tests passing
2. No redundant state management
3. Clear separation of concerns
4. Improved code maintainability
5. No performance regression 