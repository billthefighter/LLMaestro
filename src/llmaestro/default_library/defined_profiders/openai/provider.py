"""OpenAI provider configuration."""
from llmaestro.config.base import RateLimitConfig
from llmaestro.llm.capabilities import ProviderCapabilities
from llmaestro.llm.models import Provider

OPENAI_PROVIDER = Provider(
    family="openai",
    description="OpenAI API Provider",
    api_base="https://api.openai.com/v1",
    capabilities=ProviderCapabilities(
        # API Features
        supports_batch_requests=True,
        supports_async_requests=True,
        supports_streaming=True,
        supports_model_selection=True,
        supports_custom_models=False,
        # Authentication & Security
        supports_api_key_auth=True,
        supports_oauth=False,
        supports_organization_ids=True,
        supports_custom_endpoints=False,
        # Rate Limiting
        supports_concurrent_requests=True,
        max_concurrent_requests=50,
        requests_per_minute=3500,
        tokens_per_minute=180000,
        # Billing & Usage
        supports_usage_tracking=True,
        supports_cost_tracking=True,
        supports_quotas=True,
        # Advanced Features
        supports_fine_tuning=True,
        supports_model_deployment=False,
        supports_custom_domains=False,
        supports_audit_logs=True,
    ),
    rate_limits=RateLimitConfig(requests_per_minute=3500, tokens_per_minute=180000),
)
