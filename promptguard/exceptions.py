"""Custom exceptions for PromptGuard."""


class PromptGuardError(Exception):
    """Base exception for PromptGuard."""

    pass


class BundleError(PromptGuardError):
    """Error loading or validating a bundle."""

    pass


class BundleNotFoundError(BundleError):
    """Bundle file not found."""

    pass


class BundleValidationError(BundleError):
    """Bundle configuration is invalid."""

    pass


class DatasetError(PromptGuardError):
    """Error loading or parsing dataset."""

    pass


class ProviderError(PromptGuardError):
    """Error with LLM provider."""

    pass


class ProviderConfigError(ProviderError):
    """Provider configuration error (e.g., missing API key)."""

    pass


class ProviderAPIError(ProviderError):
    """Error calling provider API."""

    pass


class ContractError(PromptGuardError):
    """Error evaluating output contract."""

    pass


class SchemaValidationError(ContractError):
    """Output failed schema validation."""

    pass


class InvariantError(ContractError):
    """Output failed invariant check."""

    pass


class ThresholdError(PromptGuardError):
    """Threshold evaluation failed."""

    pass
