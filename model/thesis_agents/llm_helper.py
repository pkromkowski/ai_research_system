import os
import logging
import importlib
from anthropic import Anthropic
from typing import Any, Dict, Optional

from model.core.config import AppConfig
from model.prompts.thesis_validation_prompts import INVESTMENT_ANALYST_SYSTEM_PROMPT as BASE_INVESTMENT_ANALYST_PROMPT

logger = logging.getLogger(__name__)


class LLMHelperMixin:
    """Mixin providing LLM call/parse patterns for thesis validation agents.

    Centralizes structured and text-based LLM calls with system prompt resolution,
    JSON parsing, and client management.
    """
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS: int = 1000
    DEFAULT_TEMPERATURE: float = 0.0
    
    def _resolve_system_prompt(self, system: Optional[str]) -> Optional[str]:
        """Resolve system prompt from explicit arg, module-level constant, or base prompt."""
        base = BASE_INVESTMENT_ANALYST_PROMPT
        SYSTEM_PROMPT_MAX_LEN = 400
        try:
            if system is not None:
                return f"{base}\n\n{system}" if system.strip() else base
            module = importlib.import_module(self.__class__.__module__)
            module_prompt = getattr(module, "INVESTMENT_ANALYST_SYSTEM_PROMPT", None)
            if module_prompt:
                if len(module_prompt.strip()) <= SYSTEM_PROMPT_MAX_LEN:
                    return f"{base}\n\n{module_prompt}"
                else:
                    logger.debug("Module-level INVESTMENT_ANALYST_SYSTEM_PROMPT is long; using compact base prompt to avoid token bloat")
                    return base
            return base
        except Exception:
            return base

    def _record_intermediate(self, prompt: str, raw_text: str, parsed: Optional[str] = None) -> None:
        """Store LLM prompt/response on agent's intermediate_outputs if available."""
        io = getattr(self, "intermediate_outputs", None) or getattr(self, "_intermediate_outputs", None) or getattr(self, "_intermediate_outputs_data", None)
        if io is not None:
            try:
                setattr(io, "last_llm_prompt", prompt)
                setattr(io, "last_llm_response", raw_text)
                if parsed is not None:
                    setattr(io, "last_llm_parsed_json_text", parsed)
            except Exception:
                logger.debug("Failed to write intermediate outputs for LLM call")

    def _resolve_model_params(self, max_tokens: Optional[int], temperature: Optional[float]):
        """Resolve model parameters from agent defaults or AppConfig fallback."""
        model = getattr(self, "DEFAULT_MODEL", None)
        max_toks = max_tokens if max_tokens is not None else getattr(self, "DEFAULT_MAX_TOKENS", None)
        temp = temperature if temperature is not None else getattr(self, "DEFAULT_TEMPERATURE", None)
        claude_cfg = AppConfig.claude
        if model is None:
            model = getattr(claude_cfg, "model", None)
        if max_toks is None:
            max_toks = getattr(claude_cfg, "max_tokens", None)
        if temp is None:
            temp = getattr(claude_cfg, "temperature", None)
        return model, max_toks, temp

    def format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with the provided keyword arguments."""
        return template.format(**kwargs)

    def _call_llm_structured(
        self,
        prompt: str,
        schema: Dict,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Dict:
        """Call LLM with structured output schema and return tool input block."""
        system_prompt = self._resolve_system_prompt(system)
        model, max_tokens, temperature = self._resolve_model_params(max_tokens, temperature)
        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            tools=[schema],
            tool_choice={"type": "tool", "name": schema["name"]}
        )
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        response = self.client.messages.create(**kwargs)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == schema["name"]:
                self._record_intermediate(prompt, repr(response))
                return block.input
        raise ValueError(f"Expected tool_use response, got: {response.content}")
    
    @property
    def client(self) -> Anthropic:
        """Get or initialize Anthropic client from instance, _client, or API key."""
        if "client" in getattr(self, "__dict__", {}):
            return self.__dict__["client"]
        if getattr(self, "_client", None) is not None:
            return self._client
        api_key = getattr(self, "_api_key", None) or getattr(self, "anthropic_api_key", None) or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required for LLM calls or provide a client via set_client")
        self._client = Anthropic(api_key=api_key)
        return self._client

    def set_client(self, client: Anthropic) -> None:
        """Inject or replace Anthropic client instance."""
        self.__dict__["client"] = client
        self._client = client