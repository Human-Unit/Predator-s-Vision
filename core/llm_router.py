"""
core/llm_router.py
==================
LLM interface for the Yautja-Vision Bio-Mask HUD.

Responsibilities:
  - Owns the system prompt that defines the routing contract with the LLM.
  - Wraps the OpenAI-compatible client with deterministic (temperature=0) calls.
  - Parses, validates, and returns structured action payloads as Python dicts.
  - Raises typed exceptions on invalid JSON or unsupported action identifiers
    so callers can trigger the telemetry-corruption error state cleanly.
"""

import json
from openai import OpenAI

from config.settings import API_KEY, BASE_URL, MODEL, SUPPORTED_MODES


# ---------------------------------------------------------------------------
# System Prompt — the contract between the visor and the LLM
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""
You are the telemetry routing sub-system of a Yautja (Predator) Bio-Mask.
Your job is to translate a user's natural language command into a JSON command block.

Rules:
1. Output ONLY a valid JSON object. Do NOT wrap it in markdown code blocks.
2. The JSON must have exactly two keys: "action" and "parameters".
3. "action" must be exactly one of: {SUPPORTED_MODES}
4. "parameters" must be a dict with values matching the schemas below.

Action Schemas:
  THERMAL_VISION  → {{}}
  TACTICAL_ZOOM   → {{"scale": <float 1.5–5.0, default 2.0>}}
  TARGET_HUD      → {{}}
  CLOAK_BLUR      → {{"strength": <odd int 5–99, default 25>}}
  SPECTRUM_SHIFT  → {{"shift_type": <"invert"|"red_only"|"green_only"|"blue_only">}}
  NORMAL_VISION   → {{}}

Examples:
  "activate infrared heat vision"      → {{"action": "THERMAL_VISION",  "parameters": {{}}}}
  "zoom in 3.5x on the target"         → {{"action": "TACTICAL_ZOOM",   "parameters": {{"scale": 3.5}}}}
  "engage optical cloaking distortion" → {{"action": "CLOAK_BLUR",      "parameters": {{"strength": 25}}}}
  "switch to ultraviolet spectrum"     → {{"action": "SPECTRUM_SHIFT",  "parameters": {{"shift_type": "blue_only"}}}}
  "reset visor to normal"              → {{"action": "NORMAL_VISION",   "parameters": {{}}}}
"""


# ---------------------------------------------------------------------------
# Router Class
# ---------------------------------------------------------------------------
class YautjaRouter:
    """
    Thin, stateless wrapper around an OpenAI-compatible chat endpoint.

    Usage::

        router = YautjaRouter()
        result = router.route_command("activate heat vision")
        # → {"action": "THERMAL_VISION", "parameters": {}}

    Raises:
        ValueError  — LLM returned non-JSON text or an unknown action identifier.
        KeyError    — Valid JSON but missing "action" / "parameters" keys.
    """

    def __init__(
        self,
        api_key: str  = API_KEY,
        base_url: str = BASE_URL,
        model: str    = MODEL,
    ) -> None:
        # Local inference servers (LM Studio, Ollama) don't require a real key.
        _key = api_key if api_key != "YOUR_API_KEY_HERE" else "local-dummy"
        self.model  = model
        self.client = OpenAI(api_key=_key, base_url=base_url)

    # ------------------------------------------------------------------
    def route_command(self, user_text: str) -> dict:
        """
        Send *user_text* to the LLM and return a validated action payload.

        The LLM call uses temperature=0 to guarantee deterministic routing —
        the same natural-language phrase always maps to the same JSON action.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,          # deterministic classifier output
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences (``` or ```json … ```)
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
            raw = "\n".join(lines).strip()

        # --- JSON parse ---
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {raw!r}") from exc

        # --- Schema validation ---
        if "action" not in payload or "parameters" not in payload:
            raise KeyError(f"Payload missing 'action' or 'parameters': {payload}")

        if payload["action"] not in SUPPORTED_MODES:
            raise ValueError(
                f"Unknown action '{payload['action']}'. "
                f"Supported: {SUPPORTED_MODES}"
            )

        return payload
