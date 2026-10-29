"""Render a GameState into the compact text Mica sees.

A few hundred tokens per state, per the plan. The generic schema carries more
than a decision needs: ids, confidences and pixel boxes are for the actuator, not
for the model. So this drops geometry and keeps what a reader would need to know
where they are and what they can do.
"""

from __future__ import annotations

from core.state import GameState


def render_state_for_mica(state: GameState) -> str:
    lines: list[str] = [f"screen: {state.screen_type}"]

    if state.text_blocks:
        # One line of spoken text or a prompt; never dump OCR noise.
        primary = max(state.text_blocks, key=len)
        lines.append(f"text: {primary}")

    if state.choices:
        lines.append("choices:")
        lines.extend(f"  {i}. {choice}" for i, choice in enumerate(state.choices))

    buttons = [e for e in state.interactive_elements if e.kind in ("button", "choice")]
    if buttons:
        lines.append("actions:")
        for element in buttons:
            lines.append(f"  {element.id}: {element.label}")

    if state.numeric_indicators:
        indicators = ", ".join(f"{k}={v:g}" for k, v in sorted(state.numeric_indicators.items()))
        lines.append(f"indicators: {indicators}")

    return "\n".join(lines)
