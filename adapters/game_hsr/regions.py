"""HSR UI geometry, in 1280x720 framebuffer space.

Coordinates were read off real captured frames (see data/replays/prologue-01 and
combat-01), not guessed. Every region here is something the perception layer
actually crops; if a region is wrong the features go wrong, so keep them honest
and re-check them when the client version changes.

All tuples are (x, y, w, h).
"""

from __future__ import annotations

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- dialogue / cutscene ------------------------------------------------------

# The subtitle band: dark translucent strip carrying the spoken line.
DIALOGUE_BAND = (150, 520, 980, 190)

# Where the subtitle glyphs actually sit (inside the band, below the name).
DIALOGUE_TEXT = (250, 630, 780, 70)

# Speaker name plate, above the subtitle text.
SPEAKER_NAME = (180, 555, 400, 45)

# The advance/continue chevron at the bottom centre.
CONTINUE_CHEVRON = (620, 655, 45, 50)

# Auto / skip / log controls, top right.
DIALOGUE_TOP_CONTROLS = (940, 20, 310, 60)

# --- combat ------------------------------------------------------------------

# Circular action button (Basic ATK / Single Target), bottom right.
ACTION_BUTTON = (950, 480, 180, 180)

# Skill-point pips and the "Blast" secondary action, left of the action button.
SKILL_POINTS = (830, 600, 190, 60)
BLAST_BUTTON = (1120, 480, 120, 110)

# Party portraits + HP bars, top left.
PARTY_PANEL = (0, 0, 200, 340)

# Enemy HP bars and weakness icons, above the enemies.
ENEMY_HP = (600, 160, 420, 90)

# Speed / auto / pause, top right.
COMBAT_TOP_CONTROLS = (940, 20, 330, 60)

# Wave counter, top left.
WAVE_COUNTER = (40, 8, 90, 40)

# Tutorial banner ("Use Basic ATK on selected enemy"). It MOVES between tutorial
# steps: y~290 next to the target reticle, y~558 beside the action button. This
# box covers both; anything drawn inside it is prompt text.
TUTORIAL_BANNER = (400, 250, 560, 400)

# Movement HUD: joystick (bottom left) and the sprint/attack cluster (bottom
# right). Present while the player controls the character, absent during a
# blocking dialogue -- which is how a bark is told apart from a real one.
HUD_ACTIONS = (990, 500, 270, 160)

# The interaction prompt: a dark rounded plate with a small glyph and a label, named
# after whatever the object is ("Investigate", "Controller", "Read"). The label is
# arbitrary, so detection must be structural, not a keyword list.
#
# It is drawn in two places, and both have been seen on real frames:
#   - to the right of the character, plate around x 770-950 at y ~400 (the "Controller"
#     fixture), and
#   - in the same band the conversation choices use, x 380-1180 y 250-630, which is
#     where the terminal prompt sits.
# The region is the union; the shape filters are what discriminate. The quest tracker is
# left of x=380, the party UI is above y=250, and a distance label like "1m" is too narrow
# to be a label.
INTERACT_PROMPT = (380, 250, 880, 340)

# --- shared ------------------------------------------------------------------

# Where a modal choice list would render (no confirmed example in the current
# corpus; treated as unvalidated until one is recorded).
CHOICE_LIST = (300, 240, 680, 380)

# Full-frame region used by features that consider everything.
FULL = (0, 0, FRAME_WIDTH, FRAME_HEIGHT)

FEATURE_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "dialogue_band": DIALOGUE_BAND,
    "dialogue_text": DIALOGUE_TEXT,
    "action_button": ACTION_BUTTON,
    "party_panel": PARTY_PANEL,
    "enemy_hp": ENEMY_HP,
    "hud_actions": HUD_ACTIONS,
    "interact_prompt": INTERACT_PROMPT,
}
