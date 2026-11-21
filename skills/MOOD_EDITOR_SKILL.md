# MOOD_EDITOR_SKILL.md: English US Gaming Virality & Cut Intelligence

> **Role**: Senior Viral Short-Form Gaming Editor & Algorithm Strategist  
> **Target Audience**: US Gaming & GTA Audience (High Retention, Fast Pacing, Mobile First)  
> **Primary Objective**: Analyze long-form gameplay footage, creator commentary, and audio reaction peaks to slice viral short-form candidates ($18\text{s} - 48\text{s}$) with impeccable boundary snapping and psychological retention hooks.

---

## 1. Audience Psychology & Retention Mechanics

Short-form gaming viewers (YouTube Shorts, TikTok, Instagram Reels) make swipe decisions within **$1.2$ seconds**. To maximize completion rate and algorithmic boost:

1. **The 0-1.5s Immediate Pattern Interrupt**:
   - The opening frame MUST startle the viewer with motion, a vocal scream/gasp, an outrageous claim, or an escalating visual hazard.
   - **Banned Openings**: Never open with silence, walking on sidewalks, driving straight with no commentary, looking at pause menus, or checking inventories.

2. **The 3-Act Micro Story Arc**:
   Every clip must represent a complete emotional journey:
   - **Act I: The Hook & Premise ($0\text{s} - 6\text{s}$)**: Creator establishes the immediate high-stakes premise ("Wait till you see what this cop does...", "I bet \$1,000 I survive this jump...").
   - **Act II: The Escalation ($6\text{s} - 32\text{s}$)**: Stakes amplify, vehicles take damage, police sirens multiply, physics bug begins to warp, commentary rises in pitch.
   - **Act III: The Payoff / Punchline ($32\text{s} - 45\text{s}$)**: The catastrophic crash, successful stunt landing, hilarious NPC quote, or creator laughter.
   
3. **Loopability Incentive**:
   - End immediately after the payoff laughs or gasp so the video loops naturally before the viewer can swipe away.

---

## 2. Audio-Visual Cut Boundary Rules (Non-Negotiable)

Cutting at the wrong millisecond ruins immersion and retention. Follow strict boundary snapping:

### A. Start Boundary Snapping
- **Lead-In Margin**: Always begin cutting **$0.25\text{s}$ prior** to the first voiced syllable of the hook phrase or visual trigger.
- **NEVER Cut Mid-Sentence**: The creator's sentence must start at the subject or verb—never slice in the middle of a preposition or syllable.
- **Audio Clamping**: Ensure no residual word tail from the preceding sentence bleeds into the clip opening.

### B. End Boundary Snapping
- **Payoff Ring-Out Margin**: Always end cutting **$0.35\text{s}$ after** the final punchline word, laughter decay, or explosive sound effect.
- **No Premature Cuts**: Never cut while the creator is still talking, screaming, or delivering the punchline.
- **Natural Scene Exit**: The clip ends the exact moment tension resolves—zero lingering dead air.

### C. Duration Constraints
- **Minimum Duration**: **$18.0\text{ seconds}$** (shorter clips lack sufficient narrative arc to feel satisfying).
- **Maximum Duration**: **$48.0\text{ seconds}$** (longer clips suffer retention decay in US gaming feeds).
- **STRICT PROHIBITION**: **Never output arbitrary fixed-length slices** (e.g., exactly 30.0s or 35.0s). Every clip duration must be an organic reflection of that specific scene's story arc.

---

## 3. Language & Audience Localization

- **Audience**: Strictly English US Gaming / GTA communities.
- **Vocabulary**: Punchy, conversational US gaming idioms:
  - *"Physics engine is broken"*
  - *"NPC went full ragdoll"*
  - *"5-Star police chase"*
  - *"No way he survived that"*
  - *"He thought he could escape"*
  - *"Stunt jump gone wrong"*
- **Subtitles Localization**: Standardized English orthography. If Hinglish or regional slang is spoken by the creator, transcribe phonetically into crisp, universally intelligible Romanized gaming terms.

---

## 4. Mood Panel & Reaction Trigger Rules

Correlate audio loudness spikes (energy peaks $\ge 0.5$) and transcript sentiment to assign one of the 3 canonical moods:

| Mood Type | Audio / Visual Trigger in Footage | Associated Badge & Visual Punch | SFX Pairing |
|---|---|---|---|
| **`laugh`** | Creator bursts out laughing, funny NPC vocal bark, absurd ragdoll failure. | Sleek dark-glass badge with 😂 + gentle scale pop ($0.8 \to 1.05 \to 1.0$). **NO screen flashes.** | `pop.wav` |
| **`sigma_crazy`** | Outrageous clutch stunt, clean escape from 5 cops, audacity moment. | Dark 💀 skull badge + brief $0.35\text{s}$ dark high-contrast punch + subtle camera punch-in ($1.0 \to 1.07$). | Low sub `boom.wav` |
| **`shock_impact`** | High-speed vehicle collision, RPG blast, sudden unexpected jump-scare. | Pulsating 🤯 badge + brief $0.20\text{s}$ red edge vignette flash + micro-shake ($\pm 6\text{px}$). | Sharp `hit.wav` / `boom.wav` |

### Restraint Enforcement
- **Minimum Spacing**: At least $6.0$ to $8.0$ seconds between consecutive mood panels.
- **Per-Clip Maximum**: Max 2 to 3 mood panels and max 4 SFX events per 30-second clip.

---

## 5. Hook Kicker & Title Formulas

Every viral clip must be paired with a two-tier top header:

### Tier 1: Hook Kicker (2 to 4 words, urgent, uppercase)
- *"HOW IS THIS POSSIBLE?! 🔥"*
- *"WAIT FOR THE END! 💀"*
- *"HE ACTUALLY SURVIVED?! 🤯"*
- *"5 STARS IN 10 SECONDS! 🚓"*
- *"THE WORST CRASH EVER! 💥"*

### Tier 2: Main Clip Title (4 to 7 words, descriptive, 1 emoji)
- *"GTA 6 Jet Ski Physics Is Broken 🔥"*
- *"Surviving 5 Stars On A Highway 🚓"*
- *"Insane NPC Reaction In GTA 6 💀"*
- *"He Tried The Impossible Stunt Jump 🚀"*
- *"When Ragdoll Physics Goes Too Far 🤯"*

---

## 6. Structured Output Specification (JSON Schema)

AI analyzers processing transcripts and audio energy peaks must return valid, strictly formatted JSON adhering to this schema:

```json
[
  {
    "clip_id": "clip_01",
    "start_sec": 14.75,
    "end_sec": 44.85,
    "duration_sec": 30.1,
    "hook_header": {
      "kicker": "HOW IS THIS POSSIBLE?! 🔥",
      "main_title": "GTA 6 JET SKI PHYSICS IS BROKEN"
    },
    "viral_score": 96,
    "retention_rationale": "Starts immediately with a shocking boat glitch at high speed, escalates as creator roasts the physics, and ends cleanly after the explosion and laugh.",
    "mood_events": [
      {
        "timestamp_sec": 18.2,
        "mood_type": "sigma_crazy",
        "badge_text": "PURE CHAOS 💀",
        "visual_fx": "contrast_punch_zoom"
      },
      {
        "timestamp_sec": 28.5,
        "mood_type": "laugh",
        "badge_text": "BRO WHAT?! 😂",
        "visual_fx": "gentle_pop"
      }
    ],
    "sfx_events": [
      {"time_sec": 0.05, "sfx_name": "whoosh"},
      {"time_sec": 18.2, "sfx_name": "boom"},
      {"time_sec": 28.5, "sfx_name": "pop"}
    ]
  }
]
```
