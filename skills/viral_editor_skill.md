# Viral Short-Form Retention Guidelines for Gaming & GTA Shorts

## Core Philosophy
In vertical short-form gaming feeds (YouTube Shorts, TikTok, Instagram Reels), viewer drop-off happens within the first 1.5 seconds. Retention curves reward videos that hook immediately, sustain narrative tension, and deliver a definitive emotional or comedic punchline before looping.

---

## 1. The 0-3s Hook Rule
- **Immediate Pattern Interrupt**: Every clip must begin right before an immediate visual or auditory pattern interrupt (e.g., intense shout, loud crash, bizarre glitch, controversial statement, hilarious NPC dialogue, sudden combat).
- **Zero Dead Air**: NEVER start a clip during silent walking, driving with no dialogue, inventory management, or ambient background silence.
- **Natural Verbal Lead-In**: Start the clip 0.25s before the speaker begins their thought—never mid-syllable and never mid-word.

---

## 2. The Story Arc Rule (Setup -> Escalation -> Payoff)
Every viral Short must follow a self-contained 3-act micro-structure:
1. **Setup (0s - 6s)**: Establish the premise or challenge immediately ("Watch what happens when I mess with this cop...").
2. **Escalation (6s - 35s)**: Complications multiply, stakes escalate, music or energy intensifies (e.g., 3-star wanted level turns into 5-star chaos, car starts smoking, hilarious dialogue exchanges).
3. **Payoff / Climax (Final 5s - 10s)**: The unexpected conclusion (e.g., spectacular crash, bizarre physics bug, narrow escape, punchline reaction).
- **Duration Constraint**: Clips must be naturally sized between **18 and 55 seconds** based on the natural completion of the scene.
- **FORBIDDEN**: Never slice arbitrary fixed-length windows (e.g. exactly 30s or 35s). The clip must end cleanly when the punchline or action concludes.

---

## 3. Hook Banner Rule (Curiosity Headline)
- Generate a punchy **4-to-7 word curiosity headline** designed to appear as an on-screen title banner during the first 3 seconds of the clip.
- **Formulas**:
  - *Curiosity Question*: "GTA 6 NPCs React Like Real People?!"
  - *Disbelief / Warning*: "HE DID NOT EXPECT THIS ENDING 💀"
  - *Challenge / Stake*: "Surviving 5 Stars in GTA 6!"
  - *Spectacle*: "This New GTA 6 Car Physics Is Crazy"
- Keep it concise (4 to 7 words), readable at a glance on mobile screens, and complementary to the creator's spoken hook.

---

## 4. Audio Cue Planning (Micro SFX)
Audio sound effects enhance immersion and guide viewer focus without becoming noisy or distracting.
- Plan up to **3 subtle SFX events** per clip (max 3 so the edit never feels spammy or over-edited).
- **Supported SFX Types**:
  - `whoosh`: Quick camera pans, fast cuts, vehicle fly-bys, or when the 0-3s hook banner transitions in/out.
  - `pop`: Visual text pop-ins, item discoveries, sudden minimap pings, or meme sound elements.
  - `boom`: Massive explosions, heavy car collisions, sudden dramatic realization, or bass-drop impact moments.
- **Timestamping**: Timestamps (`time_sec`) must be specified **relative to the clip's start** (0.0s = start of clip).

---

## 5. Structured Output Schema
Output must strictly adhere to the following JSON array format:
```json
[
  {
    "start_sec": 14.5,
    "end_sec": 48.2,
    "hook_banner": "GTA 6 NPCs React Like Real People?!",
    "title": "Insane NPC Dialogue in GTA 6 #Shorts #GTA6",
    "viral_score": 94,
    "reason": "Immediate high-energy hook with an escalating police encounter and a hilarious ragdoll physics payoff.",
    "sfx_events": [
      {"time_sec": 0.2, "type": "whoosh"},
      {"time_sec": 12.4, "type": "pop"},
      {"time_sec": 31.8, "type": "boom"}
    ]
  }
]
```
