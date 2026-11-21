# STYLE.md: Channel Visual & Audio Identity Specification

> **Version**: 2.0.0  
> **Target Audience**: US Gaming & GTA Short-Form Audiences (YouTube Shorts, TikTok, Instagram Reels)  
> **Rendering Engine**: CPU-Friendly Production Pipeline (FFmpeg filtergraphs, libass subtitles, HTML/CSS Canvas; strictly **NO Three.js, NO WebGL shaders, NO Blender**).

---

## 1. Canvas & Technical Foundation

| Parameter | Specification | Purpose / Guardrail |
|---|---|---|
| **Canvas Size** | $1080 \times 1920$ pixels | Native 9:16 vertical standard for mobile full-screen feeds |
| **Framerate** | $30\text{ fps}$ (or $60\text{ fps}$) | Smooth motion without CPU rendering bottleneck |
| **Video Codec** | `libx264` | Broadest device compatibility, zero decode lag |
| **Color Space** | `yuv420p`, BT.709 standard | Prevents washed-out colors on mobile OLED displays |
| **Encoding Quality** | `-crf 18`, `-preset medium` | Visually lossless mastering with sharp edges |
| **Audio Standard** | AAC stereo, $48\text{ kHz}$, $192\text{ kbps}$ | YouTube Shorts & TikTok pristine audio compliance |
| **Target Loudness** | EBU R128 standard: $-14\text{ LUFS}$ ($\pm 1.0$), $\text{TP} \le -1.5\text{ dBFS}$ | Broadcast loudness matching YouTube Shorts platform normalizer |

---

## 2. Spatial Canvas & Vertical Safe Zones

The $1080 \times 1920$ canvas is strictly partitioned into three designated vertical zones to guarantee that motion graphics, titles, and subtitles **never obscure the primary gameplay, HUD prompts, or character action**.

```
+-------------------------------------------------------------+ 0px (Top)
|                                                             |
|   TOP ZONE (Top 26% / 0px - 500px): Clean Graphics Area     |
|   -------------------------------------------------------   |
|   [Top Safe Zone: 9% to 22% from top (172px - 422px)]       |
|     * Line 1: Hook Kicker ("HOW DID THIS HAPPEN?!")         |
|     * Line 2: Main Clip Title ("GTA 6 JET SKI PHYSICS?! 🔥")|
|                                                             |
+-------------------------------------------------------------+ 500px (26%)
|                                                             |
|   MIDDLE ZONE (46% / 500px - 1380px): Protected Gameplay    |
|   -------------------------------------------------------   |
|   * Centered Wide 16:9 Gameplay Box (1080x608 px)           |
|   * Vertically Centered at Y: 656px - 1264px                |
|   * Background: Full-Height Blurred & Darkened Copy         |
|     (blur: 18px / boxblur=10:2, brightness: 0.62)           |
|   * Gameplay action is 100% UNBLOCKED                       |
|                                                             |
+-------------------------------------------------------------+ 1380px (72%)
|                                                             |
|   BOTTOM ZONE (Bottom 28% / 1380px - 1920px): Motion Zone   |
|   -------------------------------------------------------   |
|   [Subtitle Safe Zone: 70% to 82% from top (1344px - 1574px)]|
|     * Animated Hybrid Neon Word-Pop Subtitles               |
|     * 2 to 4 words per burst                                |
|   [Channel UI Buffer: 82% to 100% (1574px - 1920px)]        |
|     * Clean buffer for platform UI (Like, Comment, Share)   |
|                                                             |
+-------------------------------------------------------------+ 1920px (Bottom)
```

### Background Blur & Darkening Recipe
- **Filtergraph Concept**: The raw 16:9 gameplay stream is duplicated into two layers:
  1. **Background Layer**: Scaled to cover the full $1080 \times 1920$ canvas, blurred softly (`boxblur=10:2` or CSS `blur(18px)`), and darkened (`eq=brightness=-0.12:contrast=0.95` or CSS `brightness(0.62)`).
  2. **Foreground Layer**: Scaled to $1080\text{px}$ width ($1080 \times 608\text{px}$) with crisp native sharpness, centered vertically at `Y = (1920 - 608) / 2 = 656px`.
  3. **Zero-Occlusion Guarantee**: Top $26\%$ ($500\text{px}$) and Bottom $28\%$ ($540\text{px}$) are entirely free from gameplay pixels, providing a pristine canvas for typography and visual punch elements.

---

## 3. Top Hook & Title Stack Specification

Located exclusively inside the **Top Safe Zone** ($9\%$ to $22\%$ from top, coordinates $Y = 172\text{px}$ to $422\text{px}$).

### Architecture: Two-Tier Stack
```
+--------------------------------------------------------------+
| [HOOK KICKER]  HOW DID THIS HAPPEN?! 🔥                      |  <- Line 1: Glowing Yellow/Red
| [MAIN TITLE]   GTA 6 JET SKI PHYSICS IS BROKEN               |  <- Line 2: Crisp White Pill
+--------------------------------------------------------------+
```

1. **Line 1: Hook Kicker**
   - **Position**: $Y \approx 200\text{px}$, centered horizontally.
   - **Typography**: 
     - Font: `Impact`, `Montserrat Black`, `Arial Black`, sans-serif.
     - Case: STRICT UPPERCASE.
     - Size: $46\text{px} - 52\text{px}$.
     - Color: Glowing Gold/Yellow (`#FFD500`) or Glowing Crimson (`#FF2E2E`).
     - Text Shadow / Glow: `0px 0px 14px rgba(255, 213, 0, 0.85)` with a $4\text{px}$ black stroke (`#000000`).
   - **Content**: Short urgent retention phrases (2 to 4 words):
     - `"HOW DID THIS HAPPEN?!"`
     - `"WAIT FOR THE END! 💀"`
     - `"HE ACTUALLY DID IT?!"`
     - `"NO WAY THIS JUST HAPPENED!"`
     - `"THE COPS DID WHAT?! 🚓"`

2. **Line 2: Main Clip Title**
   - **Position**: $Y \approx 280\text{px} - 320\text{px}$, centered horizontally.
   - **Typography**:
     - Font: `Arial Black`, `Inter`, `Montserrat ExtraBold`, sans-serif.
     - Case: STRICT UPPERCASE.
     - Size: $62\text{px} - 72\text{px}$.
     - Color: Crisp Pure White (`#FFFFFF`).
     - Backing: Subtle semi-transparent dark rounded pill container (`background: rgba(12, 14, 20, 0.72); border-radius: 18px; padding: 10px 28px; border: 1.5px solid rgba(255, 255, 255, 0.15)`).
   - **Content**: Punchy 4-to-7 word curiosity headline with exactly 1 relevant emoji (e.g., `"GTA 6 JET SKI PHYSICS?! 🔥"`).

3. **Duration & Lifecycle**:
   - Appears at $0.0\text{s}$ with a micro scale pop-in ($0.85 \to 1.0$ over $0.15\text{s}$).
   - Remains on screen for the first $3.5\text{s} - 4.5\text{s}$, accompanied by an opening `whoosh` SFX at $0.05\text{s}$.
   - Fades out cleanly (`fad(150, 250)`) right as the setup transitions into primary action escalation.

---

## 4. Bottom Animated Subtitles (Hybrid Neon Word-Pop)

Located in the **Bottom Safe Zone** ($70\%$ to $82\%$ from top, $Y \approx 1344\text{px} - 1574\text{px}$), leaving the bottom $18\%$ free from UI obstruction.

### Subtitle Rules
- **Burst Pacing**: **2 to 4 words per burst** (never long sentences, never full paragraphs).
- **Karaoke Pop Animation**: Active spoken word pops in scale ($1.0 \to 1.15$) on the exact spoken syllable.
- **Color Coding Palette**:
  - **Inactive Spoken Words**: Bold crisp white (`#FFFFFF`) with a heavy $4\text{px}$ black outline (`#000000`) and subtle drop-shadow (`2px 3px 6px rgba(0,0,0,0.9)`).
  - **Active Spoken Word**: Mood-based glowing accent:
    - 🔴 **`danger` / `crazy` / `panic` / `damage`**: Neon Crimson (`#FF2E2E`), glow `0px 0px 16px rgba(255, 46, 46, 0.9)`.
    - 🟡 **`hype` / `key` / `money` / `action`**: Neon Gold (`#FFD500`), glow `0px 0px 16px rgba(255, 213, 0, 0.9)`.
    - 🟢 **`win` / `calm` / `success` / `escape`**: Neon Emerald (`#00FF88`), glow `0px 0px 16px rgba(0, 255, 136, 0.9)`.
- **Text Styling (.ass Script Equivalent)**:
  ```ini
  Style: Default,Arial Black,74,&H00FFFFFF,&H0000FFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,1,8,4,2,60,60,380,1
  ```

---

## 5. Balanced Mood Panels & Visual Punches

Mood panels inject dopamine and guide viewer reactions at peak moments. To maintain studio quality, they must follow rigorous visual and auditory restraint.

### The 3 Canonical Mood Types

#### A. `laugh` Mood (Humor / Ridiculous NPC / Creator Laugh)
- **Visual Element**: Sleek dark-glass pop-up badge (`background: rgba(18, 18, 24, 0.78); backdrop-filter: blur(12px); border-radius: 24px; border: 1.5px solid rgba(255, 213, 0, 0.3)`).
- **Icon / Badge**: Bouncy 😂 emoji with subtle bold text (`"BRO WHAT?!"` or `"NO WAY 😂"`).
- **Animation**: Gentle elastic pop ($0.8 \to 1.05 \to 1.0$ over $0.25\text{s}$), duration $1.2\text{s} - 1.8\text{s}$.
- **Screen Filter**: **STRICTLY NO heavy screen flashes** (preserves clear visual of the funny moment).
- **Audio Cue**: Crisp, pleasant `pop.wav` SFX at $-16\text{ dBFS}$.

#### B. `sigma_crazy` Mood (High Skill / Audacious Stunt / Outrageous Escape)
- **Visual Element**: Compact sleek dark badge with 💀 skull icon + glowing red/white border.
- **Visual Punch**:
  - Brief **$0.35\text{s}$ dark high-contrast flash**: Temporary saturation boost & contrast punch (`eq=contrast=1.35:brightness=-0.10:saturation=1.25`).
  - Subtle **zoom-punch**: Camera punches in smoothly ($1.0 \to 1.07$ scale) and glides back over $0.4\text{s}$.
- **Audio Cue**: Low sub-bass `boom.wav` SFX at $-15\text{ dBFS}$ with subtle low-frequency punch.

#### C. `shock_impact` Mood (Massive Crash / Explosive Fail / Jump Scare)
- **Visual Element**: 🤯 shock icon badge with pulsating orange/red outline.
- **Visual Punch**:
  - Brief **$0.20\text{s}$ red edge vignette flash**: Red border flash (`drawbox` or vignette overlay) highlighting impact.
  - **Micro camera shake**: Pseudo-shake jitter ($\pm 6\text{px}$ random translational offset over 6 frames).
  - Optional subtle `pixelfx` spark or impact burst particle overlay.
- **Audio Cue**: Heavy impact `hit.wav` or `boom.wav` SFX layered at $-14\text{ dBFS}$.

---

## 6. Editorial Restraint Guardrails (Anti-Spam Rules)

To ensure the final video feels like an edited, high-retention production rather than chaotic low-effort spam:

1. **Mood Panel Frequency Limit**:
   - **Maximum 1 mood panel every $6.0$ to $8.0$ seconds**.
   - A typical $30\text{s}$ clip must contain **no more than 2 or 3 mood panels total**.
2. **SFX Punctuation Limit**:
   - **Maximum 4 total SFX events per 30-second clip**:
     - SFX 1: Opening hook `whoosh` at $0.05\text{s}$ (introduces hook banner).
     - SFX 2: Mid-clip escalation cue (`pop` or `whoosh`).
     - SFX 3: Climax / impact moment (`boom` or `hit`).
     - SFX 4: (Optional) Outro reaction punchline (`pop`).
3. **Audio Balancing Standards**:
   - **Creator Commentary**: Reference voice level normalized to $-16\text{ LUFS}$ to $-14\text{ LUFS}$.
   - **Background Music (BGM)**: Set to $-22\text{ dBFS}$ baseline, automatically ducked via `sidechaincompress` to $-28\text{ dBFS}$ whenever the creator speaks.
   - **Micro SFX**: Kept subtly in the mix at $-16\text{ dBFS}$ to $-14\text{ dBFS}$ with limiter protection so audio never distorts or clips.

---

## 7. Configuration Preset Schema (`STYLE.json`)

```json
{
  "canvas": {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "codec": "libx264",
    "crf": 18,
    "preset": "medium"
  },
  "layout": {
    "type": "blur_wide",
    "background_blur": 18,
    "background_brightness": 0.62,
    "top_safe_zone_pct": 26,
    "bottom_safe_zone_pct": 28
  },
  "hook_header": {
    "line1_color": "#FFD500",
    "line2_color": "#FFFFFF",
    "pill_bg": "rgba(12, 14, 20, 0.72)",
    "duration_sec": 3.8
  },
  "subtitles": {
    "preset": "hybrid_neon_pop",
    "words_per_burst": 3,
    "active_scale": 1.15,
    "color_danger": "#FF2E2E",
    "color_hype": "#FFD500",
    "color_win": "#00FF88"
  },
  "restraint": {
    "min_mood_interval_sec": 6.5,
    "max_sfx_per_30s": 4,
    "target_lufs": -14.0
  }
}
```
