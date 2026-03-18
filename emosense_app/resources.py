"""Emotion and intent aware resource recommendation engine for EmoSense."""

from __future__ import annotations


COMMON_HOTLINES = [
    {"name": "iCall (India)", "number": "9152987821", "hours": "Mon-Sat 8am-10pm"},
    {"name": "Vandrevala Foundation", "number": "1860-2662-345", "hours": "24x7"},
    {"name": "iCall WhatsApp", "number": "wa.me/919152987821", "hours": "Chat support"},
]


RESOURCE_BANK = {
    "neutral": {
        "breathing_exercises": [
            "Box Breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s for 5 rounds.",
            "Balanced Breath: Inhale 5s and exhale 5s for 2 minutes.",
        ],
        "grounding_exercises": [
            "5-4-3-2-1 Scan: Notice 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste.",
            "Foot Anchor: Feel your feet on the floor and describe pressure and temperature.",
        ],
        "articles": [
            {"title": "Building Daily Emotional Check-ins", "description": "A short framework to monitor mood with low effort."},
            {"title": "Small Habits for Mental Clarity", "description": "Micro-routines that reduce cognitive overload."},
        ],
        "music_moods": ["Soft lo-fi focus", "Acoustic ambient", "Gentle piano minimal"],
        "affirmations": [
            "I can move through today one step at a time.",
            "Steady progress is still progress.",
            "I can choose calm and clarity in this moment.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "calm": {
        "breathing_exercises": [
            "Extended Exhale: Inhale 4s, exhale 6-8s for 10 rounds.",
            "Nasal Breathing: Slow inhale and exhale through the nose for 3 minutes.",
        ],
        "grounding_exercises": [
            "Body Scan: Relax each muscle group from head to toe.",
            "Stillness Minute: Sit quietly and label sensations without judgment.",
        ],
        "articles": [
            {"title": "Protecting a Calm Routine", "description": "Ways to keep peaceful momentum during busy days."},
            {"title": "Mindful Transitions", "description": "How to shift between tasks without stress spikes."},
        ],
        "music_moods": ["Nature soundscapes", "Warm instrumental jazz", "Slow acoustic guitar"],
        "affirmations": [
            "Peace is available to me right now.",
            "I can respond gently to whatever comes next.",
            "My calm is a strength.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "happy": {
        "breathing_exercises": [
            "Smile Breath: Inhale deeply, exhale slowly while relaxing shoulders.",
            "Energy Balance: 4 slow breaths between moments of excitement.",
        ],
        "grounding_exercises": [
            "Gratitude Grounding: Name 3 things going well right now.",
            "Present Snapshot: Describe the best detail in your current environment.",
        ],
        "articles": [
            {"title": "Sustaining Positive Momentum", "description": "How to turn good days into stable routines."},
            {"title": "Sharing Joy Without Burnout", "description": "Celebrate while keeping emotional balance."},
        ],
        "music_moods": ["Upbeat indie pop", "Bright funk grooves", "Sunrise acoustic pop"],
        "affirmations": [
            "I allow myself to enjoy this moment fully.",
            "My joy can fuel meaningful action.",
            "I can share positivity without pressure.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "sad": {
        "breathing_exercises": [
            "Hand-on-Heart Breathing: Inhale 4s, exhale 6s for 2-3 minutes.",
            "Sigh Release: Take a deep inhale and release with a long sigh.",
        ],
        "grounding_exercises": [
            "Comfort Anchor: Hold a soft object and describe its texture slowly.",
            "Name and Normalize: Label your feeling and add 'this can pass'.",
        ],
        "articles": [
            {"title": "Coping With Low Mood", "description": "Simple steps for difficult emotional phases."},
            {"title": "When Sadness Feels Heavy", "description": "Practical ways to ask for support."},
        ],
        "music_moods": ["Gentle piano reflection", "Warm cinematic strings", "Soft rain ambience"],
        "affirmations": [
            "My feelings are valid and temporary.",
            "I can ask for support and still be strong.",
            "I deserve care, especially on hard days.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "angry": {
        "breathing_exercises": [
            "Cooling Breath: Inhale through nose, long exhale through mouth.",
            "Counted Pause: Inhale 4s, hold 2s, exhale 8s for 8 rounds.",
        ],
        "grounding_exercises": [
            "Temperature Reset: Splash cool water and take 3 slow breaths.",
            "Tension Release: Tighten then relax fists, shoulders, and jaw.",
        ],
        "articles": [
            {"title": "Working With Anger Safely", "description": "Channel intensity into constructive action."},
            {"title": "Conflict De-escalation Basics", "description": "Language and pauses that lower friction."},
        ],
        "music_moods": ["Percussive workout beats", "Focus-driven electronic", "Power instrumental rock"],
        "affirmations": [
            "I can feel anger without letting it control me.",
            "I can choose a response that protects my peace.",
            "My boundaries matter and can be expressed calmly.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "fearful": {
        "breathing_exercises": [
            "Grounded Breathing: Inhale 4s, exhale 6s while naming the room.",
            "Butterfly Breath: Cross arms over chest, breathe slowly for 2 minutes.",
        ],
        "grounding_exercises": [
            "Safety Statement: Repeat 'I am here, I am safe right now'.",
            "Object Focus: Describe one nearby object in detail for one minute.",
        ],
        "articles": [
            {"title": "Managing Anxiety Spikes", "description": "Fast techniques for fear surges."},
            {"title": "Returning to Baseline", "description": "How to stabilize body and thoughts after panic."},
        ],
        "music_moods": ["Deep ambient drones", "Slow piano reassurance", "Soft ocean waves"],
        "affirmations": [
            "I am safe in this moment.",
            "I can take this one breath at a time.",
            "My body can return to calm.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "disgust": {
        "breathing_exercises": [
            "Reset Breath: Inhale 4s, exhale 7s to reduce bodily tension.",
            "Clean Air Pause: Step away, breathe slowly for 90 seconds.",
        ],
        "grounding_exercises": [
            "Label and Distance: Name what felt unpleasant, then re-center on now.",
            "Sensory Neutralizer: Touch a neutral object and describe it factually.",
        ],
        "articles": [
            {"title": "Processing Aversion Reactions", "description": "How to regulate strong rejection responses."},
            {"title": "Staying Grounded After Triggers", "description": "Techniques for emotional reset."},
        ],
        "music_moods": ["Neutral ambient textures", "Instrumental chillhop", "Low-key piano loops"],
        "affirmations": [
            "I can acknowledge this feeling without judgment.",
            "I can reset and protect my energy.",
            "I choose what deserves my focus.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
    "surprised": {
        "breathing_exercises": [
            "Settle Breath: Inhale 4s, hold 2s, exhale 6s for 6 rounds.",
            "Paced Breath: Breathe in rhythm with a slow count to stabilize energy.",
        ],
        "grounding_exercises": [
            "Context Check: Name what changed and what is still stable.",
            "Three Facts: Write 3 concrete facts before making decisions.",
        ],
        "articles": [
            {"title": "Handling Sudden Change", "description": "Stabilize quickly when events shift fast."},
            {"title": "From Shock to Clarity", "description": "A framework for thoughtful next steps."},
        ],
        "music_moods": ["Curious upbeat instrumentals", "Light electronic pulse", "Playful acoustic tracks"],
        "affirmations": [
            "I can adapt to new information calmly.",
            "I can pause before reacting.",
            "I trust myself to handle uncertainty.",
        ],
        "hotlines": COMMON_HOTLINES,
    },
}


def get_resources(emotion: str, intent: str, contradiction: bool) -> dict:
    """Return filtered resources based on emotion, intent, and signal consistency."""
    emotion_key = (emotion or "neutral").lower()
    base = RESOURCE_BANK.get(emotion_key, RESOURCE_BANK["neutral"])

    if intent == "gratitude":
        selected = {}
    elif intent == "crisis":
        selected = {
            "hotlines": list(base["hotlines"]),
            "grounding_exercises": list(base["grounding_exercises"][:2]),
        }
    elif intent == "resource_request":
        selected = {
            "breathing_exercises": list(base["breathing_exercises"]),
            "grounding_exercises": list(base["grounding_exercises"]),
            "articles": list(base["articles"]),
            "music_moods": list(base["music_moods"]),
            "affirmations": list(base["affirmations"]),
            "hotlines": list(base["hotlines"]),
        }
    elif intent == "venting":
        selected = {
            "affirmations": list(base["affirmations"]),
            "breathing_exercises": list(base["breathing_exercises"][:1]),
        }
    elif intent == "seeking_advice":
        selected = {
            "articles": list(base["articles"]),
            "breathing_exercises": list(base["breathing_exercises"]),
        }
    else:
        selected = {"affirmations": list(base["affirmations"])}

    if contradiction and selected:
        selected = {
            "note": "Your voice and words seem to tell different stories. That's okay — here are some resources either way.",
            **selected,
        }

    return selected


def format_resources_for_display(resources: dict) -> dict:
    """Rename resource keys for user-facing Streamlit section titles."""
    key_map = {
        "breathing_exercises": "🌬️ Breathing Exercises",
        "grounding_exercises": "🌿 Grounding Techniques",
        "articles": "📖 Recommended Reading",
        "music_moods": "🎵 Music for Your Mood",
        "affirmations": "💬 Affirmations",
        "hotlines": "📞 Support Helplines",
    }

    formatted = {}
    for key, value in resources.items():
        if key == "note":
            formatted["note"] = value
            continue
        formatted[key_map.get(key, key)] = value
    return formatted
