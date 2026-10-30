import os
import csv
import random

# Common ISL topic-comment grammar templates and vocabulary
# Template placeholders: {SUBJECT}, {LOCATION}, {OBJECT}, {TIME}, {BE}, {NEED}, {WANT}, {GO}, {READ}, {WAIT}
# Verb placeholders are filled by conjugate() based on the subject's grammatical person/number.

GLOSS_TEMPLATES = [
    # (ISL Gloss Pattern, English Translation Template)
    ("{LOCATION} {SUBJECT} GO", "{SUBJECT_CAP} {GO} to {LOCATION}."),
    ("{LOCATION} {SUBJECT} GO {TIME}", "{SUBJECT_CAP} {TENSE_GO} to {LOCATION} {TIME}."),
    ("{OBJECT} {SUBJECT} READ", "{SUBJECT_CAP} {BE} reading the {OBJECT}."),
    ("{OBJECT} {SUBJECT} READ {TIME}", "{SUBJECT_CAP} {TENSE_READ} the {OBJECT} {TIME}."),
    ("{SUBJECT} NAME WHAT", "What is {SUBJECT_GEN} name?"),
    ("{SUBJECT} HELP WANT", "{SUBJECT_CAP} {NEED} help."),
    ("{SUBJECT} HEALTH BAD", "{SUBJECT_CAP} {BE} feeling unwell."),
    ("{LOCATION} HOSPITAL WHERE", "Where is the hospital in {LOCATION}?"),
    ("{SUBJECT} FOOD EAT WANT", "{SUBJECT_CAP} {WANT} to eat food."),
    ("{SUBJECT} WATER DRINK WANT", "{SUBJECT_CAP} {WANT} to drink water."),
    ("{TIME} MEETING HAVE", "There is a meeting {TIME}."),
    ("{SUBJECT} MONEY GIVE PLEASE", "Please give money to {SUBJECT_OBJ}."),
    ("{TIME} {LOCATION} WE MEET", "We will meet at {LOCATION} {TIME}."),
    ("{SUBJECT} BUS WAIT", "{SUBJECT_CAP} {BE} waiting for the bus.")
]

VOCAB = {
    "SUBJECT": ["I", "HE", "SHE", "THEY", "YOU", "MY FRIEND", "MOTHER", "FATHER", "DOCTOR", "TEACHER"],
    "LOCATION": ["HOME", "SCHOOL", "HOSPITAL", "COLLEGE", "MARKET", "OFFICE", "STATION", "BANK"],
    "OBJECT": ["BOOK", "LETTER", "NEWSPAPER", "DOCUMENT", "REPORT", "MESSAGE"],
    "TIME": ["TODAY", "TOMORROW", "YESTERDAY", "MORNING", "EVENING", "NIGHT", "NOW"]
}

# (nominative, objective, possessive/genitive, person)
# person: "1s" = I, "2" = you, "3s" = he/she/it, "3p" = they, "1p" = we
PRONOUN_MAP = {
    "I":         ("I",          "me",         "my",              "1s"),
    "HE":        ("He",         "him",         "his",             "3s"),
    "SHE":       ("She",        "her",         "her",             "3s"),
    "THEY":      ("They",       "them",        "their",           "3p"),
    "YOU":       ("You",        "you",         "your",            "2"),
    "MY FRIEND": ("My friend",  "my friend",   "my friend's",     "3s"),
    "MOTHER":    ("My mother",  "my mother",   "my mother's",     "3s"),
    "FATHER":    ("My father",  "my father",   "my father's",     "3s"),
    "DOCTOR":    ("The doctor", "the doctor",  "the doctor's",    "3s"),
    "TEACHER":   ("The teacher","the teacher", "the teacher's",   "3s"),
}

TIME_MAP = {
    "TODAY":     "today",
    "TOMORROW":  "tomorrow",
    "YESTERDAY": "yesterday",
    "MORNING":   "in the morning",
    "EVENING":   "in the evening",
    "NIGHT":     "at night",
    "NOW":       "now"
}


def conjugate(verb: str, person: str) -> str:
    """
    Return the correct present-tense form of a verb given grammatical person.
    Persons: "1s" (I), "2" (you), "3s" (he/she/it/3rd-sing), "3p" (they), "1p" (we).
    """
    forms = {
        "be":   {"3s": "is",   "1s": "am",  "default": "are"},
        "need": {"3s": "needs","default": "need"},
        "want": {"3s": "wants","default": "want"},
        "go":   {"3s": "goes", "default": "go"},
        "read": {"3s": "reads","default": "read"},
        "wait": {"3s": "waits","default": "wait"},
    }
    verb_forms = forms.get(verb, {"default": verb})
    if person in verb_forms:
        return verb_forms[person]
    return verb_forms["default"]


# Time slots that indicate past / future / present
_PAST_TIMES   = {"YESTERDAY"}
_FUTURE_TIMES = {"TOMORROW"}


def tense_go(time_key: str, person: str) -> str:
    """Correct tense of 'go' given a TIME gloss token."""
    if time_key in _PAST_TIMES:
        return "went"
    if time_key in _FUTURE_TIMES:
        return "will go"
    # Present — use present-continuous 'is/am/are going'
    return f"{conjugate('be', person)} going"


def tense_read(time_key: str, person: str) -> str:
    """Correct tense of 'read' given a TIME gloss token."""
    if time_key in _PAST_TIMES:
        return "read"          # past tense of read is 'read' (same spelling)
    if time_key in _FUTURE_TIMES:
        return "will read"
    # Present — use present-continuous
    return f"{conjugate('be', person)} reading"


def conjugate_go_past(person: str) -> str:
    return "went"


def conjugate_future_go(person: str) -> str:
    return "will go"


def generate_pairs(num_samples: int = 1500) -> list[tuple[str, str]]:
    pairs = []

    for _ in range(num_samples):
        template, target_fmt = random.choice(GLOSS_TEMPLATES)

        sub = random.choice(VOCAB["SUBJECT"])
        loc = random.choice(VOCAB["LOCATION"]).capitalize()
        obj = random.choice(VOCAB["OBJECT"]).lower()
        time_key = random.choice(VOCAB["TIME"])

        gloss = template.format(
            SUBJECT=sub,
            LOCATION=loc.upper(),
            OBJECT=obj.upper(),
            TIME=time_key,
        )

        nom, obj_form, gen, person = PRONOUN_MAP[sub]
        time_trans = TIME_MAP[time_key]

        # Conjugate all verbs
        be   = conjugate("be",   person)
        need = conjugate("need", person)
        want = conjugate("want", person)
        go   = conjugate("go",   person)
        read = conjugate("read", person)
        wait = conjugate("wait", person)

        # Tense-aware verb forms for TIME-qualified templates
        tense_go_str   = tense_go(time_key, person)
        tense_read_str = tense_read(time_key, person)

        sentence = target_fmt.format(
            SUBJECT_CAP=nom,
            SUBJECT_OBJ=obj_form,
            SUBJECT_GEN=gen,
            LOCATION=loc,
            OBJECT=obj,
            TIME=time_trans,
            BE=be,
            NEED=need,
            WANT=want,
            GO=go,
            READ=read,
            WAIT=wait,
            TENSE_GO=tense_go_str,
            TENSE_READ=tense_read_str,
            WILL_GO=tense_go_str,   # backward-compat alias
        )

        # Format prefix for T5
        t5_input = f"translate ISL gloss to English: {gloss}"
        pairs.append((t5_input, sentence))

    return pairs


def main():
    output_dir = "data/grammar_pairs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "isl_grammar_pairs.csv")

    pairs = generate_pairs(2000)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "target_text"])
        writer.writerows(pairs)

    print(f"Generated {len(pairs)} synthetic ISL gloss-to-sentence pairs at {filepath}")

    # Spot-check 10 random rows for subject-verb agreement
    print("\nSpot-check (10 random rows):")
    sample = random.sample(pairs, min(10, len(pairs)))
    for inp, out in sample:
        gloss_part = inp.replace("translate ISL gloss to English: ", "")
        print(f"  {gloss_part!r:50s} -> {out!r}")


if __name__ == "__main__":
    main()
