import re

def fix_sentence(text: str) -> str:
    # 1. Subject-verb agreement fixes
    text = re.sub(r'\bThey is\b', 'They are', text)
    text = re.sub(r'\bYou is\b', 'You are', text)
    text = re.sub(r'\bWe is\b', 'We are', text)
    text = re.sub(r'\bI is\b', 'I am', text)
    
    text = re.sub(r'\bI needs\b', 'I need', text)
    text = re.sub(r'\bYou needs\b', 'You need', text)
    text = re.sub(r'\bThey needs\b', 'They need', text)
    text = re.sub(r'\bWe needs\b', 'We need', text)
    
    text = re.sub(r'\bThey wants\b', 'They want', text)
    text = re.sub(r'\bYou wants\b', 'You want', text)
    text = re.sub(r'\bI wants\b', 'I want', text)
    text = re.sub(r'\bWe wants\b', 'We want', text)
    
    # 2. Tense fixes
    text = re.sub(r'\bwill go (\w+) yesterday\b', r'went \1 yesterday', text)
    text = re.sub(r'\bwill go to (\w+) yesterday\b', r'went to \1 yesterday', text)
    
    return text

def main():
    csv_path = "data/grammar_pairs/isl_grammar_pairs.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []
    corrections_count = 0

    for line in lines:
        if not line.strip():
            continue
        fixed = fix_sentence(line)
        if fixed != line:
            corrections_count += 1
        fixed_lines.append(fixed)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print(f"🎉 Corrected {corrections_count} lines with subject-verb agreement or tense errors in {csv_path}.")

if __name__ == "__main__":
    main()
