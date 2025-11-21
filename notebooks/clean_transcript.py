import re
import csv
import json
import argparse
from pathlib import Path

def is_question(text):
    """Check if text is a question using heuristics."""
    text = text.strip()
    if not text:
        return False
    # Check for explicit question labels
    if re.match(r'^(Q|Question|Interviewer):\s*', text, re.IGNORECASE):
        return True
    # Check for question marks or common question starters
    question_starters = ['what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does', 'did']
    lower_text = text.lower()
    return (text.endswith('?') or 
            any(lower_text.startswith(starter) for starter in question_starters))

def extract_speaker_and_text(line):
    """Extract speaker and text from a line (e.g., 'Speaker 1: Hello')."""
    match = re.match(r'^([^:]+):\s*(.*)$', line)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Unknown", line.strip()

def process_txt(file_path):
    """Process TXT transcript into (speaker, text) tuples."""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or re.match(r'^\d+:\d+:\d+', line):  # Skip timestamps
                continue
            speaker, text = extract_speaker_and_text(line)
            lines.append((speaker, text))
    return lines

def process_csv(file_path):
    """Process CSV transcript into (speaker, text) tuples."""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Auto-detect speaker/text columns
        speaker_col = None
        text_col = None
        for col in reader.fieldnames:
            if col.lower() in ['speaker', 'name', 'person']:
                speaker_col = col
            elif col.lower() in ['text', 'utterance', 'message', 'content']:
                text_col = col
        
        if not text_col:
            raise ValueError("CSV must contain a 'text' column (or similar).")
        
        for row in reader:
            speaker = row.get(speaker_col, 'Unknown') if speaker_col else 'Unknown'
            text = row[text_col]
            if text.strip():
                lines.append((speaker.strip(), text.strip()))
    return lines

def is_elon_musk(speaker_name):
    """Check if speaker is Elon Musk (case-insensitive, handles common variations)."""
    if not speaker_name:
        return False
    speaker_lower = speaker_name.lower()
    # Match full name, first name only, or common abbreviations
    return any(alias in speaker_lower for alias in [
        'elon musk',
        'musk',
        'elon'
    ]) and not any(false_positive in speaker_lower for false_positive in [
        'not elon',
        'fake elon',
        'imposter'
    ])

def generate_qa_pairs(lines, elon_mode=False):
    """
    Convert (speaker, text) list into question-response pairs.
    
    Args:
        lines: List of (speaker, text) tuples
        elon_mode: If True, only keep responses from Elon Musk
    """
    qa_pairs = []
    i = 0
    while i < len(lines):
        speaker, text = lines[i]
        if is_question(text):
            # Find next valid response
            j = i + 1
            while j < len(lines):
                next_speaker, next_text = lines[j]
                
                # Skip if it's another question
                if is_question(next_text):
                    j += 1
                    continue
                
                # Apply filtering based on mode
                if elon_mode:
                    # Only keep if response speaker is Elon Musk
                    if is_elon_musk(next_speaker):
                        qa_pairs.append({
                            'question': text,
                            'response': next_text,
                            'question_speaker': speaker,
                            'response_speaker': next_speaker
                        })
                        i = j
                        break
                else:
                    # Default: keep only if speakers are different
                    if next_speaker != speaker:
                        qa_pairs.append({
                            'question': text,
                            'response': next_text,
                            'question_speaker': speaker,
                            'response_speaker': next_speaker
                        })
                        i = j
                        break
                
                j += 1
            # If no valid response found, skip this question
        i += 1
    return qa_pairs

def save_to_jsonl(qa_pairs, output_path):
    """Save Q&A pairs to JSONL (one JSON object per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Clean transcript into Q&A pairs (JSONL output).')
    parser.add_argument('input_file', help='Input transcript (CSV or TXT)')
    parser.add_argument('-o', '--output', default='cleaned_transcript.jsonl', 
                        help='Output JSONL file (default: cleaned_transcript.jsonl)')
    parser.add_argument('--elon', action='store_true',
                        help='Keep only responses from Elon Musk (case-insensitive)')
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Process based on file extension
    if input_path.suffix.lower() == '.txt':
        lines = process_txt(input_path)
    elif input_path.suffix.lower() == '.csv':
        lines = process_csv(input_path)
    else:
        raise ValueError("Unsupported file format. Use .txt or .csv.")

    # Generate Q&A pairs with selected mode
    qa_pairs = generate_qa_pairs(lines, elon_mode=args.elon)
    
    # Save to JSONL
    save_to_jsonl(qa_pairs, args.output)
    
    mode = "Elon Musk responses only" if args.elon else "Different speakers only"
    print(f"Processed {len(qa_pairs)} Q&A pairs ({mode}). Saved to {args.output}")

if __name__ == '__main__':
    main()