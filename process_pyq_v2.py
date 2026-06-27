import os
import re
import time
import fitz
import requests
import openai

SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"
OPENAI_KEY = "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"

openai_client = openai.OpenAI(api_key=OPENAI_KEY)

PDF_FOLDER = r"C:\Users\hakim\Documents\neet-ai\pyq-pdfs"

PDF_CONFIG = {
    "AIPMT_2008.pdf": {"year": 2008, "format": "sol"},
    "AIPMT_2013.pdf": {"year": 2013, "format": "q_format"},
    "AIPMT_2015.pdf": {"year": 2015, "format": "standard"},
    "QP_2018.pdf":    {"year": 2018, "format": "standard"},
    "QP_2019.pdf":    {"year": 2019, "format": "standard"},
    "QP_2023.pdf":    {"year": 2023, "format": "abcd_inline"},
    "QP_2024.pdf":    {"year": 2024, "format": "standard"},
    "QP_2025.pdf":    {"year": 2025, "format": "standard"},
}

def get_embedding(text):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:500]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def detect_subject(text, position, total):
    text_lower = text.lower()
    bio_keywords = ['cell', 'plant', 'animal', 'photosynthesis', 'respiration', 'dna', 'rna', 'protein', 'enzyme', 'chromosome', 'mitosis', 'meiosis', 'ecology', 'evolution', 'biodiversity', 'kingdom', 'phylum', 'species', 'organism', 'tissue', 'organ', 'blood', 'heart', 'nerve', 'hormone', 'reproduction', 'genetics', 'mutation']
    chem_keywords = ['mole', 'atom', 'molecule', 'reaction', 'acid', 'base', 'salt', 'bond', 'orbital', 'electron', 'proton', 'neutron', 'element', 'compound', 'solution', 'concentration', 'equilibrium', 'oxidation', 'reduction', 'polymer', 'organic', 'alkane', 'alkene', 'benzene', 'ester', 'ether']
    phys_keywords = ['force', 'velocity', 'acceleration', 'momentum', 'energy', 'power', 'wave', 'frequency', 'current', 'voltage', 'resistance', 'magnetic', 'electric', 'gravitational', 'temperature', 'pressure', 'volume', 'mass', 'charge', 'field', 'lens', 'mirror', 'refraction']

    bio_score = sum(1 for k in bio_keywords if k in text_lower)
    chem_score = sum(1 for k in chem_keywords if k in text_lower)
    phys_score = sum(1 for k in phys_keywords if k in text_lower)

    max_score = max(bio_score, chem_score, phys_score)
    if max_score == 0:
        ratio = position / total
        if ratio < 0.33: return "Physics"
        elif ratio < 0.66: return "Chemistry"
        else: return "Biology"

    if bio_score == max_score: return "Biology"
    if chem_score == max_score: return "Chemistry"
    return "Physics"

def parse_standard(text, year):
    questions = []
    blocks = re.split(r'\n(?=\d+\.[\s\n])', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        num_match = re.match(r'^(\d+)\.\s*', block)
        if not num_match:
            continue

        q_num = int(num_match.group(1))
        if q_num < 1 or q_num > 200:
            continue

        lines = block.split('\n')
        q_text = []
        options = {'a': '', 'b': '', 'c': '', 'd': ''}
        answer = ''
        current_opt = None
        reading_q = True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            ans_match = re.search(r'Ans\.\s*[\(\[]?([1-4abcd])[\)\]]?', line, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1)
                reading_q = False
                continue

            opt1_match = re.match(r'^\(([1-4])\)\s*(.*)', line)
            opta_match = re.match(r'^([a-d])\.\s*(.*)', line, re.IGNORECASE)

            if opt1_match:
                reading_q = False
                opt_num = opt1_match.group(1)
                opt_text = opt1_match.group(2)
                opt_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
                current_opt = opt_map.get(opt_num)
                if current_opt:
                    options[current_opt] = opt_text
            elif opta_match:
                reading_q = False
                current_opt = opta_match.group(1).lower()
                options[current_opt] = opta_match.group(2)
            elif current_opt and not reading_q:
                options[current_opt] += ' ' + line
            elif reading_q:
                if re.match(r'^\d+\.\s*', line):
                    q_text.append(re.sub(r'^\d+\.\s*', '', line))
                else:
                    q_text.append(line)

        question_text = ' '.join(q_text).strip()
        question_text = re.sub(r'\s+', ' ', question_text)

        if len(question_text) > 20 and (options['a'] or answer):
            questions.append({
                'year': year,
                'question': question_text,
                'option_a': options['a'].strip(),
                'option_b': options['b'].strip(),
                'option_c': options['c'].strip(),
                'option_d': options['d'].strip(),
                'correct_answer': answer.strip()
            })

    return questions

def parse_sol_format(text, year):
    questions = []
    blocks = re.split(r'\n\s*\n\s*(?=\d+\.)', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        num_match = re.match(r'^(\d+)\.\s*', block)
        if not num_match:
            continue

        lines = block.split('\n')
        q_text = []
        options = {'a': '', 'b': '', 'c': '', 'd': ''}
        answer = ''
        current_opt = None
        reading_q = True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.search(r'Sol\.|Solution|Students find', line, re.IGNORECASE):
                break

            ans_match = re.search(r'(?:Ans(?:wer)?|Sol)\s*[\.\:]?\s*[\(\[]?([1-4abcd])[\)\]]?', line, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1)
                reading_q = False
                continue

            opt_match = re.match(r'^\(([1-4])\)\s*(.*)', line)
            opta_match = re.match(r'^([a-d])\.\s*(.*)', line, re.IGNORECASE)

            if opt_match:
                reading_q = False
                opt_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
                current_opt = opt_map.get(opt_match.group(1))
                if current_opt:
                    options[current_opt] = opt_match.group(2)
            elif opta_match:
                reading_q = False
                current_opt = opta_match.group(1).lower()
                options[current_opt] = opta_match.group(2)
            elif current_opt and not reading_q:
                options[current_opt] += ' ' + line
            elif reading_q:
                if re.match(r'^\d+\.\s*', line):
                    q_text.append(re.sub(r'^\d+\.\s*', '', line))
                else:
                    q_text.append(line)

        question_text = ' '.join(q_text).strip()
        question_text = re.sub(r'\s+', ' ', question_text)

        if len(question_text) > 20:
            questions.append({
                'year': year,
                'question': question_text,
                'option_a': options['a'].strip(),
                'option_b': options['b'].strip(),
                'option_c': options['c'].strip(),
                'option_d': options['d'].strip(),
                'correct_answer': answer.strip()
            })

    return questions

def parse_q_format(text, year):
    questions = []
    blocks = re.split(r'\n(?=Q\.\d+\s)', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        num_match = re.match(r'^Q\.(\d+)\s*', block)
        if not num_match:
            continue

        lines = block.split('\n')
        q_text = []
        options = {'a': '', 'b': '', 'c': '', 'd': ''}
        answer = ''
        current_opt = None
        reading_q = True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.search(r'^Sol\.|^Solution', line, re.IGNORECASE):
                break

            ans_match = re.search(r'Ans\.\s*\[([1-4])\]', line)
            if ans_match:
                ans_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
                answer = ans_map.get(ans_match.group(1), ans_match.group(1))
                reading_q = False
                continue

            opt_match = re.match(r'^\(([1-4])\)\s*(.*)', line)
            if opt_match:
                reading_q = False
                opt_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
                current_opt = opt_map.get(opt_match.group(1))
                if current_opt:
                    options[current_opt] = opt_match.group(2)
            elif current_opt and not reading_q:
                options[current_opt] += ' ' + line
            elif reading_q:
                if re.match(r'^Q\.\d+\s*', line):
                    q_text.append(re.sub(r'^Q\.\d+\s*', '', line))
                else:
                    q_text.append(line)

        question_text = ' '.join(q_text).strip()
        question_text = re.sub(r'\s+', ' ', question_text)

        if len(question_text) > 20:
            questions.append({
                'year': year,
                'question': question_text,
                'option_a': options['a'].strip(),
                'option_b': options['b'].strip(),
                'option_c': options['c'].strip(),
                'option_d': options['d'].strip(),
                'correct_answer': answer.strip()
            })

    return questions

def parse_abcd_inline(text, year):
    questions = []
    blocks = re.split(r'\n(?=\s*\d+\.)', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        num_match = re.match(r'^(\d+)\.\s*', block)
        if not num_match:
            continue

        q_num = int(num_match.group(1))
        if q_num < 1 or q_num > 200:
            continue

        lines = block.split('\n')
        q_text = []
        options = {'a': '', 'b': '', 'c': '', 'd': ''}
        answer = ''
        current_opt = None
        reading_q = True

        for line in lines:
            line = line.strip()
            if not line:
                continue

            ans_match = re.search(r'(?:Ans(?:wer)?)\s*[\.\:]?\s*([a-d])', line, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1).lower()
                continue

            opt_match = re.match(r'^([a-d])\.\s+(.*)', line, re.IGNORECASE)
            if opt_match:
                reading_q = False
                current_opt = opt_match.group(1).lower()
                options[current_opt] = opt_match.group(2)
            elif current_opt and not reading_q and not re.match(r'^[a-d]\.', line, re.IGNORECASE):
                options[current_opt] += ' ' + line
            elif reading_q:
                if re.match(r'^\d+\.\s*', line):
                    q_text.append(re.sub(r'^\d+\.\s*', '', line))
                else:
                    q_text.append(line)

        question_text = ' '.join(q_text).strip()
        question_text = re.sub(r'\s+', ' ', question_text)

        if len(question_text) > 20 and options['a']:
            questions.append({
                'year': year,
                'question': question_text,
                'option_a': options['a'].strip(),
                'option_b': options['b'].strip(),
                'option_c': options['c'].strip(),
                'option_d': options['d'].strip(),
                'correct_answer': answer.strip()
            })

    return questions

def extract_questions_from_pdf(pdf_path, config):
    year = config['year']
    fmt = config['format']
    print(f"\nProcessing {os.path.basename(pdf_path)} ({year})...")

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    full_text = re.sub(r'SPACE FOR ROUGH WORK.*?\n', '', full_text)
    full_text = re.sub(r'www\.\S+', '', full_text)
    full_text = re.sub(r'Download From.*?\n', '', full_text)
    full_text = re.sub(r'Page \d+.*?\n', '', full_text)

    if fmt == 'sol':
        questions = parse_sol_format(full_text, year)
    elif fmt == 'q_format':
        questions = parse_q_format(full_text, year)
    elif fmt == 'abcd_inline':
        questions = parse_abcd_inline(full_text, year)
    else:
        questions = parse_standard(full_text, year)

    total = len(questions)
    for i, q in enumerate(questions):
        q['subject'] = detect_subject(q['question'], i, total)

    print(f"  Extracted {len(questions)} questions")
    return questions

def upload_to_supabase(questions):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    success = 0
    failed = 0

    for i, q in enumerate(questions):
        try:
            embedding = get_embedding(q['question'])
            if embedding:
                q['embedding'] = embedding

            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=headers,
                json=q,
                timeout=30
            )

            if response.status_code in [200, 201]:
                success += 1
                print(f"  Uploaded {i+1}/{len(questions)}", end='\r')
            else:
                failed += 1
                print(f"  Failed {i+1}: {response.text[:100]}")

        except Exception as e:
            failed += 1
            print(f"  Error {i+1}: {e}")

        time.sleep(0.3)

    print(f"\n  Done — {success} uploaded, {failed} failed")

def main():
    all_questions = []

    for pdf_file, config in PDF_CONFIG.items():
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_file} — not found")
            continue

        questions = extract_questions_from_pdf(pdf_path, config)
        all_questions.extend(questions)

    print(f"\nTotal questions extracted: {len(all_questions)}")
    print("\nSample questions:")
    for q in all_questions[:3]:
        print(f"\nYear: {q['year']} | Subject: {q['subject']}")
        print(f"Q: {q['question'][:100]}")
        print(f"A: {q['option_a'][:50]}")
        print(f"Ans: {q['correct_answer']}")

    confirm = input("\nUpload to Supabase? (yes/no): ")
    if confirm.lower() == 'yes':
        print("\nUploading...")
        upload_to_supabase(all_questions)
        print("\nAll done!")
    else:
        print("Upload cancelled.")

if __name__ == "__main__":
    main()