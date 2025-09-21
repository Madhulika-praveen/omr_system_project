import csv
from collections import defaultdict

import csv
from collections import defaultdict

SUBJECTS = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]

def load_answer_key(filepath):
    """
    Loads messy CSV answer key (with quotes and missing cells)
    and returns a dictionary:
    {subject: {q_no: 'a,b,...'}}
    """
    answer_key = defaultdict(dict)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers = [h.strip() for h in headers]
        
        # Map any typos (optional)
        headers = [h if h in SUBJECTS else SUBJECTS[i] for i, h in enumerate(headers)]
        
        for row in reader:
            # pad row if shorter than headers
            if len(row) < len(headers):
                row += [''] * (len(headers) - len(row))
            for subj_idx, cell in enumerate(row):
                subject = headers[subj_idx]
                cell = cell.strip().replace('"', '')
                if not cell:
                    continue
                # handle multiple entries separated by comma
                entries = cell.split(',')
                for entry in entries:
                    parts = entry.split('-')
                    if len(parts) != 2:
                        continue
                    try:
                        q_no = int(parts[0].strip())
                        ans = parts[1].strip().lower()
                        if q_no in answer_key[subject]:
                            answer_key[subject][q_no] += ',' + ans
                        else:
                            answer_key[subject][q_no] = ans
                    except ValueError:
                        continue

    # Ensure all subjects exist
    for subj in SUBJECTS:
        if subj not in answer_key:
            answer_key[subj] = {}

    return dict(answer_key)



def score_sheet(student_answers, answer_key):
    """
    Compare student answers to answer key.
    Handles multiple correct answers.
    Returns:
        scores: {subject: score}
        total_score: sum of all correct answers
    """
    scores = {}
    for subject, questions in answer_key.items():
        scores[subject] = 0
        for q_no, correct_ans in questions.items():
            student_ans = student_answers.get(subject, {}).get(q_no, '').lower().replace(' ', '')
            correct_set = set([a.strip().lower() for a in correct_ans.split(',')])
            # handle multi-answer in student bubble as well
            student_set = set(student_ans.split(',')) if student_ans else set()
            if student_set & correct_set:  # intersection not empty
                scores[subject] += 1

    total_score = sum(scores.values())
    return scores, total_score

