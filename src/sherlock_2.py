# ============================================================
# ARCHIVIST'S PUZZLE: FINAL PRODUCTION PIPELINE
# ============================================================

!pip install -q sentence-transformers pandas numpy scikit-learn networkx matplotlib seaborn tqdm

import re
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ============================================================
# SECTION 1: SETTINGS & MODEL LOADING
# ============================================================

# UPDATE THESE PATHS
BOOK_A_PATH   = '/content/data/Hackrush 2026 ML PS/Test Data/BookA_test.csv'
BOOK_B_PATH   = '/content/data/Hackrush 2026 ML PS/Test Data/BookB_test.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Using device: {device}")

# Using the high-accuracy model for the final run
model = SentenceTransformer('all-mpnet-base-v2').to(device)

# ============================================================
# SECTION 2: ADVANCED FEATURE EXTRACTION
# ============================================================

def get_chapter_info(text):
    """Strictly identify chapter markers to use as anchor points."""
    head = text[:300].strip().upper()
    
    # Patterns for Chapter I, Chapter 1, or Chapter One
    roman = re.search(r'CHAPTER\s+([IVXLCDM]+)', head)
    if roman: return {'is_start': True, 'num': roman_to_int(roman.group(1))}
    
    numeric = re.search(r'CHAPTER\s+(\d+)', head)
    if numeric: return {'is_start': True, 'num': int(numeric.group(1))}
    
    words = {'ONE':1,'TWO':2,'THREE':3,'FOUR':4,'FIVE':5,'SIX':6} # Expand as needed
    for word, n in words.items():
        if f'CHAPTER {word}' in head: return {'is_start': True, 'num': n}
        
    return {'is_start': False, 'num': None}

def roman_to_int(s):
    vals = {'I':1,'V':5,'X':10,'L':50,'C':100}
    res, prev = 0, 0
    for char in reversed(s):
        v = vals.get(char, 0)
        res += v if v >= prev else -v
        prev = v
    return res

def get_linguistic_signals(text):
    """Detect 'Lock and Key' continuity signals."""
    return {
        'incomplete': not text.rstrip().endswith(('.', '!', '?', '"')),
        'lowercase': text[0].islower() if text else False,
        'pronoun_ratio': sum(1 for w in text.split() if w.lower() in ['he','she','it','they']) / max(len(text.split()), 1)
    }

# ============================================================
# SECTION 3: THE TWO-PHASE OPTIMIZER
# ============================================================

def run_production_pipeline(df, book_name):
    print(f"\n🚀 Processing {book_name}...")
    
    # 1. Generate Embeddings (all-mpnet-base-v2)
    embs = model.encode(df['text'].tolist(), show_progress_bar=True, device=device)
    sim_matrix = cosine_similarity(embs)
    
    # 2. Assign Pages to Chapter Buckets (The "Hard Constraint")
    # Identify anchor pages
    df['ch_info'] = df['text'].apply(get_chapter_info)
    df['signals'] = df['text'].apply(get_linguistic_signals)
    
    # Simple assignment: Assign each page to the most recent chapter marker seen
    # For shuffled data, we assign pages to the chapter marker they are most SIMILAR to
    markers = df[df['ch_info'].apply(lambda x: x['is_start'])]
    page_to_chapter = {}
    
    for idx, row in df.iterrows():
        if row['ch_info']['is_start']:
            page_to_chapter[idx] = row['ch_info']['num']
        else:
            # Assign to the chapter marker with highest semantic similarity
            best_ch = 1
            best_sim = -1
            for _, m_row in markers.iterrows():
                s = sim_matrix[idx][m_row.name]
                if s > best_sim:
                    best_sim = s
                    best_ch = m_row['ch_info']['num']
            page_to_chapter[idx] = best_ch

    # 3. Order pages WITHIN each chapter bucket
    final_ordering = []
    for ch_num in sorted(set(page_to_chapter.values())):
        ch_indices = [i for i, c in page_to_chapter.items() if c == ch_num]
        
        if len(ch_indices) > 1:
            # Run SA only on this chapter's pages
            ordered_ch = optimize_subsequence(ch_indices, sim_matrix, df)
            final_ordering.extend(ordered_ch)
        else:
            final_ordering.extend(ch_indices)
            
    return final_ordering

def optimize_subsequence(indices, sim_matrix, df):
    """Greedy start followed by high-intensity SA for a specific chapter."""
    # 1. Greedy local start
    current = [indices[0]] # Simplify: start with the first found
    remaining = set(indices[1:])
    while remaining:
        last = current[-1]
        next_node = max(remaining, key=lambda x: sim_matrix[last][x])
        current.append(next_node)
        remaining.remove(next_node)
    
    # 2. Intensive SA (Delta-Score)
    for _ in range(50000):
        i, j = random.sample(range(len(current)), 2)
        
        # Calculate Delta (Local edges only)
        before = get_local_score(current, i, sim_matrix, df) + get_local_score(current, j, sim_matrix, df)
        current[i], current[j] = current[j], current[i]
        after = get_local_score(current, i, sim_matrix, df) + get_local_score(current, j, sim_matrix, df)
        
        if after < before: # Revert if worse
            current[i], current[j] = current[j], current[i]
            
    return current

def get_local_score(seq, pos, sim, df):
    score = 0
    # Link to previous
    if pos > 0:
        score += sim[seq[pos-1]][seq[pos]]
        # Linguistic Bonus (Sentence continuity)
        if df.iloc[seq[pos-1]]['signals']['incomplete'] and df.iloc[seq[pos]]['signals']['lowercase']:
            score += 0.5
    # Link to next
    if pos < len(seq) - 1:
        score += sim[seq[pos]][seq[pos+1]]
    return score

# ============================================================
# SECTION 4: EXECUTION
# ============================================================

# Load Data
bookA = pd.read_csv(BOOK_A_PATH)
bookB = pd.read_csv(BOOK_B_PATH)

# Run Pipeline
order_A = run_production_pipeline(bookA, "Book A")
order_B = run_production_pipeline(bookB, "Book B")

# Create Final CSVs
def save_sub(ordering, df, name):
    pd.DataFrame({
        'original_page': range(1, len(df) + 1),
        'shuffled_page': [df.iloc[i]['page'] for i in ordering]
    }).to_csv(name, index=False)
    print(f"✅ Saved {name}")

save_sub(order_A, bookA, "BookA_FINAL_v3.csv")
save_sub(order_B, bookB, "BookB_FINAL_v3.csv")