import pandas as pd
import spacy
from collections import Counter

# --- Configuration ---
INPUT_FILE = "data/bbc_news.csv"
OUTPUT_FILE = "data/bbc_news_tagged.csv"

def load_data(filepath):
    """
    Loads the CSV data from the given filepath.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {filepath} with {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def process_nlp(df):
    """
    Applies NLP to extract Named Entities (NER) and Part-of-Speech (POS) tags
    using spaCy.
    """
    # Load the English language model
    # Note: Run 'python -m spacy download en_core_web_sm' first in your terminal
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Error: Model not found. Please run: python -m spacy download en_core_web_sm")
        return df

    print("Processing text (this may take a moment)...")

    # Inner function to apply to each row
    def extract_features(text):
        doc = nlp(str(text))
        
        # 1. Named Entity Recognition (NER)
        # Format: (Entity Text, Label) -> e.g., ("London", "GPE")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # 2. Part of Speech (POS)
        # Format: (Word, POS Tag) -> e.g., ("Running", "VERB")
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return entities, pos_tags

    # Apply the function to the 'title' column
    # The function returns a tuple (entities, pos_tags)
    results = df['title'].apply(extract_features)
    
    # Unzip the results into two separate new columns
    df['entities'], df['pos_tags'] = zip(*results)

    return df

def show_stats(df):
    """
    Prints simple statistics about the extracted entities.
    """
    # Flatten the list of lists of entities to count the most common ones
    # We only take the entity text (ent[0]) not the label for this count
    all_ents = [ent[0] for row in df['entities'] for ent in row]
    
    if not all_ents:
        print("\n No entities found.")
        return

    top_ents = Counter(all_ents).most_common(5)
    
    print("\n Top 5 Entities Found in Headlines:")
    for ent, count in top_ents:
        print(f"   - {ent}: {count}")

def main():
    # 1. Load Data
    df = load_data(INPUT_FILE)
    
    if df is not None:
        # 2. Process Data
        df_tagged = process_nlp(df)
        
        # 3. Show Statistics
        show_stats(df_tagged)
        
        # 4. Save Results
        df_tagged.to_csv(OUTPUT_FILE, index=False)
        print(f"\n Processed data saved successfully to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
