import os
import string
from collections import Counter

STOP_WORDS = {
    "from", "as", "was", "than", "too", "to", "been", "if", "with", "or", "on",
    "are", "but", "that", "be", "is", "this", "it", "the", "being", "a", "so",
    "for", "by", "then", "were", "at", "of", "in", "an", "and"
}

def read_file(filename="doc.txt"):
    """Read the content of a file."""
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(filepath, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"[!] Error: '{filename}' not found in the script's directory.")
        exit()

def clean_text(text):
    """Lowercase, remove punctuation and stop words."""
    translator = str.maketrans("", "", string.punctuation)
    words = text.lower().translate(translator).split()
    return [word for word in words if word not in STOP_WORDS]

def generate_ngrams(words, n):
    """Generate n-grams from a list of words."""
    return list(zip(*[words[i:] for i in range(n)]))

def top_ngrams(ngrams_list, top_n=5):
    """Return the top N most common n-grams."""
    return Counter(ngrams_list).most_common(top_n)

def display_ngrams(title, ngrams):
    """Nicely display the n-grams with their counts."""
    print(f"\n=== {title} ===\n{'-' * (len(title) + 8)}")
    for idx, (ngram, count) in enumerate(ngrams, 1):
        print(f"{idx}. {' '.join(ngram)}  ->  {count} times")

def main():
    text = read_file()
    tokens = clean_text(text)

    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)

    top_bigrams = top_ngrams(bigrams)
    top_trigrams = top_ngrams(trigrams)

    display_ngrams("Most Frequent Bigrams", top_bigrams)
    display_ngrams("Most Frequent Trigrams", top_trigrams)

if __name__ == "__main__":
    main()
