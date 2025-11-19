import json
import os
import re
from collections import defaultdict
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import pickle
import math

class Indexer:
    def __init__(self, corpus_path, output_dir="index_output", partial_index_threshold=10000,
                 enable_ngrams=True, enable_positions=True, enable_anchor_text=True):
        
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.stemmer = PorterStemmer()
        self.inverted_index = defaultdict(list)  # token -> list of postings
        self.doc_id_to_url = {}  # doc_id -> URL mapping
        self.doc_count = 0
        self.partial_index_count = 0
        self.partial_index_threshold = partial_index_threshold
        
        self.enable_ngrams = enable_ngrams
        self.enable_positions = enable_positions
        self.enable_anchor_text = enable_anchor_text
        
        # N-gram indexes
        self.bigram_index = defaultdict(list) if enable_ngrams else None
        self.trigram_index = defaultdict(list) if enable_ngrams else None
        
        # Duplicate detection
        self.seen_content_hashes = set()  # Track exact duplicates
        self.duplicates_found = 0
        
        # Anchor text index
        self.anchor_text_index = defaultdict(list) if enable_anchor_text else None
        
        # Statistics
        self.stats = {
            'total_docs': 0,
            'unique_tokens': 0,
            'index_size_kb': 0,
            'duplicates_removed': 0,
            'bigrams': 0,
            'trigrams': 0,
            'anchor_texts': 0
        }
    
    def tokenize(self, text):
        # Find all alphanumeric sequences
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens
    
    def compute_content_hash(self, content):
        import hashlib
        # Normalize: remove whitespace and lowercase
        normalized = re.sub(r'\s+', '', content.lower())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, content):
        content_hash = self.compute_content_hash(content)
        if content_hash in self.seen_content_hashes:
            return True
        self.seen_content_hashes.add(content_hash)
        return False
    
    def generate_ngrams(self, tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def extract_content_with_importance(self, html_content, doc_id):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return {}, {}, []
        
        token_importance = defaultdict(float)
        token_positions = defaultdict(list) if self.enable_positions else None
        anchor_links = []  # (anchor_text, target_url) pairs
        position_counter = 0
        
        # Extract title (highest importance)
        title = soup.find('title')
        if title:
            tokens = self.tokenize(title.get_text())
            for token in tokens:
                stemmed = self.stemmer.stem(token)
                token_importance[stemmed] += 3.0
                if self.enable_positions:
                    token_positions[stemmed].append(position_counter)
                    position_counter += 1
        
        # Extract headings (high importance)
        for tag in ['h1', 'h2', 'h3']:
            for heading in soup.find_all(tag):
                tokens = self.tokenize(heading.get_text())
                importance = 2.5 if tag == 'h1' else (2.0 if tag == 'h2' else 1.5)
                for token in tokens:
                    stemmed = self.stemmer.stem(token)
                    token_importance[stemmed] += importance
                    if self.enable_positions:
                        token_positions[stemmed].append(position_counter)
                        position_counter += 1
        
        # Extract bold text (medium importance)
        for bold in soup.find_all(['b', 'strong']):
            tokens = self.tokenize(bold.get_text())
            for token in tokens:
                stemmed = self.stemmer.stem(token)
                token_importance[stemmed] += 1.5
                if self.enable_positions:
                    token_positions[stemmed].append(position_counter)
                    position_counter += 1
        
        # Extract anchor text
        if self.enable_anchor_text:
            for link in soup.find_all('a', href=True):
                anchor_text = link.get_text().strip()
                target_url = link['href']
                if anchor_text and target_url:
                    anchor_links.append((anchor_text, target_url))
        
        # Extract all other text (normal importance)
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        text = soup.get_text()
        tokens = self.tokenize(text)
        for token in tokens:
            stemmed = self.stemmer.stem(token)
            if stemmed not in token_importance:
                token_importance[stemmed] += 1.0
            if self.enable_positions:
                token_positions[stemmed].append(position_counter)
            position_counter += 1
        
        return token_importance, token_positions, anchor_links
    
    def compute_tf(self, token_importance, doc_length):
        tf_scores = {}
        for token, importance in token_importance.items():
            # TF = (frequency * importance) / doc_length
            # Use raw frequency with importance weighting
            tf_scores[token] = importance
        
        return tf_scores
    
    def add_document_to_index(self, doc_id, url, token_tf_scores, token_positions=None, 
                               raw_tokens=None, anchor_links=None):
        self.doc_id_to_url[doc_id] = url
        
        # Add to main inverted index
        for token, tf_score in token_tf_scores.items():
            posting = {
                'doc_id': doc_id,
                'tf': tf_score
            }
            
            # Add positions
            if self.enable_positions and token_positions and token in token_positions:
                posting['positions'] = token_positions[token]
            
            self.inverted_index[token].append(posting)
        
        # Add n-grams
        if self.enable_ngrams and raw_tokens:
            # Generate and add bigrams
            bigrams = self.generate_ngrams(raw_tokens, 2)
            for bigram in bigrams:
                stemmed_bigram = tuple(self.stemmer.stem(t) for t in bigram)
                posting = {'doc_id': doc_id, 'tf': 1.0}
                self.bigram_index[stemmed_bigram].append(posting)
            
            # Generate and add trigrams
            trigrams = self.generate_ngrams(raw_tokens, 3)
            for trigram in trigrams:
                stemmed_trigram = tuple(self.stemmer.stem(t) for t in trigram)
                posting = {'doc_id': doc_id, 'tf': 1.0}
                self.trigram_index[stemmed_trigram].append(posting)
        
        # Add anchor text index
        if self.enable_anchor_text and anchor_links:
            for anchor_text, target_url in anchor_links:
                tokens = self.tokenize(anchor_text)
                for token in tokens:
                    stemmed = self.stemmer.stem(token)
                    posting = {'doc_id': doc_id, 'anchor_text': anchor_text, 'target_url': target_url}
                    self.anchor_text_index[stemmed].append(posting)
    
    def offload_partial_index(self):
        partial_index_file = self.output_dir / f"partial_index_{self.partial_index_count}.pkl"
        
        print(f"Offloading partial index {self.partial_index_count} with {len(self.inverted_index)} tokens...")
        
        # Prepare data to offload
        data_to_offload = {
            'inverted_index': dict(self.inverted_index)
        }
        
        # Include n-grams if enabled
        if self.enable_ngrams:
            data_to_offload['bigram_index'] = dict(self.bigram_index)
            data_to_offload['trigram_index'] = dict(self.trigram_index)
            print(f"  Bigrams: {len(self.bigram_index)}, Trigrams: {len(self.trigram_index)}")
        
        # Include anchor text if enabled
        if self.enable_anchor_text:
            data_to_offload['anchor_text_index'] = dict(self.anchor_text_index)
            print(f"  Anchor texts: {len(self.anchor_text_index)}")
        
        with open(partial_index_file, 'wb') as f:
            pickle.dump(data_to_offload, f)
        
        self.partial_index_count += 1
        
        # Clear memory
        self.inverted_index.clear()
        if self.enable_ngrams:
            self.bigram_index.clear()
            self.trigram_index.clear()
        if self.enable_anchor_text:
            self.anchor_text_index.clear()
    
    def process_document(self, json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            url = data.get('url', '')
            content = data.get('content', '')
            
            if not content:
                return
            
            # Check for exact duplicates
            if self.is_duplicate(content):
                self.duplicates_found += 1
                return  # Skip duplicate
            
            # Extract tokens with importance and positions
            token_importance, token_positions, anchor_links = self.extract_content_with_importance(content, self.doc_count)
            
            if not token_importance:
                return
            
            # Get raw tokens for n-gram generation
            raw_tokens = []
            if self.enable_ngrams:
                try:
                    soup = BeautifulSoup(content, 'html.parser')
                    for script in soup(['script', 'style']):
                        script.decompose()
                    text = soup.get_text()
                    raw_tokens = self.tokenize(text)
                except:
                    pass
            
            # Compute TF scores
            doc_length = sum(token_importance.values())
            tf_scores = self.compute_tf(token_importance, doc_length)
            
            # Add to index
            doc_id = self.doc_count
            self.add_document_to_index(doc_id, url, tf_scores, token_positions, raw_tokens, anchor_links)
            self.doc_count += 1
            
            # Check if we need to offload
            if self.doc_count % self.partial_index_threshold == 0:
                self.offload_partial_index()
                
        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")
    
    def build_index(self):
        print("Starting index construction...")
        
        # Traverse all domain folders
        for domain_folder in self.corpus_path.iterdir():
            if not domain_folder.is_dir():
                continue
            
            print(f"Processing domain: {domain_folder.name}")
            
            # Process all JSON files in domain folder
            for json_file in domain_folder.glob('*.json'):
                self.process_document(json_file)
        
        # Offload any remaining documents
        if self.inverted_index:
            self.offload_partial_index()
        
        print(f"\nIndex construction complete!")
        print(f"Total documents processed: {self.doc_count}")
        print(f"Partial indexes created: {self.partial_index_count}")
    
    def organize_index(self):
        print("\nOrganizing index for efficient search...")
        
        term_to_partial = {}  # Maps term to partial_index_id
        bigram_to_partial = {}  # Maps bigram to pid
        trigram_to_partial = {}  # Maps trigram to pid
        anchor_to_partial = {}  # Maps anchor term to pid
        
        unique_tokens = set()
        unique_bigrams = set()
        unique_trigrams = set()
        unique_anchor_terms = set()
        
        # Build term dictionaries by scanning all partial indexes
        for i in range(self.partial_index_count):
            partial_file = self.output_dir / f"partial_index_{i}.pkl"
            
            print(f"Scanning partial index {i}...")
            with open(partial_file, 'rb') as f:
                data = pickle.load(f)
            
            # Process main inverted index
            partial_index = data.get('inverted_index', {})
            for token in partial_index.keys():
                if token not in term_to_partial:
                    term_to_partial[token] = []
                term_to_partial[token].append(i)
                unique_tokens.add(token)
            
            # Process bigram index
            if self.enable_ngrams and 'bigram_index' in data:
                bigram_index = data['bigram_index']
                for bigram in bigram_index.keys():
                    if bigram not in bigram_to_partial:
                        bigram_to_partial[bigram] = []
                    bigram_to_partial[bigram].append(i)
                    unique_bigrams.add(bigram)
            
            # Process trigram index
            if self.enable_ngrams and 'trigram_index' in data:
                trigram_index = data['trigram_index']
                for trigram in trigram_index.keys():
                    if trigram not in trigram_to_partial:
                        trigram_to_partial[trigram] = []
                    trigram_to_partial[trigram].append(i)
                    unique_trigrams.add(trigram)
            
            # Process anchor text index
            if self.enable_anchor_text and 'anchor_text_index' in data:
                anchor_index = data['anchor_text_index']
                for term in anchor_index.keys():
                    if term not in anchor_to_partial:
                        anchor_to_partial[term] = []
                    anchor_to_partial[term].append(i)
                    unique_anchor_terms.add(term)
        
        # Save term dictionaries
        term_dict_file = self.output_dir / "term_dictionary.pkl"
        print(f"Saving term dictionary...")
        with open(term_dict_file, 'wb') as f:
            pickle.dump(term_to_partial, f)
        
        if self.enable_ngrams:
            bigram_dict_file = self.output_dir / "bigram_dictionary.pkl"
            trigram_dict_file = self.output_dir / "trigram_dictionary.pkl"
            with open(bigram_dict_file, 'wb') as f:
                pickle.dump(bigram_to_partial, f)
            with open(trigram_dict_file, 'wb') as f:
                pickle.dump(trigram_to_partial, f)
            print(f"Saved bigram and trigram dictionaries")
        
        if self.enable_anchor_text:
            anchor_dict_file = self.output_dir / "anchor_dictionary.pkl"
            with open(anchor_dict_file, 'wb') as f:
                pickle.dump(anchor_to_partial, f)
            print(f"Saved anchor text dictionary")
        
        # Save document mapping
        doc_map_file = self.output_dir / "doc_id_to_url.pkl"
        with open(doc_map_file, 'wb') as f:
            pickle.dump(self.doc_id_to_url, f)
        
        # Calculate statistics
        self.stats['total_docs'] = self.doc_count
        self.stats['unique_tokens'] = len(unique_tokens)
        self.stats['duplicates_removed'] = self.duplicates_found
        self.stats['bigrams'] = len(unique_bigrams)
        self.stats['trigrams'] = len(unique_trigrams)
        self.stats['anchor_texts'] = len(unique_anchor_terms)
        
        # Calculate index size (all partial indexes + dictionaries + doc mapping)
        total_size = 0
        for i in range(self.partial_index_count):
            partial_file = self.output_dir / f"partial_index_{i}.pkl"
            total_size += partial_file.stat().st_size
        total_size += term_dict_file.stat().st_size
        if self.enable_ngrams:
            total_size += (self.output_dir / "bigram_dictionary.pkl").stat().st_size
            total_size += (self.output_dir / "trigram_dictionary.pkl").stat().st_size
        if self.enable_anchor_text:
            total_size += (self.output_dir / "anchor_dictionary.pkl").stat().st_size
        total_size += doc_map_file.stat().st_size
        
        self.stats['index_size_kb'] = total_size / 1024
        
        print(f"\nFinal index statistics:")
        print(f"Total documents: {self.stats['total_docs']}")
        print(f"Unique tokens: {self.stats['unique_tokens']}")
        if self.enable_ngrams:
            print(f"Unique bigrams: {self.stats['bigrams']}")
            print(f"Unique trigrams: {self.stats['trigrams']}")
        if self.enable_anchor_text:
            print(f"Anchor text terms: {self.stats['anchor_texts']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Total index size: {self.stats['index_size_kb']:.2f} KB")
        print(f"Partial indexes kept: {self.partial_index_count}")
    
    
def main():
    corpus_path = "/Users/maxbader/Desktop/DEV"
    
    # Initialize indexer
    indexer = Indexer(
        corpus_path, 
        output_dir="index_output", 
        partial_index_threshold=2000,
        enable_ngrams=True,          # 2-gram and 3-gram indexing (2 points)
        enable_positions=True,       # Word position tracking (2 points)
        enable_anchor_text=True      # Anchor text indexing (1 point)
    )
    
    # Build index
    indexer.build_index()
    
    # Create term dictionary. Keeps partials separate
    indexer.organize_index()

if __name__ == "__main__":
    main()