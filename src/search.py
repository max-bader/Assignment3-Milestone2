"""
OPTIMIZED Search Component - Milestone 2
Precomputes and caches IDF scores for fast startup and queries
"""

import pickle
import math
import time
from pathlib import Path
from nltk.stem import PorterStemmer
import re
from collections import defaultdict

class SearchEngine:
    def __init__(self, index_dir="index_output", force_recompute_idf=False):
        """
        Initialize search engine with IDF caching
        
        Args:
            index_dir: Directory containing the index files
            force_recompute_idf: Force recomputation of IDF scores
        """
        self.index_dir = Path(index_dir)
        self.stemmer = PorterStemmer()
        
        print("Loading search engine...")
        start_time = time.time()
        
        # Load term dictionary (small - maps terms to partial indexes)
        term_dict_file = self.index_dir / "term_dictionary.pkl"
        with open(term_dict_file, 'rb') as f:
            self.term_to_partial = pickle.load(f)
        
        # Load document mapping
        doc_map_file = self.index_dir / "doc_id_to_url.pkl"
        with open(doc_map_file, 'rb') as f:
            self.doc_id_to_url = pickle.load(f)
        
        self.total_docs = len(self.doc_id_to_url)
        
        # Load or compute IDF scores (with caching!)
        idf_cache_file = self.index_dir / "idf_scores.pkl"
        
        if idf_cache_file.exists() and not force_recompute_idf:
            print("Loading cached IDF scores...")
            with open(idf_cache_file, 'rb') as f:
                self.idf_scores = pickle.load(f)
            print(f"✓ IDF scores loaded from cache")
        else:
            print("Computing IDF scores (this will take a while, but only once)...")
            self.idf_scores = self._compute_and_cache_idf_scores(idf_cache_file)
        
        load_time = (time.time() - start_time) * 1000
        print(f"✓ Search engine ready! Loaded in {load_time:.2f}ms")
        print(f"✓ Total documents: {self.total_docs}")
        print(f"✓ Total terms: {len(self.term_to_partial)}")
    
    def _compute_and_cache_idf_scores(self, cache_file):
        """
        Compute IDF scores and cache them for future use
        This only needs to be done once!
        
        Returns:
            dict: {term: idf_score}
        """
        idf_scores = {}
        term_doc_freq = defaultdict(set)  # term -> set of doc_ids
        
        # Get number of partial indexes
        partial_files = sorted(self.index_dir.glob("partial_index_*.pkl"))
        total_partials = len(partial_files)
        
        print(f"Scanning {total_partials} partial indexes...")
        
        for idx, partial_file in enumerate(partial_files):
            if idx % 1 == 0:
                print(f"  Processing partial {idx+1}/{total_partials}...")
            
            with open(partial_file, 'rb') as f:
                data = pickle.load(f)
                inverted_index = data.get('inverted_index', {})
                
                # For each term, collect unique doc_ids
                for term, postings in inverted_index.items():
                    for posting in postings:
                        term_doc_freq[term].add(posting['doc_id'])
        
        print("Calculating IDF scores...")
        # Calculate IDF for each term
        for term, doc_ids in term_doc_freq.items():
            df = len(doc_ids)  # Document frequency
            idf = math.log(self.total_docs / df) if df > 0 else 0
            idf_scores[term] = idf
        
        # Cache the IDF scores
        print(f"Caching IDF scores to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(idf_scores, f)
        
        print(f"✓ IDF scores computed and cached for {len(idf_scores)} terms")
        return idf_scores
    
    def tokenize(self, text):
        """Extract alphanumeric tokens from text"""
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens
    
    def get_postings_for_term(self, term):
        """
        Get postings for a term by loading ONLY the relevant partial indexes.
        
        Args:
            term: Stemmed term to look up
            
        Returns:
            List of postings: [{'doc_id': int, 'tf': float, ...}, ...]
        """
        # Check if term exists
        if term not in self.term_to_partial:
            return []
        
        # Get which partial indexes contain this term
        partial_ids = self.term_to_partial[term]
        
        all_postings = []
        
        # Load ONLY the relevant partial indexes
        for partial_id in partial_ids:
            partial_file = self.index_dir / f"partial_index_{partial_id}.pkl"
            
            with open(partial_file, 'rb') as f:
                data = pickle.load(f)
                inverted_index = data.get('inverted_index', {})
            
            # Get postings for this term
            if term in inverted_index:
                all_postings.extend(inverted_index[term])
        
        return all_postings
    
    def search(self, query, top_k=5):
        """
        Search for query using Boolean AND with TF-IDF ranking
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            tuple: (results, query_time_ms)
                results: List of (url, score) tuples
                query_time_ms: Query processing time in milliseconds
        """
        start_time = time.time()
        
        # Tokenize and stem query
        query_tokens = self.tokenize(query)
        query_terms = [self.stemmer.stem(token) for token in query_tokens]
        query_terms = list(set(query_terms))  # Remove duplicates
        
        if not query_terms:
            return [], 0
        
        # Boolean AND: Get documents that contain ALL query terms
        doc_ids_with_all_terms = None
        term_postings_map = {}  # {term: {doc_id: posting}}
        
        for term in query_terms:
            postings = self.get_postings_for_term(term)
            
            if not postings:
                # Term not found - no results for AND query
                query_time = (time.time() - start_time) * 1000
                return [], query_time
            
            # Convert to dict for faster lookup
            postings_dict = {p['doc_id']: p for p in postings}
            term_postings_map[term] = postings_dict
            
            # For AND: intersect document sets
            current_doc_ids = set(postings_dict.keys())
            if doc_ids_with_all_terms is None:
                doc_ids_with_all_terms = current_doc_ids
            else:
                doc_ids_with_all_terms = doc_ids_with_all_terms.intersection(current_doc_ids)
            
            # Early termination if no common documents
            if not doc_ids_with_all_terms:
                query_time = (time.time() - start_time) * 1000
                return [], query_time
        
        # Calculate TF-IDF scores for documents that have all terms
        doc_scores = {}
        
        for doc_id in doc_ids_with_all_terms:
            score = 0.0
            
            for term in query_terms:
                posting = term_postings_map[term][doc_id]
                tf = posting['tf']  # Term frequency (with importance weighting)
                idf = self.idf_scores.get(term, 0)
                
                # TF-IDF score
                tf_idf = tf * idf
                score += tf_idf
            
            doc_scores[doc_id] = score
        
        # Sort by score and get top K
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Convert doc_ids to URLs
        results = [(self.doc_id_to_url[doc_id], score) for doc_id, score in ranked_docs]
        
        query_time = (time.time() - start_time) * 1000
        return results, query_time
    
    def display_results(self, query, results, query_time):
        """Display search results"""
        print("\n" + "="*80)
        print(f"Query: '{query}'")
        print(f"Query time: {query_time:.2f}ms")
        print("="*80)
        
        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results:\n")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. Score: {score:.4f}")
                print(f"   URL: {url}")
                print()
        
        print("="*80)


def run_test_queries():
    """
    Run the required test queries for M2
    """
    # Initialize search engine (will use cached IDF if available)
    search_engine = SearchEngine("index_output")
    
    # Required test queries for M2
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]
    
    print("\n" + "="*80)
    print("MILESTONE 2 - TEST QUERIES")
    print("="*80)
    
    all_results = {}
    
    for query in test_queries:
        results, query_time = search_engine.search(query, top_k=5)
        search_engine.display_results(query, results, query_time)
        all_results[query] = (results, query_time)
        
        # Check if query time meets requirement
        if query_time < 300:
            print(f"✓ Query time OK: {query_time:.2f}ms < 300ms")
        else:
            print(f"⚠ Query time: {query_time:.2f}ms (should be < 300ms)")
        print()
    
    # Generate summary for report
    print("\n" + "="*80)
    print("SUMMARY FOR M2 REPORT")
    print("="*80)
    
    for query in test_queries:
        results, query_time = all_results[query]
        print(f"\nQuery: '{query}'")
        print(f"Query time: {query_time:.2f}ms")
        print("Top 5 URLs:")
        for i, (url, score) in enumerate(results, 1):
            print(f"  {i}. {url}")
    
    return all_results


def interactive_search():
    """
    Interactive search interface for testing
    """
    search_engine = SearchEngine("index_output")
    
    print("\n" + "="*80)
    print("INTERACTIVE SEARCH")
    print("="*80)
    print("Enter queries to search (or 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        query = input("Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        results, query_time = search_engine.search(query, top_k=10)
        search_engine.display_results(query, results, query_time)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Interactive mode
        interactive_search()
    else:
        # Run test queries for M2 report
        run_test_queries()