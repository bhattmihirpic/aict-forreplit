# plagiarism_detector.py

import os, re, pickle
from text_processor import extract_text_from_file, clean_up_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports with graceful fallbacks
try:
    import faiss
    import torch.multiprocessing as mp
    from sentence_transformers import SentenceTransformer
    
    # Multiprocessing & OpenMP settings
    mp.set_sharing_strategy('file_system')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    faiss.omp_set_num_threads(1)
    
    SEMANTIC_AVAILABLE = True
    print("✅ Semantic plagiarism detection available")
except ImportError as e:
    print(f"⚠️ Advanced ML libraries not available: {e}")
    print("⚠️ Using TF-IDF fallback for plagiarism detection")
    SEMANTIC_AVAILABLE = False

class PlagiarismDetector:
    """Plagiarism checker with adjustable semantic threshold and TF-IDF fallback."""

    def __init__(
        self,
        ref_folder="reference_texts",
        semantic_threshold=0.6,
        tfidf_refs=None
    ):
        """
        semantic_threshold: cosine similarity cutoff for FAISS matches.
        tfidf_refs: list of fallback reference strings for TF-IDF detection.
        """
        self.threshold = semantic_threshold
        self.tfidf_refs = tfidf_refs or [\
            "Academic integrity is important in education. Students should do their own work.",\
            "The scientific method involves observation, hypothesis, and experimentation.",\
            "Climate change is caused by human activities and greenhouse gases.",\
            "Technology has changed how we communicate and learn."\
        ]
        self.text_analyzer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1,3))

        # Build or load semantic index if available
        global SEMANTIC_AVAILABLE
        if SEMANTIC_AVAILABLE:
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.index_path = "plag_index.faiss"
                self.mapping_path = "mapping.pkl"
                self.ref_folder = ref_folder

                if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
                    self.index = faiss.read_index(self.index_path)
                    with open(self.mapping_path, "rb") as f:
                        self.mapping = pickle.load(f)
                else:
                    self.index, self.mapping = self.build_index(ref_folder)
                    if self.mapping:  # Only save if we have data
                        faiss.write_index(self.index, self.index_path)
                        with open(self.mapping_path, "wb") as f:
                            pickle.dump(self.mapping, f)
                print("✅ Plagiarism detection ready (semantic mode)")
            except Exception as e:
                print(f"⚠️ Semantic detection failed: {e}")
                print("⚠️ Falling back to TF-IDF mode")
                SEMANTIC_AVAILABLE = False
                self.model = None
                self.mapping = []
        else:
            self.model = None
            self.mapping = []
            print("✅ Plagiarism detection ready (TF-IDF mode)")

    def build_index(self, folder):
        texts, mapping = [], []
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                path = os.path.join(folder, fn)
                try: 
                    raw = extract_text_from_file(path)
                except: 
                    continue
                clean = clean_up_text(raw)
                words = clean.split()
                for i in range(0, max(len(words)-150,1), 150):
                    chunk = words[i:i+200]
                    if len(chunk)>=50:
                        txt = " ".join(chunk)
                        texts.append(txt)
                        mapping.append((fn, txt))
        
        if not texts or not SEMANTIC_AVAILABLE:
            if SEMANTIC_AVAILABLE:
                dim = self.model.get_sentence_embedding_dimension()
                return faiss.IndexFlatIP(dim), []
            else:
                return None, []
        
        try:
            embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, num_workers=0)
            faiss.normalize_L2(embs)
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            return idx, mapping
        except Exception as e:
            print(f"⚠️ Index building failed: {e}")
            return None, []

    def _tfidf_similarity(self, text):
        t1 = re.sub(r'\s+', ' ', text.lower()).strip()
        best = 0.0
        for ref in self.tfidf_refs:
            t2 = re.sub(r'\s+', ' ', ref.lower()).strip()
            vecs = self.text_analyzer.fit_transform([t1, t2])
            sim = cosine_similarity(vecs)[0,1] * 100
            best = max(best, sim)
        return round(best,1)

    def check_for_plagiarism(self, submitted_text, top_k=5):
        clean = clean_up_text(submitted_text)
        words = clean.split()
        
        # If no semantic capability or no refs, use TF-IDF
        if not SEMANTIC_AVAILABLE or len(self.mapping)==0 or len(words)<50:
            score = self._tfidf_similarity(submitted_text)
            # Use threshold for confidence determination in TF-IDF mode
            threshold_percent = self.threshold * 100  # Convert 0.6 to 60%
            high_threshold = max(threshold_percent + 20, 80)  # At least 80%, or threshold + 20%
            medium_threshold = max(threshold_percent, 50)     # At least 50%, or threshold value
            
            conf = 'high' if score > high_threshold else 'medium' if score > medium_threshold else 'low'
            
            # Create suspicious parts if score exceeds threshold
            suspicious_parts = []
            if score > threshold_percent:
                suspicious_parts.append({
                    "submitted_chunk": submitted_text[:200] + "..." if len(submitted_text) > 200 else submitted_text,
                    "reference_file": "built-in references",
                    "reference_chunk": "Academic reference content",
                    "similarity": round(score/100, 3)
                })
            
            return {
                "overall_score": score,
                "confidence": conf,
                "suspicious_parts": suspicious_parts,
                "message": f"TF-IDF detection (threshold: {threshold_percent:.0f}%)"
            }
        
        # Semantic matching
        try:
            chunks = []
            for i in range(0, max(len(words)-150,1), 150):
                c = words[i:i+200]
                if len(c)>=50: 
                    chunks.append(" ".join(c))
            
            if not chunks:
                return {
                    "overall_score": 0.0,
                    "confidence": 'low',
                    "suspicious_parts": [],
                    "message": "Text too short for analysis"
                }
            
            embs = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, num_workers=0)
            faiss.normalize_L2(embs)
            hits, parts = set(), []
            
            for idx, emb in enumerate(embs):
                D, I = self.index.search(emb.reshape(1,-1), top_k)
                print(f"Chunk {idx} sims: {D[0].tolist()}")
                for score, ref_i in zip(D[0], I[0]):
                    if score >= self.threshold:
                        fn, txt = self.mapping[ref_i]
                        parts.append({
                            "submitted_chunk": chunks[idx],
                            "reference_file": fn,
                            "reference_chunk": txt,
                            "similarity": round(float(score),3)
                        })
                        hits.add(idx)
                        
            overall = 100.0 * len(hits) / len(chunks) if chunks else 0.0
            conf = 'high' if overall>80 else 'medium' if overall>50 else 'low'
            return {
                "overall_score": round(overall,1),
                "confidence": conf,
                "suspicious_parts": parts,
                "message": "Semantic match results"
            }
        except Exception as e:
            print(f"⚠️ Semantic analysis failed: {e}")
            # Fallback to TF-IDF
            score = self._tfidf_similarity(submitted_text)
            # Use threshold for confidence determination
            threshold_percent = self.threshold * 100
            high_threshold = max(threshold_percent + 20, 80)
            medium_threshold = max(threshold_percent, 50)
            
            conf = 'high' if score > high_threshold else 'medium' if score > medium_threshold else 'low'
            
            # Create suspicious parts if score exceeds threshold
            suspicious_parts = []
            if score > threshold_percent:
                suspicious_parts.append({
                    "submitted_chunk": submitted_text[:200] + "..." if len(submitted_text) > 200 else submitted_text,
                    "reference_file": "built-in references", 
                    "reference_chunk": "Academic reference content",
                    "similarity": round(score/100, 3)
                })
                
            return {
                "overall_score": score,
                "confidence": conf,
                "suspicious_parts": suspicious_parts,
                "message": f"TF-IDF fallback (threshold: {threshold_percent:.0f}%)"
            }