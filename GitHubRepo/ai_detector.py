# ai_detector.py

# Optional imports with graceful fallbacks
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available")
except ImportError:
    print("⚠️ Transformers library not available - using pattern-based detection")
    TRANSFORMERS_AVAILABLE = False

class AIDetector:
    """AI content detective with adjustable sensitivity."""

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=512,
        stride=50,
        ai_weight=0.7,
        pattern_weight=0.3,
        high_threshold=80,
        medium_threshold=60
    ):
        """
        ai_weight/pattern_weight: weights for smart vs pattern analysis.
        high_threshold/medium_threshold: score cutoffs for confidence.
        """
        self.max_tokens = max_tokens
        self.stride = stride
        self.ai_weight = ai_weight
        self.pattern_weight = pattern_weight
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        self.ai_model = None
        self.sklearn_model = None
        self.model_working = False

        # Try to load local sklearn model first
        try:
            import pickle
            import os
            if os.path.exists('models/ai_detection_model.pkl'):
                with open('models/ai_detection_model.pkl', 'rb') as f:
                    self.sklearn_model = pickle.load(f)
                self.model_working = True
                print("✅ Local AI detection model loaded successfully!")
            else:
                print("⚠️ Local model not found, trying transformers...")
        except Exception as e:
            print(f"⚠️ Couldn't load local model: {e}")

        # Fallback to transformers if available and local model failed
        if not self.model_working and TRANSFORMERS_AVAILABLE:
            try:
                print("🤖 Loading transformer AI detection model...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.ai_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=self.tokenizer,
                    device="cpu",
                    return_all_scores=True
                )
                self.model_working = True
                print("✅ Transformer AI detection model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Couldn't load transformer model: {e}")
                print("Will use pattern-based detection instead")
        
        if not self.model_working:
            print("⚠️ No AI models available - using pattern-based detection")
            print("✅ AI detector ready (pattern-based mode)")

    def look_for_ai_patterns(self, text):
        if not text:
            return {'ai_score': 0.0, 'clues_found': []}

        text_lower = text.lower()
        ai_score = 0.0
        clues = []

        ai_phrases = {
            'as an ai': 30,
            'furthermore': 8,
            'moreover': 8,
            'in conclusion': 10,
            'it is important to note': 15,
            'however, it is worth noting': 12,
            'on the other hand': 8,
            'additionally': 6,
            'consequently': 8
        }

        for phrase, points in ai_phrases.items():
            count = text_lower.count(phrase)
            if count > 0:
                ai_score += points * count
                clues.append(f"Uses '{phrase}' {count} time(s) - AI tools love this phrase!")

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            avg = sum(lengths) / len(lengths)
            if avg > 20 and all(abs(l - avg) < 5 for l in lengths):
                ai_score += 15
                clues.append(f"All sentences are suspiciously similar length (avg: {avg:.1f} words)")

        transition_words = ['however', 'furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless']
        transition_count = sum(text_lower.count(w) for w in transition_words)
        if transition_count > len(sentences) * 0.3:
            ai_score += 12
            clues.append(f"Uses too many fancy transition words ({transition_count} found)")

        return {'ai_score': min(100.0, ai_score), 'clues_found': clues}

    def use_smart_ai_model(self, text):
        """Use the smart AI detection model if available."""
        if not self.model_working or not text:
            return None

        try:
            # Use sklearn model if available
            if self.sklearn_model:
                ai_prob = self.sklearn_model.predict_proba([text])[0][1] * 100
                return {
                    'ai_score': round(ai_prob, 1),
                    'chunk_details': [{'chunk_index': 0, 'ai_probability': round(ai_prob, 1)}],
                    'chunks_analyzed': 1
                }
            
            # Fallback to transformer model
            elif self.ai_model:
                encoding = self.tokenizer(
                    text,
                    return_overflowing_tokens=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    stride=self.stride,
                    padding="max_length",
                    return_tensors="pt"
                )

                total_score = 0.0
                chunk_details = []

                for i in range(encoding.input_ids.size(0)):
                    input_ids = encoding.input_ids[i].unsqueeze(0)
                    attention_mask = encoding.attention_mask[i].unsqueeze(0)

                    outputs = self.ai_model.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Detach before converting to NumPy
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

                    ai_prob = float(probs[1]) * 100
                    total_score += ai_prob

                    chunk_details.append({
                        'chunk_index': i,
                        'ai_probability': round(ai_prob, 1)
                    })

                avg_score = total_score / encoding.input_ids.size(0)

                return {
                    'ai_score': round(avg_score, 1),
                    'chunk_details': chunk_details,
                    'chunks_analyzed': encoding.input_ids.size(0)
                }

        except Exception as e:
            print(f"Error using AI model: {e}")
            return None

    def detect_ai_content(self, text):
        """Main function to detect if AI wrote this text."""
        if not text or len(text.strip()) < 50:
            return {
                'ai_probability': 0.0,
                'confidence': 'low',
                'details': 'Text too short to analyze',
                'clues_found': []
            }

        smart_result = self.use_smart_ai_model(text)
        pattern_result = self.look_for_ai_patterns(text)

        if smart_result:
            final_score = (smart_result['ai_score'] * self.ai_weight) + (pattern_result['ai_score'] * self.pattern_weight)
            method_used = "Smart AI model + Pattern analysis"
        else:
            final_score = pattern_result['ai_score']
            method_used = "Pattern analysis only"

        if final_score > self.high_threshold:
            confidence = 'high'
        elif final_score > self.medium_threshold:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'ai_probability': round(final_score, 1),
            'confidence': confidence,
            'details': f'Analysis completed using: {method_used}',
            'clues_found': pattern_result['clues_found'],
            'smart_model_working': self.model_working
        }