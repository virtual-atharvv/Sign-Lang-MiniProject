import os

class ISLGrammarCorrector:
    """
    Translates ordered ISL gloss lists into fluent English sentences using fine-tuned T5-small.
    """
    def __init__(self, model_dir: str = "models/grammar_correction/t5_isl_checkpoint"):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_dir):
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
                print(f"Loaded ISLGrammarCorrector from {self.model_dir}")
            except Exception as e:
                print(f"Failed to load fine-tuned T5 model: {e}")
                self.model = None
        else:
            print(f"Fine-tuned checkpoint not found at {self.model_dir}. Rule-based fallback active.")

    def translate(self, glosses: list[str]) -> str:
        """
        Input: list of gloss strings e.g. ["HOME", "I", "GO"]
        Output: English sentence string e.g. "I am going home."
        """
        if not glosses:
            return ""

        gloss_str = " ".join(glosses).upper()

        if self.model and self.tokenizer:
            input_text = f"translate ISL gloss to English: {gloss_str}"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return sentence

        # Rule-based linguistic fallback if weights aren't loaded yet
        return self._rule_based_fallback(glosses)

    def _rule_based_fallback(self, glosses: list[str]) -> str:
        # Basic heuristic mapping for common patterns
        g = [x.upper() for x in glosses]
        
        if "HOME" in g and "I" in g and "GO" in g:
            return "I am going home."
        if "WATER" in g and "WANT" in g:
            return "I want water."
        if "HELP" in g:
            return "I need help."
        if "NAME" in g and "WHAT" in g:
            return "What is your name?"
        if "THANK" in g or "THANKS" in g:
            return "Thank you very much."

        # Default clean capitalization fallback
        return " ".join([word.capitalize() for word in glosses]) + "."
