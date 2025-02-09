from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

class ArabicTranslator:
    def __init__(self, target_lang="en"):
        """
        Initialize the translator with the desired target language
        
        Args:
            target_lang (str): Target language code (default: "en" for English)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = f"Helsinki-NLP/opus-mt-ar-{target_lang}"
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            print(f"Error loading translation model: {str(e)}")
            raise

    def translate(self, text_list):
        """
        Translate a list of Arabic text segments
        
        Args:
            text_list (list): List of Arabic text segments to translate
            
        Returns:
            list: List of translated text segments
        """
        translations = []
        
        try:
            for text in tqdm(text_list, desc="Translating"):
                # Skip empty text
                if not text.strip():
                    translations.append("")
                    continue
                
                # Tokenize the text
                encoded = self.tokenizer(text, return_tensors="pt", padding=True)
                
                # Move input to the same device as the model
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Generate translation
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
                # Decode the generated tokens
                translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                translations.append(translated)
                
            return translations
            
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return []