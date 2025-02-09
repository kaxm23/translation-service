from translator import MarianTranslator

def main():
    # Initialize translator
    translator = MarianTranslator(
        source_lang="en",
        target_lang="ar"
    )
    
    # Single text translation
    text = "Hello, how are you?"
    translated = translator.translate(text)
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    
    # Batch translation
    texts = [
        "Good morning!",
        "How are you doing?",
        "Have a nice day!"
    ]
    translations = translator.translate_batch(texts)
    
    for original, translated in zip(texts, translations):
        print(f"\nOriginal: {original}")
        print(f"Translated: {translated}")
    
    # Print statistics
    print("\nTranslation Statistics:")
    for key, value in translator.get_statistics().items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()