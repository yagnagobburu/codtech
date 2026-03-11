from transformers import pipeline
import sys

def generate_paragraph(prompt_text, length=150):
    
    try:
       
        generator = pipeline('text-generation', model='gpt2')
        
        print(f"\n-> Generating content for: \"{prompt_text}\"")
        
        
        output = generator(prompt_text, 
                           max_length=length, 
                           num_return_sequences=1, 
                           temperature=0.8, 
                           truncation=True)
        
        return output[0]['generated_text']
        
    except Exception as e:
        return f"[Error] Could not generate text: {e}"

def main():
    print("="*50)
    print("AI TEXT GENERATOR - TASK 4")
    print("="*50)
    
    print("\nInstructions: Enter a topic or a starting sentence.")
    print("Type 'exit' to close the program.")
    
    while True:
        user_prompt = input("\nYour Prompt: ").strip()
        
        if user_prompt.lower() == 'exit':
            print("Closing the generator. Goodbye!")
            break
            
        if not user_prompt:
            print("[System] Please enter a valid prompt.")
            continue

        print("-> Processing... (AI is thinking)")
        paragraph = generate_paragraph(user_prompt)

        print("\n" + "-"*20 + " GENERATED PARAGRAPH " + "-"*20)
        print(paragraph)
        print("-" * 61)

if __name__ == "__main__":
    main()