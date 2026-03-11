import sys
from transformers import pipeline

def create_summary(text, max_len=130, min_len=30):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Generate the summary
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def main():
    print("="*50)
    print("AI TEXT SUMMARIZER - NLP TASK 1")
    print("="*50)
    
    print("\nInstructions: Paste your article below.")
    print("To finish input, press 'Enter' on an empty line.")
    print("-" * 30)

    input_lines = []
    while True:
        line = sys.stdin.readline()
        if line == '\n':  # Break if the user hits Enter on an empty line
            break
        input_lines.append(line)
    
    article_text = "".join(input_lines).strip()

    if len(article_text) < 50:
        print("\n[Error] The text is too short to summarize. Please provide a longer article.")
        return

    print("\nProcessing... (This may take a few seconds)")
    
    # Call the summarization function
    result = create_summary(article_text)

    # Displaying the output 
    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print(result)
    print("="*49)
    print("\nTask Completed Successfully.")

if __name__ == "__main__":
    main()