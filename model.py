from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import pandas as pd
import torch

# STEP 1: Define Configuration Variables
BASE_DIR = Path(".")
DATA_RAW = BASE_DIR / "Dataset"  
DATA_PROC = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"

ENCODER_MODEL = "facebook/bart-base"
DECODER_MODEL = "microsoft/DialoGPT-small"

# STEP 2: Load Tokenizers
print("Loading Tokenizers...")
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)

# Fix for DialoGPT padding error
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

# STEP 3: Load Main Network Models
print("Loading Models...")
encoder_model = AutoModelForSeq2SeqLM.from_pretrained(ENCODER_MODEL).float() 

decoder_config = AutoConfig.from_pretrained(DECODER_MODEL)
decoder_config.add_cross_attention = True

decoder_model = AutoModelForCausalLM.from_pretrained(DECODER_MODEL, config=decoder_config).float() 

# STEP 4: Offline Data Mocking (No Internet or File Required!)
def load_and_preview_data():
    print("\n[Offline Mode] Creating a clean mock dataset inside Python memory...")
    
    mock_data = {
        "conv_id": ["hit1_0", "hit2_1"],
        "context": ["sentimental", "afraid"],
        "prompt": ["I remember feeling so lonely.", "I heard a loud noise downstairs!"],
        "utterance": ["I remember feeling so lonely yesterday.", "I heard a loud noise downstairs last night!"]
    }
    
    df = pd.DataFrame(mock_data)
    
    print("\n--- Dataset Overview ---")
    print(f"Total rows found: {len(df)}")
    print("\nAvailable Data Columns:", list(df.columns))
    print("\nFirst 2 rows of data:")
    print(df.head(2))
    return df

# STEP 5: Tokenize Text Data for the Models
def preprocess_dataset(df):
    print("\nTokenizing sentences into model inputs...")
    
    contexts = df["prompt"].tolist()
    utterances = df["utterance"].tolist()
    
    encoder_inputs = encoder_tokenizer(
        contexts, 
        max_length=128, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    decoder_inputs = decoder_tokenizer(
        utterances, 
        max_length=128, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    print("Tokenization completed successfully!")
    print(f"Encoder Input Shape: {encoder_inputs['input_ids'].shape}")
    print(f"Decoder Input Shape: {decoder_inputs['input_ids'].shape}")
    
    return encoder_inputs, decoder_inputs


def run_model_inference(user_input_string):
    """
    Takes a text string from the web UI, processes it through the encoder,
    passes the representations to the decoder, and returns the final decoded string.
    """
    with torch.no_grad():
        # 1. Tokenize user input for the encoder
        encoder_inputs = encoder_tokenizer(
            user_input_string, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # 2. Extract hidden states from the encoder model
        encoder_outputs = encoder_model.base_model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"]
        )
        hidden_states = encoder_outputs.last_hidden_state

        # 3. Create a starting token for DialoGPT text generation
        decoder_start_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

        # 4. Generate response tokens
        predicted_token_ids = decoder_model.generate(
            input_ids=decoder_input_ids,  
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
            do_sample=True,     
            top_k=50,           
            top_p=0.95          
        )
        
        # 5. Decode the tokens back into readable words with warning fix added here:
        final_text = decoder_tokenizer.decode(
            predicted_token_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
    return final_text


# This block ensures your code can still be tested directly
if __name__ == "__main__":
    print("\nTesting function locally...")
    test_result = run_model_inference("Your sample test sentence here")
    print("\nOutput:", test_result)