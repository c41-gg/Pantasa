import logging
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from Training import train_model_with_pos_tags  # Import your training function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    logging.info("Starting the process...")

    # Load custom tokenizer (with the POS tag vocabulary added)
    logging.info("Loading the custom tokenizer...")
    pos_tokenizer = RobertaTokenizerFast.from_pretrained(
        "/content/Pantasa/model/pos_tokenizer",  # Path to your custom tokenizer with POS tags
        truncation=True,
        padding="max_length",
        max_length=1000,
        add_prefix_space=True,  # Ensures the tokenizer works with pretokenized inputs
        use_cache= False
    )
    logging.info("Tokenizer loaded successfully.")

    # Load the pre-trained model
    logging.info("Loading the pre-trained model...")
    model = RobertaForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-base")
    logging.info("Model loaded successfully.")

    # Resize the model's token embeddings to match the tokenizer's vocabulary size
    logging.info("Resizing model token embeddings to match tokenizer vocabulary...")
    model.resize_token_embeddings(len(pos_tokenizer))  # Resize embeddings based on tokenizer's vocab size

    logging.info("Model loaded and resized successfully with custom tokens.")

    # Path to the CSV input file
    csv_input = "/content/Pantasa/rules/MPoSM/pos_tags_output.csv"

    # Path to the output CSV where tokenized data will be saved
    output_csv = "/content/Pantasa/rules/MPoSM/tokenized_output.csv"

        # Define the directory for saving checkpoints
    checkpoint_dir = "./results"

    # Check if a checkpoint exists in the output directory
    checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint")]
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            logging.info(f"Resuming from checkpoint: {checkpoint_path}") 

    logging.info("Starting training with file: %s", csv_input)

    # Start training and pass the checkpoint path
    train_model_with_pos_tags(csv_input, pos_tokenizer, model, output_csv, checkpoint_path)

    logging.info("Training completed.")