# Text Summarization Using BART Transformer Model

This project demonstrates text summarization using the BART (Bidirectional and Auto-Regressive Transformers) model, implemented in Python with Hugging Face Transformers and the DialogSum dataset. The notebook covers both zero-shot summarization (no fine-tuning) and supervised fine-tuning on a dialogue dataset. 

> **Note:**  
> My previous GitHub account was unexpectedly suspended. This project was originally created earlier and has been re-uploaded here. All work was done gradually over time, and original commit history has been preserved where possible.

---

## Project Overview

Text summarization is the process of generating a concise and coherent summary of a longer text. This project focuses on conversational/dialogue summarization—a challenging variant due to the informal, turn-based nature of dialogues.

The BART model, a state-of-the-art sequence-to-sequence transformer, is first used as a pre-trained summarizer and then fine-tuned on the `knkarthick/dialogsum` dataset for improved, context-specific results.

---

## Workflow

### 1. Setup and Data Loading

- Install and import required libraries (`datasets`, `transformers`).
- Load the [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum), which contains multi-turn dialogues and their human-written summaries.

### 2. Exploratory Data Access

- View the structure and sample content of the dataset.
- Example:
  - **Dialogue sample:**  
    ```
    #Person1#: Hello Mrs. Parker, how have you been?
    #Person2#: Hello Dr. Peters. Just fine thank you. Ricky and I are here for his vaccines.
    ...
    ```
  - **Summary sample:**  
    `Mrs Parker takes Ricky for his vaccines. Dr. Peters checks the record and then gives Ricky a vaccine.`

### 3. Text Summarization Without Fine-Tuning

- Use Hugging Face's `pipeline` with `facebook/bart-large-cnn` to generate summaries on raw dialogue samples.
- Demonstrates quick, out-of-the-box summarization capability.

### 4. Fine-Tuning BART on DialogSum

- Load `facebook/bart-base`.
- Tokenize and preprocess the dataset for supervised learning.
- Use Hugging Face `Trainer` and `TrainingArguments` to fine-tune the BART model for 2 epochs.
- Log training and evaluation with Weights & Biases (wandb).

### 5. Evaluation & Saving

- Evaluate model performance on the test split.
- Save the fine-tuned model and tokenizer for future use.

### 6. Summarizing Custom Texts

- Illustrate how to load the saved model and summarize custom input (e.g., a blog post or new dialogue).

---

## Example: Summarizing a Blog Post

After training and saving the model, you can summarize custom texts:

```python
def summarize(blog_post):
    inputs = tokenizer(blog_post, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

blog_post = """As Yogi Berra famously said, it’s tough to make predictions, especially about the future. ..."""
print("Summary:", summarize(blog_post))
```

---

## Requirements

- Python 3.8+
- `transformers`
- `datasets`
- `torch`
- `wandb` (for experiment tracking, optional)
- GPU recommended for training

---

## Usage

1. Clone this repository.
2. Install requirements.
3. Open and execute `Text_Summarization (1).ipynb` in Jupyter/Colab.
4. Follow the notebook sections for:
   - Out-of-the-box summarization
   - Fine-tuning and evaluation
   - Using the fine-tuned model

---

## Dataset

- [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum):  
  - Features: `id`, `dialogue`, `summary`, `topic`
  - Splits: train/validation/test

---

## Results

- **Zero-shot BART:** Quickly summarizes dialogue but may miss context-specific details.
- **Fine-tuned BART:** Produces more context-aware, concise summaries for conversational data.
- **Custom Inference:** The model can be used to summarize new dialogues or articles after fine-tuning.

---

## License

This project is for educational and research purposes only.  
Please check model and dataset licenses if using for commercial applications.

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)

---

**If you find this useful, please star the repository!**
