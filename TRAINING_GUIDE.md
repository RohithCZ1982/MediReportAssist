# LLM Training Guide for MediReportAssist

This guide covers different approaches to train and improve the LLM in your RAG system.

## Overview

Your system uses:
- **LLM**: Ollama with llama3.2 (for answer generation)
- **Embeddings**: Sentence Transformers `all-MiniLM-L6-v2` (for document retrieval)
- **Vector DB**: ChromaDB (for storing document chunks)

## Training Options (Easiest to Hardest)

### Option 1: Custom Ollama Model with Medical System Prompt (Easiest) ⭐ Recommended

**What it does**: Creates a custom Ollama model with a medical-focused system prompt that improves responses for discharge summaries.

**Time**: 5-10 minutes  
**Resources**: Minimal (just text editing)

#### Steps:

1. **Create a Modelfile**:

Create a file named `Modelfile` in your project root:

```dockerfile
FROM llama3.2

SYSTEM """You are a specialized medical assistant focused on helping patients understand their discharge instructions. Your role is to:

1. Interpret medical discharge summaries accurately
2. Translate complex medical terminology into simple, patient-friendly language
3. Provide clear, actionable instructions about medications, diet, activities, and follow-up care
4. Identify and highlight important warnings, contraindications, and emergency signs
5. Answer questions with empathy and reassurance while maintaining medical accuracy

Guidelines:
- Always base answers on the provided discharge summary context
- Use simple language that patients without medical training can understand
- Include specific details: medication names, dosages, timing, and frequencies
- Format information clearly with bullet points or numbered lists when appropriate
- If information is missing from the context, clearly state this
- Never provide medical advice beyond what's in the discharge summary
- Encourage patients to contact their healthcare provider for clarification when needed

Response Style:
- Professional yet warm and reassuring
- Clear and concise
- Well-structured with proper formatting
- Focused on actionable patient instructions
"""
```

2. **Create the custom model**:

```bash
ollama create mediassist -f Modelfile
```

3. **Test the model**:

```bash
ollama run mediassist "What should I know about discharge instructions?"
```

4. **Update your application**:

Set the environment variable to use your custom model:

**Windows (PowerShell)**:
```powershell
$env:LLM_MODEL="mediassist"
```

**Windows (Command Prompt)**:
```cmd
set LLM_MODEL=mediassist
```

**Linux/Mac**:
```bash
export LLM_MODEL=mediassist
```

5. **Restart your application**:

```bash
python app.py
```

**Benefits**:
- ✅ Immediate improvement in medical domain responses
- ✅ No training data needed
- ✅ No GPU required
- ✅ Fast to implement

---

### Option 2: Fine-tune Embedding Model for Medical Domain (Moderate)

**What it does**: Trains the embedding model on medical text to improve document retrieval accuracy.

**Time**: 2-4 hours  
**Resources**: GPU recommended (but can use CPU)

#### Steps:

1. **Install additional dependencies**:

```bash
pip install datasets transformers torch
```

2. **Create training script** (`train_embeddings.py`):

```python
"""
Fine-tune Sentence Transformer for medical domain
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import json
from pathlib import Path

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
# You can use your discharge summaries as training data
training_data = []

# Option 1: Use your existing discharge summaries
upload_dir = Path("uploads")
for file_path in upload_dir.glob("*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        # Create pairs of similar sentences from the same document
        sentences = text.split('.')
        for i in range(len(sentences) - 1):
            if len(sentences[i].strip()) > 20 and len(sentences[i+1].strip()) > 20:
                training_data.append(InputExample(
                    texts=[sentences[i].strip(), sentences[i+1].strip()],
                    label=0.8  # High similarity (same document)
                ))

# Option 2: Use medical domain datasets
# Download medical Q&A pairs or medical text pairs
# Example: Medical Question Answering datasets from Hugging Face

# Create data loader
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='./medical-embeddings-model'
)

print("Model fine-tuned and saved to ./medical-embeddings-model")
```

3. **Run training**:

```bash
python train_embeddings.py
```

4. **Update RAG system to use fine-tuned model**:

In `rag_system.py`, change:

```python
def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
    # Change to:
    # Check if fine-tuned model exists
    if Path("./medical-embeddings-model").exists():
        embedding_model = "./medical-embeddings-model"
    
    self.embedding_model = SentenceTransformer(embedding_model)
```

**Benefits**:
- ✅ Better document retrieval for medical queries
- ✅ Improved semantic understanding of medical terminology
- ✅ Can use your existing discharge summaries as training data

**Note**: For better results, use medical domain datasets from Hugging Face or create question-answer pairs from your discharge summaries.

---

### Option 3: Fine-tune Base LLM with Medical Data (Advanced)

**What it does**: Fine-tunes the llama3.2 model on medical discharge summary Q&A pairs.

**Time**: Several hours to days  
**Resources**: GPU with 16GB+ VRAM (or use cloud services)

#### Using Ollama's Fine-tuning (Recommended for Ollama users):

Ollama supports creating fine-tuned models using a Modelfile with training examples.

1. **Prepare training data**:

Create a JSONL file with Q&A pairs from discharge summaries:

```json
{"instruction": "Extract medication information", "input": "Discharge summary mentions: Patient should take amoxicillin 500mg twice daily for 7 days.", "output": "Medication: Amoxicillin\nDosage: 500mg\nFrequency: Twice daily\nDuration: 7 days"}
{"instruction": "Explain dietary restrictions", "input": "Patient should avoid spicy foods and alcohol for 2 weeks.", "output": "Dietary Restrictions:\n- Avoid spicy foods\n- Avoid alcohol\n- Duration: 2 weeks"}
```

2. **Create enhanced Modelfile**:

```dockerfile
FROM llama3.2

SYSTEM """You are a medical assistant..."""

# Add training examples (few-shot learning)
TEMPLATE """{{ if .System }}System: {{ .System }}

{{ end }}{{ if .Prompt }}User: {{ .Prompt }}

{{ end }}Assistant: {{ .Response }}"""
```

3. **Use parameter-efficient fine-tuning**:

For full fine-tuning, you'll need to use tools like:
- **Unsloth** (fast and efficient)
- **LoRA** (Low-Rank Adaptation)
- **QLoRA** (Quantized LoRA)

#### Using Unsloth (Easiest for full fine-tuning):

1. **Install Unsloth**:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

2. **Create training script** (`train_llm.py`):

```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Prepare training data
from datasets import load_dataset
dataset = load_dataset("your_medical_qa_dataset", split="train")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

trainer.train()

# Save model
model.save_pretrained("medical-llama3.2")
```

3. **Convert to Ollama format**:

After training, convert the model to Ollama format or use it directly with transformers.

**Benefits**:
- ✅ Best performance for medical domain
- ✅ Model understands medical context better
- ✅ More accurate responses

**Challenges**:
- ❌ Requires significant computational resources
- ❌ Needs large, high-quality training dataset
- ❌ Time-consuming

---

### Option 4: Improve RAG with Better Prompts (Quick Win)

**What it does**: Optimizes the prompt used in RAG system without training.

**Time**: 10-15 minutes  
**Resources**: None

#### Steps:

1. **Edit `rag_system.py`**:

Find the `generate_answer` method and improve the prompt:

```python
# Create prompt for medical assistant
prompt = f"""You are a specialized medical assistant helping patients understand their discharge instructions.

CONTEXT FROM DISCHARGE SUMMARY:
{context_text}

PATIENT QUESTION: {query}

INSTRUCTIONS:
1. Answer ONLY based on the provided discharge summary context
2. Use simple, clear language that patients can understand
3. If the question asks about:
   - Medications: Include name, dosage, frequency, timing, and duration
   - Diet: List specific foods to eat/avoid and duration
   - Activities: Specify what's allowed/prohibited and for how long
   - Follow-up: Provide appointment details, dates, and contact information
   - Symptoms/Warnings: Clearly state what to watch for and when to seek help
4. Format your response with:
   - Clear headings or bullet points
   - Specific numbers, dates, and dosages
   - Action items in a numbered list
5. If information is not in the context, say: "This information is not mentioned in your discharge summary. Please contact your healthcare provider."
6. End with reassurance and encourage contacting healthcare provider for clarification

ANSWER:"""
```

2. **Test and iterate**:

Try different prompt variations and see which works best for your use case.

**Benefits**:
- ✅ Immediate improvement
- ✅ No training needed
- ✅ Easy to experiment with

---

## Recommended Approach

**For most users**: Start with **Option 1** (Custom Ollama Model) + **Option 4** (Better Prompts). This gives you 80% of the benefit with 20% of the effort.

**For better retrieval**: Add **Option 2** (Fine-tune Embeddings) if you have many discharge summaries.

**For maximum performance**: Use **Option 3** (Fine-tune LLM) if you have:
- Large dataset of medical Q&A pairs (1000+ examples)
- GPU with 16GB+ VRAM
- Time and resources for training

---

## Creating Training Data from Your Discharge Summaries

You can create training data from your existing discharge summaries:

1. **Extract Q&A pairs**:

```python
# create_training_data.py
from document_processor import DocumentProcessor
import json

doc_processor = DocumentProcessor()
training_pairs = []

# Process each discharge summary
for file_path in Path("uploads").glob("*.txt"):
    text = doc_processor.process_document(str(file_path))
    
    # Extract medication information
    # Create questions like "What medications should I take?"
    # Extract answers from the text
    
    # Extract dietary information
    # Extract activity restrictions
    # etc.
    
    training_pairs.append({
        "instruction": "Answer medical questions about discharge instructions",
        "input": f"Question: {question}\nContext: {text}",
        "output": answer
    })

# Save to JSONL
with open("training_data.jsonl", "w") as f:
    for pair in training_pairs:
        f.write(json.dumps(pair) + "\n")
```

---

## Testing Your Trained Model

After training, test with:

```python
# test_model.py
from rag_system import RAGSystem

rag = RAGSystem()

# Test queries
test_queries = [
    "When should I take my medication?",
    "What foods should I avoid?",
    "When is my follow-up appointment?",
]

for query in test_queries:
    context = rag.retrieve_context("test_doc_id", query)
    answer = rag.generate_answer(query, context)
    print(f"Q: {query}")
    print(f"A: {answer}\n")
```

---

## Resources

- **Ollama Modelfile Docs**: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- **Sentence Transformers Training**: https://www.sbert.net/docs/training/overview.html
- **Unsloth (Fast LLM Fine-tuning)**: https://github.com/unslothai/unsloth
- **Medical Datasets**: 
  - Hugging Face: https://huggingface.co/datasets?search=medical
  - MIMIC (requires approval): https://mimic.mit.edu/

---

## Next Steps

1. ✅ Start with Option 1 (Custom Ollama Model)
2. ✅ Improve prompts (Option 4)
3. ⏭️ Consider Option 2 if you have many documents
4. ⏭️ Option 3 only if you need maximum performance

Would you like me to help you implement any of these options?

