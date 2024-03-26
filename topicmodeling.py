# Create your own environment using poetry


# Install BERTopic and other dependencies
pip install -q bertopic==0.7.1
pip install -q transformers==4.12.0
pip install -q sentence-transformers==2.1.0
pip install -q spacy==3.2.1
python -m spacy download en_core_web_sm

from datasets import load_dataset
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your dataset
ds = load_dataset("OpenAssistant/oasst2")
train = ds['train']

# Filter and sample the dataset
df = train.to_pandas()
df = df[df['lang'] == 'en'] # Select English messages
df = df[df['rank'] == 1] # Select top-ranked answers
df = df.sample(1000, random_state=42) # Get a sample for the example
docs = list(df.text)

# Initialize the LLM model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a BERTopic model with the LLM for fine-tuning
topic_model = BERTopic(
    language="english",
    verbose=True,
    calculate_probabilities=True,
    min_topic_size=10,
    n_gram_range=(1, 2),
    low_memory=True,
    seed_topic_list=["example topic"],
    embedding_model=model,
    tokenizer=tokenizer
)

# Fit the model to your documents
topics, probabilities = topic_model.fit_transform(docs)

# Print the topics
topic_model.get_topic_info()