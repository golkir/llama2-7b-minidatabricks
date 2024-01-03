from datasets import load_dataset

dataset_name = "databricks/databricks-dolly-15k"

# Load the dataset
dataset = load_dataset(dataset_name)

# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=55).select(range(1000))

# Define a function to transform the data


def transform_sample(sample):
    instruction = sample["instruction"]
    response = sample["response"]
    formatted_prompt = f'<s>[INST] {instruction} [/INST] {response} </s>'

    return {'text': formatted_prompt}


# Apply the transformation
transformed_dataset = dataset.map(transform_sample, remove_columns=[
                                  "context", "instruction", "response", "category"])


transformed_dataset.push_to_hub("databricks-dolly-llama2-1k")
