# Llama 2 Fine-Tuning on Databricks Dolly 15k Subset

The code in this repository fine-tunes a Llama 2 model on a 1000-sample subset of the [Dolly 15k instruction dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) using Supervised Fine-Tuning (SFT) with QLoRA 4-bit precision.

## Overview

1. Clone this repository:

   ```bash
   git clone https://github.com/golkir/llama2-7b-minidatabricks.git
   cd llama2-7b-minidatabricks

   ```

2. Install dependencies

   ```bash
   pip install .
   ```

3. Run the dataset subset creation script which fetches the Dolly 15k dataset and processes it in Llama 2 instruction format.

   ```bash
   python load-databricks.py

   ```

4. Run the fine-tuning script:

   ```bash
   python finetuning.py
   ```

## Acknowledgments

- The Dolly 15k dataset is originally provided by Databricks. [Link to Dolly 15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
- The Llama 2 model can be found in [HuggingFace repository](https://huggingface.co/meta-llama/Llama-2-7b-hf).

## License

This code is licensed under the Apache 2.0 License.
