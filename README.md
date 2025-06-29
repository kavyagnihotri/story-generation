# story-generation

**Task Description**: Train a model to generate a continuation of a story given an initial prompt or context. This task is fun and creative because it allows the model to produce imaginative and coherent narratives, blending creativity with linguistic understanding. It’s engaging for exploring how models handle open-ended, creative text generation while maintaining coherence and context.

**Huggingface Dataset**: `writingprompts`  
- **Description**: The `writingprompts` dataset contains short story prompts paired with human-written story continuations from the Reddit r/WritingPrompts community. It’s ideal for creative text generation tasks, as it includes diverse prompts (e.g., fantasy, sci-fi, horror) and corresponding stories.

#### Steps to Approach the Task

1. **Read and Preprocess the Dataset**:
   - **Loading**: Use `from datasets import load_dataset; dataset = load_dataset("writingprompts")` to load the dataset.
   - **Preprocessing**:
     - Filter out prompts or stories that are too short (e.g., <50 tokens) to ensure meaningful inputs and outputs.
     - Tokenize the text using a tokenizer compatible with your chosen models (e.g., BART or T5 tokenizer from Huggingface’s `transformers`).
     - Truncate prompts and stories to a maximum length (e.g., 512 tokens for prompts, 1024 for stories) to manage computational resources.
     - Split the dataset into training, validation, and test sets (e.g., 80-10-10 split).
   - **Reasoning**: Preprocessing ensures the data is clean, appropriately sized for model input, and representative of creative writing tasks.

2. **Dataset Statistics**:
   - **Document Count**: Count the number of prompt-story pairs in the dataset (e.g., `len(dataset["train"])`).
   - **Document Length**: Calculate the average, minimum, and maximum length (in tokens or words) of prompts and stories using the tokenizer.
   - **Token Count**: Compute the total number of tokens across the dataset and the distribution of tokens per prompt and story.
   - **Target Length**: Analyze the length of story continuations (e.g., mean, median, standard deviation).
   - **Vocabulary Size**: Estimate the unique token count after tokenization.
   - **Reasoning**: These statistics provide insights into the dataset’s scale, complexity, and diversity, which inform model selection and preprocessing decisions.

3. **LDA Topic Modeling**:
   - **Implementation**: Use `gensim` to implement Latent Dirichlet Allocation (LDA).
     - Preprocess the stories by removing stop words, lemmatizing, and creating a bag-of-words representation.
     - Train an LDA model (e.g., 10 topics) on the stories to identify common themes (e.g., fantasy, dystopian, romance).
   - **Analysis**: Visualize topics using `pyLDAvis` and interpret the dominant words per topic. For example, a topic with words like “dragon,” “magic,” and “quest” might indicate fantasy stories.
   - **Reasoning**: Topic modeling reveals the thematic diversity of the dataset, which helps understand the creative scope and guides model evaluation (e.g., checking if generated stories align with these themes).

4. **Text Generation Models**:
   - **Model 1: BART (facebook/bart-base)**:
     - BART is a transformer-based model combining an encoder-decoder architecture, pre-trained for sequence-to-sequence tasks like summarization and translation, but adaptable for creative generation.
     - Fine-tune BART on the `writingprompts` dataset by feeding prompts as input and stories as output. Alternatively, use zero-shot decoding with a pre-trained BART model by providing prompts and generating continuations.
   - **Model 2: T5 (t5-small or t5-base)**:
     - T5 is another encoder-decoder model that treats all tasks as text-to-text transformations, making it versatile for story generation.
     - Fine-tune or decode similarly to BART, using prompts as input and stories as target outputs.
   - **Implementation**: Use Huggingface’s `transformers` library to load pre-trained models, tokenize data, and set up training or decoding pipelines.
   - **Reasoning**: BART and T5 are suitable for creative text generation due to their strong language modeling capabilities and flexibility in handling open-ended tasks.

5. **Evaluation and Comparison**:
   - **Metrics**:
     - **BLEU**: Measures n-gram overlap between generated and reference stories. Useful for assessing lexical similarity but may not capture creativity.
     - **ROUGE**: Evaluates overlap in unigrams (ROUGE-1), bigrams (ROUGE-2), and longest common subsequences (ROUGE-L) to assess content similarity.
     - **Perplexity**: Measures how well the model predicts the next token in the reference story (lower is better).
     - **Human-like Metrics**: Use BERTScore to assess semantic similarity between generated and reference stories, capturing meaning beyond exact matches.
     - **Diversity Metrics**: Compute Self-BLEU or unique n-grams to evaluate the diversity of generated stories.
   - **Comparison**: Compare BART and T5 on these metrics. For example, BART may produce more coherent stories due to its denoising pre-training, while T5 might generate more diverse outputs due to its text-to-text framework.
   - **Reasoning**: Multiple metrics are used to balance fluency, coherence, and creativity. Observations (e.g., BART’s higher BLEU but lower diversity) can be justified by model architectures or pre-training objectives.

6. **Report Writing**:
   - **Content**:
     - Describe the task, dataset, and preprocessing steps with justifications.
     - Present dataset statistics with visualizations (e.g., histograms of document lengths).
     - Discuss LDA topics with examples and their relevance to the dataset.
     - Explain the architecture of BART and T5 (e.g., encoder-decoder layers, attention mechanisms).
     - Summarize evaluation results in tables/graphs and discuss observations (e.g., why one model outperforms another).
   - **Format**: Write a concise PDF report (5-8 pages), including sections for introduction, methodology, results, discussion, and conclusion.
   - **Submission**: Submit the report and the Colab notebook with code for preprocessing, LDA, model training/decoding, and evaluation.
