# story-generation

Story continuation: given a short writing prompt, generate a story. Fine-tunes and compares two encoder-decoder models, BART and T5, on the WritingPrompts dataset.

## What's in the notebook

[main.ipynb](main.ipynb) (written for Google Colab with a GPU) covers:

1. **Data:** loads `euclaise/writingprompts` (prompts and stories from Reddit r/WritingPrompts) and merges its splits. Keeps prompts with ≥10 words and stories with ≥150 words, then re-splits 80/10/10 into train, validation and test.
2. **Dataset statistics:** prompt and story word counts, and their length distributions.
3. **LDA topic modelling:** a 5-topic `gensim` LDA model on the stories. It finds topics for dialogue/physical interaction, existential themes, sci-fi, casual reflection and sensory setting.
4. **Models:** fine-tunes `facebook/bart-base` and `t5-base` (prompt → story) on a 10k-pair subset with `Seq2SeqTrainer`. It also runs zero-shot baselines with `bart-large-cnn` and `t5-base`.
5. **Evaluation:** ROUGE-1/2/L/Lsum, BERTScore and length ratio, plus sample generations.

## Results

| Model (fine-tuned, 6000 steps) | Val loss | Best ROUGE-L | BERTScore F1 |
|---|---|---|---|
| BART-base | 2.80 | 0.041 | ~0.37 |
| T5-base | 4.88 | lower than BART | ~0.33 |

BART did better than T5. Both models produced generic, repetitive stories, and ROUGE scores were low because a story can be good without matching the reference wording. See the report for full tables and discussion.

## Running

Open `main.ipynb` in Colab and add an `HF_TOKEN` secret (a Hugging Face access token). The first cells install the dependencies (`datasets`, `transformers`, `evaluate`, `gensim`, `nltk`, `rouge_score`, `bert_score`, `sentencepiece`).

## Repo layout

```
main.ipynb       notebook: data, statistics, LDA, BART/T5 training and evaluation
latex/           report source (nlpaics2024.tex), LDA figures, and compiled report.pdf
```
