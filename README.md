Pipeline at a glance

raw Kaggle CSVs

   │
   
   ▼
   
[1] corpus acquisition + cleaning  ──►  cleaned CSVs (per corpus)

   │
   
   ▼
[2] target-word filtering (BERT only)  ──►  ./filtered/*.csv

   │
   
   ├──► [3] Word2Vec training + Procrustes alignment  ──►  .bin models + SNR tables
   
   │
   
   └──► [4] BERT centroid extraction  ──►  cosine-distance tables


Corpus&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kaggle dataset (Link)

arXiv CS abstracts&nbsp;&nbsp;&nbsp;&nbsp;          https://www.kaggle.com/datasets/devintheai/arxiv-cs-papers-multi-label-classification-200k-v1

PubMed clinical trials&nbsp;&nbsp;&nbsp;&nbsp;      https://www.kaggle.com/datasets/matthewjansen/pubmed-200k-rtc/data

Guardian news article&nbsp;&nbsp;&nbsp;&nbsp;       https://www.kaggle.com/datasets/adityakharosekar2/guardian-news-articles


2. Notebooks, in run order

Stage 1 — Corpus acquisition & cleaning

Notebook&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Corpus&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;What it does	

cs_arXiv_papers_api.ipynb&nbsp;&nbsp;&nbsp;&nbsp;	  arXiv  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Loads cs_papers_api.csv, splits by year into the three periods, merges title + abstract into a single text field

pubmed_abstract.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	      PubMed&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  	Loads the raw RCT CSV, drops metadata columns (abstract_id, line_id, line_number, abstract_text), runs NLTK cleaning (lowercase, stopword removal, tokenisation)

guardian_news.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	        Guardian &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Loads guardian_articles.csv (149,839 × 7), drops non-text columns, runs the same NLTK cleaning pass


Cleaning is identical across all three: lowercase, NLTK stopword removal, whitespace tokenisation, keep hyphenated tokens whole (so feed-forward and physics-informed survive as single tokens). 
Cleaning code is commented out after the first run, deliberately. Everything downstream reads from the saved CSV snapshot, not a live re-clean, so numbers don't quietly drift between reruns.



Stage 2 — BERT filtering

Notebook/script&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	                        What it does

bert_filter_csv.ipynb </br>
(implements filter_target_words.py)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Keeps only rows containing at least one target word, per corpus and, for arXiv, per period. Run once — every downstream BERT step reads its output.



Stage 3 — Word2Vec: training, alignment, drift

Notebook &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                  Scope	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          Target words

word2vec.ipynb	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        Cross-domain (arXiv → PubMed, arXiv → Guardian) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; neural, memory, cell, agent, network, node vs. PubMed; cloud, stream, port vs. Guardian

alignmentwithtemporal.ipynb	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Temporal (within arXiv, 1990→2012→2020)	   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    reward, attention, agent, cloud, container, pipeline, hallucination, transformer


Trained models are .bin files, named by corpus, period, and seed:

CS_arXiv/u_model_1990_{seed}.bin  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   CS_arXiv/u_model_2012_{seed}.bin  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   CS_arXiv/u_model_2020_{seed}.bin
pubmed/model_pubmed_200_{seed}.bin &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; guardian/guardian_{seed}.bin


Stage 4 — BERT: contextual centroids

Notebook     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     What it does

bert_demo.ipynb	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    Loads bert-base-uncased, samples up to 500 sentences per word per corpus slice from the ./filtered/ CSVs, 
                    averages WordPiece sub-token embeddings, L2-normalises, and computes a centroid cosine distance to the PubMed/Guardian reference centroid
