# Awesome Rerankers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of reranking models, libraries, and resources for building high-quality Retrieval-Augmented Generation (RAG) applications.

Rerankers are specialized models that refine initial search results by re-scoring and reordering documents based on their semantic relevance to a query. They serve as a crucial second-stage filter in RAG pipelines, significantly improving retrieval quality and reducing LLM hallucinations.

## Contents

- [What are Rerankers?](#what-are-rerankers)
- [Open Source Models](#open-source-models)
  - [Cross-Encoder Models](#cross-encoder-models)
  - [T5-Based Models](#t5-based-models)
  - [LLM-Based Models](#llm-based-models)
- [Commercial APIs](#commercial-apis)
- [Libraries & Frameworks](#libraries--frameworks)
- [RAG Framework Integrations](#rag-framework-integrations)
  - [LangChain](#langchain)
  - [LlamaIndex](#llamaindex)
  - [Haystack](#haystack)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Evaluation Metrics](#evaluation-metrics)
- [Research Papers](#research-papers)
- [Tutorials & Resources](#tutorials--resources)
- [Tools & Utilities](#tools--utilities)
- [Related Awesome Lists](#related-awesome-lists)

## What are Rerankers?

Rerankers are models that take a query and a set of retrieved documents as input and output relevance scores for reordering. They differ from traditional retrieval in several key ways:

- **Two-Stage Architecture** - Initial retrieval casts a wide net (e.g., 100-1000 candidates), then reranking refines to top-k most relevant
- **Cross-Attention** - Unlike bi-encoders that encode query and document separately, rerankers jointly encode both for better semantic understanding
- **Quality vs Speed Tradeoff** - More computationally expensive than vector search but significantly more accurate
- **Types** - Pointwise (score each doc independently), pairwise (compare doc pairs), and listwise (score entire list)

## Open Source Models

### Cross-Encoder Models

Cross-encoders jointly encode query and document pairs, providing highly accurate relevance scores.

- **[BGE-Reranker](https://github.com/FlagOpen/FlagEmbedding)** - BAAI's state-of-the-art reranking models trained on massive datasets. Available in base, large, and v2-m3 variants with multilingual support.
- **[bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)** - Lightweight model (278M parameters) optimized for speed and efficiency.
- **[bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)** - High-performance model (560M parameters) for maximum accuracy.
- **[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)** - Multilingual model supporting 100+ languages with 568M parameters.
- **[bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)** - Based on Google's Gemma architecture for improved performance.
- **[Jina Reranker v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)** - Multilingual reranker with 1024 token context length and sliding window support.
- **[jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)** - Supports 100+ languages with function-calling and code search capabilities.
- **[mixedbread-ai/mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2)** - Qwen-2.5-based model with 0.5B parameters, outperforming competitors on BEIR.
- **[mixedbread-ai/mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)** - 1.5B parameter model achieving top BEIR benchmark scores.
- **[ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)** - Microsoft's efficient cross-encoder trained on MS MARCO dataset.
- **[ms-marco-TinyBERT-L-6](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-6)** - Ultra-lightweight variant for resource-constrained environments.

### T5-Based Models

Sequence-to-sequence models leveraging T5 architecture for text ranking.

- **[MonoT5](https://huggingface.co/castorini/monot5-base-msmarco)** - Pointwise T5-base reranker fine-tuned on MS MARCO, scores documents independently.
- **[DuoT5](https://huggingface.co/castorini/duot5-3b-msmarco)** - Pairwise T5-3B reranker for comparing document pairs with O(n²) complexity.
- **[RankT5](https://github.com/castorini/rank_llm)** - T5 variant fine-tuned with specialized ranking losses for improved performance.
- **[PyTerrier T5](https://github.com/terrierteam/pyterrier_t5)** - T5-based reranking models integrated with PyTerrier IR platform.

### LLM-Based Models

Large language models adapted for reranking tasks with zero-shot or few-shot capabilities.

- **[RankLLM](https://github.com/castorini/rank_llm)** - Unified framework supporting RankVicuna, RankZephyr, and RankGPT with vLLM/SGLang/TensorRT-LLM integration.
- **[RankGPT](https://github.com/sunnweiwei/RankGPT)** - Zero-shot listwise reranking using GPT-3.5/GPT-4 with permutation generation.
- **[LiT5](https://github.com/castorini/rank_llm)** - Listwise reranking model based on T5 architecture.
- **[RankVicuna](https://github.com/castorini/rank_llm)** - Vicuna LLM fine-tuned for ranking tasks.
- **[RankZephyr](https://github.com/castorini/rank_llm)** - Zephyr-based model optimized for reranking.

## Commercial APIs

Production-ready reranking services with enterprise support and scalability.

- **[Cohere Rerank](https://docs.cohere.com/docs/reranking)** - Leading reranking API with multilingual support (100+ languages) and "Nimble" variant for low latency.
- **[Voyage AI Rerank](https://docs.voyageai.com/docs/reranker)** - Instruction-following rerankers (rerank-2.5/rerank-2.5-lite) with 200M free tokens.
- **[Jina AI Reranker API](https://jina.ai/reranker/)** - Cloud-hosted Jina reranker models with pay-as-you-go pricing.
- **[Pinecone Rerank](https://docs.pinecone.io/guides/rerank)** - Integrated reranking service within Pinecone's vector database platform.
- **[Mixedbread AI Reranker API](https://www.mixedbread.ai/api-reference/endpoints/reranking)** - API access to mxbai-rerank models with competitive pricing.
- **[NVIDIA NeMo Retriever](https://www.nvidia.com/en-us/ai/ai-enterprise-suite/nemo-retriever/)** - Enterprise-grade reranking optimized for NVIDIA hardware.

## Libraries & Frameworks

### Unified Reranking Libraries

- **[rerankers](https://github.com/AnswerDotAI/rerankers)** - Lightweight Python library providing unified API for all major reranking models (FlashRank, Cohere, RankGPT, cross-encoders).
- **[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)** - Ultra-lite (~4MB) reranking library with zero torch/transformers dependencies, supports CPU inference.
- **[Sentence-Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)** - Popular library for training and using cross-encoder reranking models.
- **[rank-llm](https://pypi.org/project/rank-llm/)** - Python package for listwise and pairwise reranking with LLMs.

### Specialized Tools

- **[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)** - BAAI's comprehensive toolkit for embeddings and reranking, includes BGE models and training code.
- **[PyTerrier](https://github.com/terrier-org/pyterrier)** - Information retrieval platform with extensive reranking support and experimentation tools.

## RAG Framework Integrations

### LangChain

Node postprocessors and document transformers for reranking in LangChain pipelines.

- **[Cohere Reranker](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/)** - Official Cohere integration using ContextualCompressionRetriever.
- **[FlashRank Reranker](https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/)** - Lightweight reranking without heavy dependencies.
- **[RankLLM Reranker](https://python.langchain.com/docs/integrations/document_transformers/rankllm-reranker/)** - LLM-based listwise reranking for LangChain.
- **[Cross Encoder Reranker](https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/)** - Hugging Face cross-encoder models integration.
- **[Pinecone Rerank](https://python.langchain.com/docs/integrations/retrievers/pinecone_rerank/)** - Native Pinecone reranking support.
- **[VoyageAI Reranker](https://python.langchain.com/docs/integrations/document_transformers/voyageai-reranker/)** - Voyage AI models for document reranking.

### LlamaIndex

Postprocessor modules for enhancing retrieval in LlamaIndex query engines.

- **[CohereRerank](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank/)** - Top-N reranking using Cohere's API.
- **[SentenceTransformerRerank](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)** - Cross-encoder reranking from sentence-transformers.
- **[LLMRerank](https://docs.llamaindex.ai/en/latest/api_reference/postprocessor/llm_rerank/)** - Uses LLMs to score and reorder retrieved nodes.
- **[JinaRerank](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)** - Jina AI reranker integration.
- **[RankLLM Rerank](https://pypi.org/project/llama-index-postprocessor-rankllm-rerank/)** - RankLLM models as postprocessors.
- **[NVIDIA Rerank](https://pypi.org/project/llama-index-postprocessor-nvidia-rerank/)** - NVIDIA NeMo Retriever integration.

### Haystack

Ranker components for deepset's Haystack framework.

- **[CohereRanker](https://docs.haystack.deepset.ai/docs/cohereranker)** - Semantic reranking with Cohere models.
- **[SentenceTransformersRanker](https://docs.haystack.deepset.ai/docs/rankers)** - Cross-encoder based reranking.
- **[JinaRanker](https://haystack.deepset.ai/integrations/jina)** - Jina reranker models for Haystack pipelines.
- **[MixedbreadAIRanker](https://haystack.deepset.ai/integrations/mixedbread-ai)** - Mixedbread AI reranker integration.
- **[LostInTheMiddleRanker](https://docs.haystack.deepset.ai/docs/rankers)** - Optimizes document ordering to combat the "lost in the middle" phenomenon.

## Datasets & Benchmarks

### Training & Evaluation Datasets

- **[MS MARCO](https://microsoft.github.io/msmarco/)** - Large-scale passage and document ranking datasets with real Bing queries.
- **[MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/Datasets)** - 8.8M passages with 500k+ training queries for passage retrieval.
- **[MS MARCO Document Ranking](https://microsoft.github.io/msmarco/Datasets)** - 3.2M documents for full document ranking tasks.
- **[BEIR](https://github.com/beir-cellar/beir)** - Heterogeneous benchmark with 18 diverse datasets for zero-shot evaluation.
- **[TREC Deep Learning Track](https://trec.nist.gov/data/deep.html)** - High-quality test collections (TREC-DL-2019, TREC-DL-2020) for passage/document ranking.
- **[TREC-DL-2019](https://trec.nist.gov/data/deep/2019.html)** - 200 queries with dense relevance judgments.
- **[TREC-DL-2020](https://trec.nist.gov/data/deep/2020.html)** - 200 queries with expanded corpus coverage.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)** - Google's dataset of real user questions for QA and retrieval.
- **[SciRerankBench](https://arxiv.org/abs/2508.08742)** - Specialized benchmark for scientific document reranking.

### Benchmark Suites

- **[BEIR Benchmark](https://github.com/beir-cellar/beir)** - Zero-shot evaluation across 18 retrieval tasks (NQ, HotpotQA, FiQA, ArguAna, etc.).
- **[MTEB Reranking](https://github.com/embeddings-benchmark/mteb)** - Massive Text Embedding Benchmark including reranking tasks.

## Evaluation Metrics

Key metrics for assessing reranker performance:

- **[NDCG (Normalized Discounted Cumulative Gain)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)** - Standard metric emphasizing top results, commonly reported as NDCG@10.
- **[MRR (Mean Reciprocal Rank)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)** - Measures average inverse rank of first relevant result, used by MS MARCO (MRR@10).
- **[MAP (Mean Average Precision)](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Mean_average_precision)** - Average precision across all relevant documents.
- **[Recall@K](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Recall)** - Percentage of relevant documents in top-K results.
- **[Precision@K](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Precision)** - Proportion of relevant documents in top-K results.

## Research Papers

### Foundational Papers

- **[Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)** (2020) - Introduces MonoT5 and DuoT5 for text ranking.
- **[RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](https://arxiv.org/abs/2210.10634)** (2022) - Specialized ranking losses for T5 models.
- **[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542)** (2023) - Introduces RankGPT and zero-shot LLM reranking.
- **[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)** (2021) - Establishes BEIR benchmark suite.

### Recent Advances

- **[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)** (2023) - Introduces BGE reranking models.
- **[JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking](https://arxiv.org/abs/2411.00142)** (2024) - LLM-based reranking with reasoning capabilities.
- **[Rank1: Test-Time Compute for Reranking in Information Retrieval](https://arxiv.org/abs/2502.18418)** (2025) - Novel test-time optimization for reranking.
- **[How Good are LLM-based Rerankers? An Empirical Analysis](https://arxiv.org/abs/2508.16757)** (2025) - Comprehensive comparison of LLM reranking approaches.
- **[SciRerankBench: Benchmarking Rerankers Towards Scientific RAG-LLMs](https://arxiv.org/abs/2508.08742)** (2025) - Specialized evaluation for scientific literature.
- **[A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems](https://arxiv.org/abs/2507.18910)** (2025) - Comprehensive RAG survey including reranking techniques.

### Survey Papers

- **[Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467)** (2020) - Survey of neural ranking models.
- **[Neural Models for Information Retrieval](https://arxiv.org/abs/1705.01509)** (2017) - Foundational survey of neural IR approaches.

## Tutorials & Resources

### Comprehensive Guides

- **[Top 7 Rerankers for RAG](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)** - Analytics Vidhya's comparison of leading reranking models.
- **[Comprehensive Guide on Reranker for RAG](https://www.analyticsvidhya.com/blog/2025/03/reranker-for-rag/)** - In-depth tutorial on implementing rerankers in RAG systems.
- **[Improving RAG Accuracy with Rerankers](https://www.infracloud.io/blogs/improving-rag-accuracy-with-rerankers/)** - Practical guide with implementation examples.
- **[Mastering RAG: How to Select A Reranking Model](https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model)** - Selection criteria and comparison framework.

### Implementation Tutorials

- **[Boosting RAG: Picking the Best Embedding & Reranker Models](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)** - LlamaIndex guide with benchmarks.
- **[Advanced RAG: Evaluating Reranker Models using LlamaIndex](https://akash-mathur.medium.com/advanced-rag-enhancing-retrieval-efficiency-through-evaluating-reranker-models-using-llamaindex-3f104f24607e)** - Step-by-step evaluation tutorial.
- **[Enhancing Advanced RAG Systems Using Reranking with LangChain](https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311)** - LangChain implementation patterns.
- **[Training and Finetuning Reranker Models with Sentence Transformers v4](https://huggingface.co/blog/train-reranker)** - Official Hugging Face training guide.
- **[Fine-Tuning Re-Ranking Models for LLM-Based Search](https://www.rohan-paul.com/p/fine-tuning-re-ranking-models-for)** - Domain-specific fine-tuning techniques.

### Video Tutorials

- **[Implementing Rerankers in Your AI Workflows](https://blog.n8n.io/implementing-rerankers-in-your-ai-workflows/)** - n8n's practical workflow tutorial.
- **[Cohere Rerank on LangChain Integration Guide](https://docs.cohere.com/docs/rerank-on-langchain)** - Official Cohere tutorial.

### Blog Posts & Articles

- **[Rerankers in RAG](https://medium.com/@avd.sjsu/rerankers-in-rag-2f784fc977f3)** - Conceptual overview of reranking in RAG pipelines.
- **[Sentence Embeddings: Cross-encoders and Re-ranking](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)** - Technical deep-dive into cross-encoder architectures.
- **[The Four Types of Passage Reranker in RAG](https://medium.com/@autorag/the-four-types-of-passage-reranker-in-rag-02c907b4f747)** - Classification and comparison of reranker types.
- **[RAG in 2025: From Quick Fix to Core Architecture](https://medium.com/@hrk84ya/rag-in-2025-from-quick-fix-to-core-architecture-9a9eb0a42493)** - Industry trends and best practices.
- **[Boosting Your Search and RAG with Voyage's Rerankers](https://blog.voyageai.com/2024/03/15/boosting-your-search-and-rag-with-voyages-rerankers/)** - Voyage AI's technical blog.

## Tools & Utilities

### Evaluation Tools

- **[ranx](https://github.com/AmenRa/ranx)** - Fast IR evaluation library supporting NDCG, MAP, MRR, and more.
- **[ir-measures](https://github.com/terrierteam/ir_measures)** - Comprehensive IR metrics library with TREC integration.
- **[MTEB](https://github.com/embeddings-benchmark/mteb)** - Massive Text Embedding Benchmark for systematic evaluation.

### Development Tools

- **[Haystack Studio](https://haystack.deepset.ai/overview/haystack-studio)** - Visual pipeline builder with reranking components.
- **[LangSmith](https://www.langchain.com/langsmith)** - Debugging and monitoring for LangChain pipelines including rerankers.
- **[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)** - Automated RAG optimization including reranker selection.

### Visualization Tools

- **[Text Embeddings Visualization](https://projector.tensorflow.org/)** - TensorFlow's embedding projector for understanding model behavior.
- **[Phoenix](https://github.com/Arize-ai/phoenix)** - LLM observability platform with retrieval tracing.

## Related Awesome Lists

- **[Awesome RAG](https://github.com/tholman/awesome-rag)** - Comprehensive RAG resources and frameworks.
- **[Awesome LLM](https://github.com/Hannibal046/Awesome-LLM)** - Large Language Models resources and tools.
- **[Awesome Information Retrieval](https://github.com/harpribot/awesome-information-retrieval)** - IR papers, datasets, and tools.
- **[Awesome Embedding Models](https://github.com/Hannibal046/Awesome-Embedding-Models)** - Vector embeddings and similarity search.
- **[Awesome Neural Search](https://github.com/currentslab/awesome-neural-search)** - Neural search and dense retrieval resources.
- **[Awesome Vector Search](https://github.com/currentslab/awesome-vector-search)** - Vector databases and search engines.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

To add a new item:
1. Search previous suggestions before making a new one
2. Make an individual pull request for each suggestion
3. Use the following format: `**[Name](link)** - Description.`
4. New categories or improvements to the existing categorization are welcome
5. Keep descriptions concise and informative
6. Check your spelling and grammar
7. Make sure your text editor is set to remove trailing whitespace

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this work.
