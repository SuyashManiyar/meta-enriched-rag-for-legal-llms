# Meta-Enriched RAG for Legal LLMs

A comprehensive research project implementing and evaluating metadata-enhanced Retrieval-Augmented Generation (RAG) systems for legal document analysis, with a focus on Australian legal documents.

## 🎯 Project Overview

This project investigates the effectiveness of metadata-enriched chunking strategies in RAG systems for legal document retrieval and question answering. The research compares baseline recursive chunking with an enhanced meta-recursive approach that incorporates document metadata to improve retrieval performance.

### Key Research Questions
- How does metadata enrichment affect retrieval accuracy in legal RAG systems?
- What is the optimal chunking strategy for legal documents?
- How do different retrieval methods (dense vs. sparse vs. hybrid) perform on legal queries?

## 📊 Datasets

### Australian Legal Dataset
- **Size**: 146 queries across 200 legal documents
- **Document Types**: Federal Court judgments, High Court decisions, legislation
- **Ground Truth**: Manual annotations with span-level relevance judgments
- **Files**: 
  - `australian_legal_data/aus_test_qa_corrected.json` - Corrected Q&A pairs
  - `australian_legal_data/australian_legal_w_query_ids.json` - Extended dataset with query IDs

### MAUD Dataset (Merger Agreement Understanding Dataset)
- **Size**: Contract analysis questions with span-level annotations
- **Document Types**: M&A agreements, merger contracts, acquisition documents
- **Ground Truth**: Expert annotations for contract clause identification
- **Files**:
  - `MUAD_test_cs685.csv` - Test set with contract excerpts and labels
  - `test_data/maud.json` - Structured Q&A pairs with document spans
  - `RAG_pipeline/window_summary_chunking/generated_chunks/all_maud_chunks_*.json` - Processed chunks
- **Categories**: General Information, Deal Protection, Conditions, etc.

### Privacy QA Dataset
- **Purpose**: Comparative evaluation on privacy-related legal questions
- **Location**: `privacyQA/` directory

## 🏗️ Architecture

### RAG Pipeline Components

#### 1. Document Processing (`RAG_pipeline/other_experiments/`)
- **Chunking**: `chunking.py` - Implements multiple chunking strategies
  - Fixed-size chunking with sliding windows
  - Recursive chunking with metadata preservation
  - Meta-recursive chunking with enhanced metadata
- **Embedding**: `create_embedding.py` - Document vectorization using transformer models

#### 2. Retrieval Systems
- **Dense Retrieval**: Semantic similarity using embeddings
- **Sparse Retrieval**: BM25-based keyword matching
- **Hybrid Retrieval**: Combined dense + sparse approaches

#### 3. Evaluation Framework (`RAG_pipeline/other_experiments/evaluate.py`)
- **Metrics**:
  - Document Retrieval Mismatch Rate (DRM)
  - Span Precision and Recall
  - Alternative precision/recall metrics
- **Multi-k Evaluation**: Performance across k ∈ {1, 2, 4, 8, 16, 32, 64}

## 🧪 Experiments

### Baseline Experiments
**Location**: `australian_legal_data/results_recursive/`

- **Approach**: Standard recursive chunking
- **Retrieval**: Dense-only, Sparse-only, Dense+Sparse hybrid
- **Results**: Comprehensive evaluation across all k values

### Enhanced Experiments  
**Location**: `australian_legal_data/results_meta_recursive/`

- **Approach**: Meta-recursive chunking with metadata enrichment
- **Innovation**: Incorporates document structure, citation information, and legal metadata
- **Comparison**: Direct comparison with baseline approaches

### Stochastic Analysis
**Script**: `extract_stochastic_ranges.py`

- **Method**: Bootstrap sampling for statistical significance
- **Confidence Intervals**: 95% CI for all performance metrics
- **Output**: `stochastic_ranges_table.tex` - LaTeX table with statistical results

## 📈 Key Results

### Performance Improvements (Meta-Recursive vs. Baseline)

| Metric | k=1 | k=4 | k=16 | Statistical Significance |
|--------|-----|-----|------|-------------------------|
| **DRM** | 0.0pp | -13.5pp | -17.0pp | Significant improvement |
| **Span Recall** | +0.153 | +0.236 | +0.306 | Significant improvement |

### Key Findings
1. **Metadata enrichment significantly improves retrieval accuracy** at higher k values
2. **Span-level performance shows consistent improvements** across all k values
3. **Hybrid dense+sparse retrieval outperforms** individual approaches
4. **Statistical significance confirmed** through bootstrap analysis

## �  Quick Start

### Prerequisites
- Python 3.8 or higher
- 16GB+ RAM recommended
- GPU optional but recommended for faster processing

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd meta-enriched-rag-for-legal-llms
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model** (optional):
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

## 🛠️ Usage Guide

### Complete Workflow

#### Option A: Use Pre-computed Data (Recommended)
```bash
# Download from Google Drive link above
# Extract to project root directory
# Skip to step 3 (Run Experiments)
```

#### Option B: Generate Data from Scratch

#### 1. Data Collection (Optional)
```bash
# Scrape Australian legal documents
cd utils/
pip install -r scraper_requirements.txt
python scrape_australian_legal_docs.py
```

#### 2. Document Processing
```bash
# Process and chunk documents
cd RAG_pipeline/other_experiments/
python chunking.py

# Create embeddings
python create_embedding.py
```

#### 3. Run Experiments

**Baseline Experiments (Recursive Chunking)**:
```bash
# Dense-only retrieval
python evaluate.py --config recursive_dense_only

# Dense + Sparse hybrid
python evaluate.py --config recursive_dense_sparse
```

**Enhanced Experiments (Meta-Recursive Chunking)**:
```bash
# Meta-recursive with metadata enhancement
python evaluate.py --config meta_recursive_dense_sparse
```

#### 4. Statistical Analysis
```bash
# Generate stochastic evaluation with confidence intervals
python extract_stochastic_ranges.py

# Create performance plots
cd ../../utils/
python create_line_plots.py
```

### MAUD Dataset Experiments

1. **Document Processing**:
   ```bash
   cd RAG_pipeline/window_summary_chunking/
   python chunk_maud_docs.py
   ```

2. **Add Metadata**:
   ```bash
   cd utils/
   python add_file_metadata.py
   python add_doc_summary.py
   ```

3. **Run Evaluation**:
   ```bash
   cd ../other_experiments/
   python for_maud_run_before_eval.py
   python evaluate.py --dataset maud
   ```

### Privacy QA Experiments

```bash
cd privacyQA/
# Follow dataset-specific instructions
python run_privacy_qa_experiments.py
```

### DPO (Direct Preference Optimization) Experiments

The project includes DPO fine-tuning for legal LLMs to improve response quality:

1. **Data Preparation**:
   ```bash
   cd dpo/
   jupyter notebook data_preperation.ipynb
   ```

2. **LLaMA Fine-tuning with DPO**:
   ```bash
   jupyter notebook llama_finetuning.ipynb
   ```

3. **Model Inference**:
   ```bash
   jupyter notebook inference_on_finetuned_model.ipynb
   ```

4. **Evaluation**:
   ```bash
   jupyter notebook eval_bert_score.ipynb
   ```

#### DPO Model Weights
Pre-trained DPO weights are available on Google Drive:
- **Download Link**: [DPO Weights](https://drive.google.com/drive/folders/1x9Nzx59bugAB7anFNO_EnL3U44vZwm_n?usp=sharing)
- **Usage**: Download and place in `dpo/models/` directory
- **Models**: Fine-tuned LLaMA variants optimized for legal question answering

## 📁 Data and Model Downloads

### Complete RAG Pipeline Files
All RAG-related files, datasets, and pre-computed results are available on Google Drive:
- **📂 RAG Files**: [Complete RAG Pipeline Data](https://drive.google.com/drive/folders/1pyszc_Uz21Rwu5_R-2E7a_1B3ZfYP9Kq?usp=drive_link)

**Contents include**:
- **📄 Legal Documents**: 200 Australian legal documents (PDFs)
- **🔍 Generated Chunks**: Pre-processed document chunks (recursive & meta-recursive)
- **🎯 Embeddings**: Pre-computed dense embeddings for all chunks
- **📊 Evaluation Results**: Complete experimental results (baseline & enhanced)
- **📈 Statistical Analysis**: Bootstrap results and confidence intervals
- **🗂️ Datasets**: Australian Legal, MAUD, and Privacy QA datasets
- **⚙️ Model Checkpoints**: Trained retrieval models and configurations

### Download Instructions
1. **Download the complete folder** from the Google Drive link
2. **Extract to project root**: Place contents in appropriate directories
3. **Verify structure**: Ensure `australian_legal_data/`, `RAG_pipeline/`, etc. are populated
4. **Run experiments**: Skip data preparation and go directly to evaluation

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# API Keys (if using external services)
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Model Configurations
EMBEDDING_MODEL=thenlper/gte-large
CHUNK_SIZE=380
WINDOW_SIZE=50

# Paths
DATA_DIR=./australian_legal_data
OUTPUT_DIR=./results
```

### Experiment Configurations
- **Chunking strategies**: Fixed, recursive, meta-recursive
- **Retrieval methods**: Dense-only, sparse-only, hybrid
- **Evaluation metrics**: DRM, span precision/recall, document retrieval
- **Statistical analysis**: Bootstrap sampling, confidence intervals

### Key Configuration Files
- `.env` - Environment variables and API keys
- `RAG_pipeline/australian_legal_experiments/evaluate_stochastic.py` - Stochastic evaluation setup

## 🛠️ Utility Scripts

### Data Processing Utilities
- **`RAG_pipeline/utils/add_spans_to_ground_truth.py`** - Enhances ground truth with proper span information for accurate evaluation
- **`RAG_pipeline/window_summary_chunking/utils/add_file_metadata.py`** - Adds document-level metadata to chunks (categories, user rights, etc.)
- **`RAG_pipeline/window_summary_chunking/utils/add_doc_summary.py`** - Incorporates document summaries into chunk metadata
- **`RAG_pipeline/window_summary_chunking/utils/merge_summarized_chunks.py`** - Merges chunks with their generated summaries

### Analysis & Visualization
- **`utils/create_line_plots.py`** - Generates performance line plots across different k values
- **`utils/extract_stochastic_ranges.py`** - Extracts statistical ranges from stochastic evaluation results
- **`extract_stochastic_ranges.py`** - Main statistical analysis script for confidence intervals

### Document Collection & Web Scraping
- **`utils/scrape_australian_legal_docs.py`** - Comprehensive web scraper for Australian legal documents
  - **Sources**: Federal Court, High Court, NSW Caselaw, legislation.gov.au, state legislation
  - **Features**: Rate limiting, error handling, metadata extraction, multiple formats (PDF/HTML/DOCX)
  - **Output**: 200+ documents with structured metadata
- **`utils/scraper_requirements.txt`** - Dependencies for the web scraper

#### Running the Scraper:
```bash
cd utils/
pip install -r scraper_requirements.txt
python scrape_australian_legal_docs.py
```

#### Scraper Features:
- **Multi-source collection**: Federal Court, High Court, NSW Caselaw, legislation databases
- **Respectful scraping**: Random delays, proper headers, error handling
- **Metadata preservation**: URLs, document types, file sizes, source domains
- **Format support**: PDF, HTML, DOCX documents
- **Logging**: Comprehensive logging for monitoring and debugging

## 📁 Project Structure

```
├── australian_legal_data/           # Main dataset and results
│   ├── results_recursive/           # Baseline experiment results
│   ├── results_meta_recursive/      # Enhanced experiment results
│   ├── generated_chunks/            # Processed document chunks
│   └── aus_test_qa_corrected.json   # Ground truth Q&A pairs
├── RAG_pipeline/                    # Core RAG implementation
│   ├── other_experiments/           # Main experimental scripts
│   ├── australian_legal_experiments/ # Specialized legal experiments
│   ├── window_summary_chunking/     # MAUD dataset processing
│   │   └── utils/                   # Metadata and summarization utilities
│   └── utils/                       # Core utility functions
├── utils/                           # Analysis and visualization utilities
├── test_data/                       # Test datasets
│   └── maud.json                    # MAUD Q&A pairs
├── privacyQA/                       # Privacy QA dataset
├── rag/                            # Additional RAG components
├── dpo/                            # Direct Preference Optimization
│   ├── data_preperation.ipynb      # DPO data preparation
│   ├── llama_finetuning.ipynb      # LLaMA fine-tuning with DPO
│   ├── inference_on_finetuned_model.ipynb # Model inference
│   ├── eval_bert_score.ipynb       # BERT score evaluation
│   └── README.md                   # DPO-specific documentation
├── MUAD_test_cs685.csv             # MAUD test set
└── extract_stochastic_ranges.py    # Statistical analysis script
```

## 📊 Evaluation Metrics

### Document Retrieval Mismatch Rate (DRM)
- **Definition**: Fraction of ground truth documents not retrieved in top-k
- **Range**: [0, 1] (lower is better)
- **Significance**: Measures document-level retrieval accuracy

### Span Precision/Recall
- **Definition**: Overlap between retrieved and ground truth text spans
- **Calculation**: Character-level overlap normalized by span length
- **Significance**: Measures fine-grained retrieval quality

### Statistical Validation
- **Method**: Bootstrap sampling with 95% confidence intervals
- **Iterations**: Multiple runs for robust statistical inference
- **Output**: Publication-ready LaTeX tables with significance markers

## 🔬 Research Contributions

1. **Novel Metadata Enrichment**: First systematic study of metadata-enhanced chunking for legal RAG
2. **DPO for Legal LLMs**: Direct Preference Optimization fine-tuning for improved legal question answering
3. **Comprehensive Evaluation**: Multi-metric evaluation framework with statistical validation
4. **Legal Domain Focus**: Specialized evaluation on Australian legal documents with multiple datasets
5. **Open Source Implementation**: Complete pipeline available for reproduction with pre-trained models

## 📚 Dependencies

### Core Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 1.13.0+ (CPU or GPU)
- **Transformers**: 4.21.0+ (Hugging Face)
- **NumPy/SciPy**: Scientific computing

### RAG Components
- **sentence-transformers**: Dense embeddings
- **faiss-cpu/gpu**: Vector similarity search
- **rank-bm25**: Sparse retrieval
- **PyPDF2**: PDF text extraction

### Analysis & Visualization
- **matplotlib/seaborn**: Plotting
- **pandas**: Data manipulation
- **scipy/statsmodels**: Statistical analysis

### Optional Dependencies
- **GPU Support**: Install `torch` with CUDA and `faiss-gpu`
- **Web Scraping**: `selenium` for JavaScript-heavy sites
- **Advanced NLP**: `spacy` with language models

See `requirements.txt` for complete list with version specifications.

## 🐛 Troubleshooting

### Common Issues

**1. FAISS Installation Issues**
```bash
# If faiss-cpu fails, try:
pip install faiss-cpu --no-cache-dir
# Or for GPU:
pip install faiss-gpu
```

**2. PyTorch CUDA Issues**
```bash
# Install PyTorch with specific CUDA version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Memory Errors**
- Reduce `CHUNK_SIZE` in configuration
- Process documents in smaller batches
- Use CPU instead of GPU for large datasets

**4. PDF Extraction Errors**
```bash
# Install additional PDF libraries:
pip install pdfplumber pypdf
```

**5. Missing spaCy Model**
```bash
python -m spacy download en_core_web_sm
```

### Performance Optimization

**For Faster Processing**:
- Use GPU for embedding generation
- Enable multi-threading in FAISS
- Cache embeddings to disk
- Use smaller embedding models for testing

**For Lower Memory Usage**:
- Process documents in batches
- Use quantized models
- Reduce chunk overlap
- Clear cache between experiments

## 🤝 Contributing

This research project welcomes contributions in:
- Additional legal datasets
- Novel chunking strategies
- Improved evaluation metrics
- Statistical analysis methods

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{meta_enriched_rag_legal,
  title={Meta-Enriched RAG for Legal LLMs: Enhancing Retrieval with Document Metadata},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 📞 Contact

For questions about this research or collaboration opportunities, please open an issue or contact the research team.

---

## 📋 Expected Outputs

### After Running Complete Pipeline

```
project_root/
├── australian_legal_documents_final/     # 200 legal documents (PDFs)
├── australian_legal_data/
│   ├── generated_chunks/                 # Processed document chunks
│   ├── results_recursive/                # Baseline experiment results
│   └── results_meta_recursive/           # Enhanced experiment results
├── results/                              # Additional analysis results
├── logs/                                 # Execution logs
└── stochastic_ranges_table.tex          # Final statistical results
```

### Key Result Files
- **`stochastic_ranges_table.tex`**: LaTeX table with statistical analysis
- **Performance plots**: Line plots showing metrics across k values
- **Evaluation JSONs**: Detailed per-query and macro results
- **Metadata files**: Document and chunk metadata
- **Log files**: Execution logs for debugging

### Interpreting Results
- **DRM (Document Retrieval Mismatch)**: Lower is better (0.0 = perfect)
- **Span Recall**: Higher is better (measures text-level accuracy)
- **Confidence Intervals**: Statistical significance of improvements
- **k-values**: Number of retrieved documents (1, 2, 4, 8, 16, 32, 64)

## 🔄 Reproducing Paper Results

To reproduce the exact results from the paper:

1. **Use the same datasets**: Australian Legal (146 queries) + MAUD + Privacy QA
2. **Run baseline experiments**: Recursive chunking with dense+sparse retrieval
3. **Run enhanced experiments**: Meta-recursive with metadata enrichment
4. **Generate statistics**: Bootstrap analysis with 95% confidence intervals
5. **Create visualizations**: Performance plots and comparison tables

Expected runtime: 
- **With pre-computed data**: 30 minutes (evaluation only)
- **From scratch**: 2-4 hours on modern hardware with GPU acceleration

---

**Note**: This project is part of ongoing research in legal AI and document retrieval systems. Results and methodologies are subject to peer review and continuous improvement.