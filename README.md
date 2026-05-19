<div align="center">
  
  <h1>PURR: Pipeline for Understanding Requirements Reliability 🐾</h1>
  
  <p><b>A Privacy-First, Hybrid ML & LLM Pipeline for Automated Software Requirements Analysis</b></p>

  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Transformers-HuggingFace-F5A900?style=flat-square&logo=huggingface" alt="HuggingFace"></a>
  <a href="https://meta.com/llama/"><img src="https://img.shields.io/badge/LLM-Llama_3.1_(8B)-0466C8?style=flat-square" alt="Llama 3"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-success?style=flat-square" alt="License"></a>
</div>

<br>

## 📖 Overview

Poorly defined software requirements are the root cause of **37% of software project failures**. However, manually triaging thousands of lines of specification text is impossible for fast-moving Agile teams, and sending proprietary corporate architectures to external APIs (like ChatGPT) presents a massive data security risk.

**PURR** is a full-stack, entirely local machine learning pipeline designed to solve this. It combines the lightning-fast routing of classical ML (SVMs) with the deep semantic reasoning of Transformers (RoBERTa) and the generative power of local Large Language Models (Llama 3.1) to not only detect ambiguous requirements but **actively rewrite them** to strict ISO 29148 engineering standards.

## ✨ Key Features

* **🔒 100% Data Privacy:** Runs entirely on local hardware. Zero external API calls, ensuring corporate data never leaves the machine.
* **⚡ Hardware Optimized:** Utilizes 8-bit GGUF quantization and Streamlit Singleton caching (`@st.cache_resource`) to run an 8-billion parameter LLM smoothly on standard consumer CPUs without memory leaks.
* **🔄 Closed-Loop Validation:** Generated rewrites are automatically fed back into the pipeline's Critic model to mathematically prove ambiguity resolution.
* **📊 Custom Dataset:** Built on a newly contributed dataset of **4,000 uniquely annotated software requirements**.

---

## 🏗️ The 4-Stage Architecture

PURR employs a tiered "Mullet Protocol"—business in the front (fast, classical ML), party in the back (heavy, generative AI)—to optimize compute resources.

1. **The Gatekeeper (LinearSVC):** Instantly parses TF-IDF vectors to filter Functional vs. Non-Functional requirements.
2. **The Specialist (DistilRoBERTa):** Uses deep semantic understanding to categorize complex non-functional subtypes (e.g., Security, Usability, Performance).
3. **The Critic (DistilRoBERTa):** Analyzes linguistic context and intentionally retained stopwords to detect subjective ambiguity.
4. **The Generative Fixer (Llama 3.1 8B):** If flagged as ambiguous, the LLM actively rewrites the text, replacing vague adjectives with concrete, testable metrics.

<div align="center">
  <i>Architecture flowchart goes here</i>
</div>

---

## 🐈 Getting Started

### Prerequisites
* Python 3.10+
* 16GB+ RAM recommended for local LLM inference.
* Linux environments (Arch-based distributions like EndeavourOS or CachyOS) are highly recommended for optimal memory management during local model execution, though Windows/macOS are supported.

### Installation

1. **Clone the repository:**
  ```bash
  git clone [https://github.com/yourusername/PURR.git](https://github.com/yourusername/PURR.git)
  cd PURR
  ```

2. **Set up a virtual environment:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  
  ```


3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  
  ```


4. **Download the GGUF Model:**
* Download the `Llama-3.1-8B-Instruct.Q8_0.gguf` file from HuggingFace.
* Place it in the `models/llm/` directory.



### Running the Application

Launch the Streamlit dashboard:

  ```bash
  streamlit run app.py
  
  ```

*The UI will spin up locally on `localhost:8501`. From the dashboard, you can paste single requirements or upload batch CSV files for processing.*

---

## 🔬 Data Methodology & The "Split-First" Protocol

To guarantee absolute mathematical rigor and prevent data leakage, this project strictly adheres to a **Split-First Protocol**. The custom 4,000-row dataset was stratified into an 80/20 train/test split *prior* to any SMOTE class balancing. This ensures the testing partition remained completely unseen by the models during training.

The dataset and training notebooks can be found in the `/notebooks` directory.

---

## 👨‍💻 Author

**Mohamad Ajaz Imran** BSc (Hons) Software Engineering | Heriot-Watt University

[LinkedIn](https://www.linkedin.com/in/mohamad-ajaz/) | [Portfolio](https://mohamad-ajaz.vercel.app/)

*If you found this research useful, consider leaving a ⭐ on the repository!*
