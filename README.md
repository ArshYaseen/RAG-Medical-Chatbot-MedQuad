# Medical Question Answering Chatbot using RAG and MedQuad

This repository contains the code for a medical chatbot built using a Retrieval-Augmented Generation (RAG) approach. It leverages the MedQuad dataset, Sentence-Transformers for retrieval, FAISS for efficient indexing, the T5 transformer model for generation, and Dash for the user interface.

## Project Overview

The goal of this project is to create a chatbot capable of answering medical questions based on information contained within the MedQuad dataset. It employs a RAG pipeline:

1.  **Retrieval:** When a user asks a question, Sentence-Transformers encode the query into an embedding. FAISS is used to efficiently search a pre-computed index of embeddings (derived from the MedQuad dataset) to find the most relevant context(s).
2.  **Generation:** The retrieved context(s) and the original user query are then fed into a T5 text-to-text generation model. The model uses this combined information to generate a coherent and contextually relevant answer.
3.  **Interface:** A simple web interface built with Dash allows users to interact with the chatbot.

## Features

*   Answers medical questions based on the MedQuad dataset.
*   Utilizes state-of-the-art Sentence-BERT embeddings for semantic understanding.
*   Employs FAISS for fast and efficient context retrieval.
*   Uses the T5 model for high-quality text generation.
*   Provides a simple web-based UI using Dash.

## Technology Stack

*   **Programming Language:** Python 3
*   **Data Handling:** Pandas
*   **Embeddings & Retrieval:**
    *   `sentence-transformers` (specifically `all-MiniLM-L6-v2`)
    *   `faiss-cpu` (Facebook AI Similarity Search)
*   **Generation:**
    *   `transformers` (Hugging Face library)
    *   `T5` (specifically `t5-small` or `t5-large` - *adjust based on your final code*)
    *   `torch` (PyTorch)
*   **Web Framework:** Dash, Dash Bootstrap Components
*   **Deployment (for Colab):** `pyngrok`, `ngrok`

## Dataset

This project uses the **MedQuad** dataset, which contains a collection of medical questions and answers sourced from reputable NIH websites (like cancer.gov, niddk.nih.gov, GARD, MedlinePlus).

*   **Source:** (You might want to add a link to where you obtained MedQuad if possible, e.g., the original paper or a Kaggle link)
*   **Preprocessing:** Basic text cleaning (lowercasing, stripping whitespace) was applied to questions and answers. Rows with missing essential data were dropped.

## Architecture (RAG Pipeline)

1.  **Indexing (Offline):**
    *   Answers (or Questions, depending on your final implementation - *the Dash code seems to index Questions and retrieve corresponding Answers as context*) from the MedQuad dataset are loaded.
    *   Sentence-Transformer model encodes these texts into dense vector embeddings.
    *   A FAISS index is built using these embeddings for fast similarity search.
2.  **Querying (Online):**
    *   User submits a query via the Dash interface.
    *   The query is encoded using the same Sentence-Transformer model.
    *   FAISS index is searched using the query embedding to find the indices of the top-`k` most similar items (e.g., questions) from the indexing phase.
    *   The corresponding contexts (e.g., answers associated with the retrieved questions) are fetched from the original dataset.
    *   The retrieved context(s) and the original query are formatted and passed to the T5 generation model.
    *   T5 generates the final answer based on the provided context and query.
    *   The generated answer is displayed back to the user in the Dash UI.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    *   It's recommended to use a virtual environment.
    *   Install the required packages:
        ```bash
        pip install pandas faiss-cpu sentence-transformers transformers torch dash dash-bootstrap-components pyngrok
        # Or ideally, if you create a requirements.txt:
        # pip install -r requirements.txt
        ```
        *(Note: Ensure you have a `requirements.txt` file for best practice)*

3.  **Download Dataset:**
    *   Obtain the `medquad.csv` file.
    *   Place it in the root directory of the project (or update the path in the script).

4.  **Run the application:**
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file, e.g., `medical_chatbot.py`)*

5.  **Access the Chatbot:**
    *   **Local:** The application will typically be available at `http://127.0.0.1:8050/` or `http://localhost:8050/`. Check the console output when you run the script.
    *   **Google Colab:**
        *   Ensure you have uploaded `medquad.csv`.
        *   You need an `ngrok` account and authtoken. Add your token using `!ngrok authtoken YOUR_AUTHTOKEN`.
        *   Run the cell that starts the Dash app.
        *   An `ngrok` URL will be printed in the output (usually ending in `.ngrok.io`). Open this URL in your browser to access the chatbot.

## Potential Future Improvements

*   Use larger/more powerful embedding and generation models (e.g., `T5-large`, other Sentence-Transformer models).
*   Fine-tune the models on the MedQuad dataset for better domain adaptation.
*   Incorporate more diverse medical datasets.
*   Add chat history and context management for multi-turn conversations.
*   Implement more robust error handling and fallback mechanisms.
*   Add evaluation metrics (e.g., ROUGE, BLEU, semantic similarity) to assess answer quality.
*   Improve the UI/UX.

## Disclaimer

This chatbot is an experimental project based on the MedQuad dataset. **It is not a substitute for professional medical advice.** Do not rely on its answers for medical decisions. Always consult a qualified healthcare provider for any health concerns.

---

*Optional: Add License information (e.g., MIT License)*
