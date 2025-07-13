# ⚖️ LegalAgent

## Installation and Setup

1.  **Clone repository:**
    ```bash
    git clone https://github.com/hungtp2504/Legal-Agent.git
    cd LegalAgent
    ```

2.  **Configure the environment:**
    Create a `.env` file in the project's root directory and enter your API Key.
    ```bash
    GEMINI_API_KEY="..."
    ```
3. **Download and Set Up Data**
* [Download Data](https://drive.google.com/drive/folders/1iM-lVNrdHGSzze7ElzPqyLqL9Krw2j1N?usp=sharing)
* Unzip the file you just downloaded. You will find two subfolders: `parsed_json_output` and `chroma_db`.
* Copy both of these folders and paste them into the project's `data` directory.
  
4. **Install Necessary Libraries**

```bash
pip install -r requirements.txt
```

5. **Download the Embedding Model**
```bash
python save_model.py
```

6. **Running the Application**

```bash
streamlit run app.py
```
