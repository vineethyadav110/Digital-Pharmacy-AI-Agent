# 💊 AI-Powered Pharmacy Assistant

An intelligent, empathetic RAG (Retrieval-Augmented Generation) agent that helps patients find over-the-counter remedies using natural language.

## 🚀 Features
- **Semantic Search:** Understands slang and casual descriptions of symptoms.
- **Empathy-First Design:** Professional, caring responses with out-of-stock apologies.
- **Real-time Streaming:** Immediate UI feedback using Streamlit.
- **RAG Architecture:** Grounded in a local product database to prevent hallucinations.

## 🛠️ Installation & Setup

**Prerequisites:** You will need Python 3.9+ and a free Google Gemini API Key.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vineethyadav110/Digital_Pharmacy_agent.git](https://github.com/vineethyadav110/Digital_Pharmacy_agent.git)
   cd Digital_Pharmacy_agent
   
2. Create and Activate a Virtual Environment:
Mac/Linux:
Bash
python3 -m venv .venv
source .venv/bin/activate

--- Windows:
Bash
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies:
Bash
pip install -r requirements.txt

4. Set up your Environment Variables:

Get a free API key from Google AI Studio.

5. Create a new file in the root directory named exactly .env.

Add your key to the file like this:
GEMINI_API_KEY="your_actual_api_key_here"

6. Build the Local Vector Database:
Run this script first to embed the product data and generate the local pharmacy_db folder.

Bash
python build_knowledge_base.py
Launch the Web App:

Bash
streamlit run app.py


