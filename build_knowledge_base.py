import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma

# 1. Secure API Key Loading
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("Initializing Expanded Pharmacy Catalog...")

# 2. Expanded Synthetic Product Catalog
data = {
    "product_id": ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"],
    "product_name": [
        "AllergyRelief Non-Drowsy",
        "SleepWell PM",
        "CoughSoothe Max",
        "JointFlex Cream",
        "ClearVision Eye Drops",
        "SootheThroat Cough Drops",
        "EarRelief Drops",
        "Kids Cold & Mucus Syrup"
    ],
    "description": [
        "24-hour non-drowsy allergy relief pill. Active ingredient: Loratadine. Great for daytime focus and work.",
        "Nighttime sleep aid and pain reliever. Active ingredient: Diphenhydramine. Causes heavy drowsiness.",
        "Maximum strength cough syrup. Soothes dry throat and suppresses persistent cough.",
        "Topical pain relief cream for arthritis, muscle, and joint pain. Contains cooling Menthol.",
        "Fast-acting lubricating eye drops for dry, red, or itchy eyes. Great for screen fatigue or allergies.",
        "Honey-lemon flavored cough drops with menthol. Temporarily relieves minor sore throat and scratchiness.",
        "Ear drops to relieve pain and itching from swimmer's ear or general earache.",
        "Children's daytime cold and mucus relief syrup. Cherry flavor, alcohol-free."
    ],
    "price": [14.99, 9.99, 12.50, 18.00, 8.50, 4.25, 11.00, 13.99]
}
df = pd.DataFrame(data)

# 3. Format the Data for the AI
loader = DataFrameLoader(df, page_content_column="description")
docs = loader.load()

# 4. Initialize the Embedding Model
print("Connecting to Google AI for text embedding...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# 5. Build and Save the Vector Database
print("Translating text to vectors and building the Chroma database locally...")
vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./pharmacy_db")

print("\n✅ Success! Expanded Knowledge Base built and saved.")