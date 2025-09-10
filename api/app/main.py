from fastapi import FastAPI
from pydantic import BaseModel

from pydantic import BaseModel, Field

import re
import random
from google.cloud import storage

from google import genai
from google.genai import types
from google.cloud import bigquery
import json
import traceback
import logging
import datetime
import os
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import secretmanager
from connectors.postgres import CloudSQLPostgresConnector
from langchain_postgres.vectorstores import PGVector
import sqlalchemy
from langchain_google_vertexai import VertexAIEmbeddings

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
SECRET_ID_DB = os.environ.get('SECRET_ID_DB')
driver = os.environ.get("DRIVER", "pg8000")

def access_secret():
    # secret manager
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_DB}/versions/latest"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    db_secret = json.loads(payload)
    return db_secret

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location="us-central1",
)
embeddings = VertexAIEmbeddings(model="gemini-embedding-001")
db_secret  = access_secret()
connector = CloudSQLPostgresConnector(
    instance_name=db_secret["INSTANCE_CONNECTION_NAME"],
    user=db_secret["DB_USER"],
    password=db_secret["DB_PASS"],
    database=db_secret["DB_NAME"],
    driver=driver
)

system_instruction = """
    ### **System Instruction Prompt for Bank ABC Fraud Analyst Agent**

    Anda adalah "Fraud Analyst Assistant," sebuah AI asisten analitik internal untuk Bank ABC. Tugas utama Anda adalah membantu pengguna dengan menjawab pertanyaan apa pun yang berkaitan dengan data transaksi kartu kredit dan pengetahuan umum mengenai penipuan (*fraud*).

    Untuk melakukan ini, Anda harus secara cerdas memilih dan menggunakan fungsi yang tersedia untuk mengambil data dari *database* (BigQuery) atau dari basis pengetahuan dokumen (RAG). Anda juga harus menjawab dalam bahasa yang sama dengan yang digunakan oleh pengguna (Bahasa Indonesia atau Inggris).

    ---

    ### **Alat (Tools) yang Tersedia**

    Anda memiliki akses ke empat fungsi yang dibagi menjadi dua alur kerja utama:

    **1. Alur Kerja Kueri Data Transaksi (Database)**

    * **`retrieving_table_information() -> str`**
        * **Deskripsi:** Fungsi ini mengambil skema dan deskripsi detail dari tabel data transaksi. **Wajib dipanggil** sebelum membuat kueri SQL untuk memastikan Anda memahami nama kolom, tipe data, dan kontennya.
        * **Output:** String berisi deskripsi lengkap tabel dan kolom-kolomnya.

    * **`retrieving_data_db(query_syntax: str) -> dict`**
        * **Deskripsi:** Fungsi ini mengeksekusi *query* SQL yang valid pada *database* transaksi kartu kredit dan mengembalikan hasilnya.
        * **Input (`query_syntax`)**: Sebuah string yang berisi *query* SQL yang lengkap dan valid untuk Google BigQuery.
        * **Output**: Mengembalikan daftar (list) dari objek JSON, di mana setiap objek mewakili satu baris dari hasil *query*.

    **2. Alur Kerja Pengambilan Informasi (RAG)**

    * **`retrieving_rag_info() -> str`**
        * **Deskripsi:** Fungsi ini mengambil ringkasan dari dokumen PDF yang tersedia dalam basis pengetahuan. Gunakan ini untuk mendapatkan gambaran umum tentang informasi apa yang bisa dijawab melalui RAG.
        * **Output:** String berisi ringkasan dari dokumen yang tersedia.

    * **`retrieving_data_rag(question: str) -> list`**
        * **Deskripsi:** Fungsi ini mencari dan mengambil informasi yang relevan dari dokumen PDF berdasarkan pertanyaan yang diberikan.
        * **Input (`question`)**: Sebuah string berisi pertanyaan yang jelas dan spesifik untuk dijawab dari dokumen.
        * **Output**: Mengembalikan daftar (list) objek JSON yang berisi cuplikan konten (`page_content`) dan sumber dokumennya.

    ---

    ### **Aturan dan Prosedur**

    1.  **Identifikasi Maksud Pengguna**: Langkah pertama dan terpenting adalah menganalisis pertanyaan pengguna untuk menentukan apakah mereka membutuhkan:
        * **Data Kuantitatif/Spesifik**: Pertanyaan tentang jumlah, total, rata-rata, daftar transaksi, atau data spesifik dari *database* (misalnya, "Berapa total kerugian...?", "Tampilkan 5 transaksi terakhir...").
        * **Pengetahuan Umum/Konseptual**: Pertanyaan tentang definisi, metode, statistik umum, atau dampak penipuan (misalnya, "Apa itu *skimming*?", "Bagaimana dampak penipuan terhadap *merchant*?").

    2.  **Gunakan Alur Kerja yang Tepat**:
        * Jika maksudnya adalah **Data Kuantitatif/Spesifik**, Anda **wajib** mengikuti alur ini:
            1.  Panggil `retrieving_table_information()` untuk memahami struktur tabel.
            2.  Gunakan informasi tersebut untuk membuat *query* SQL yang akurat.
            3.  Panggil `retrieving_data_db()` dengan *query* yang telah Anda buat.
        * Jika maksudnya adalah **Pengetahuan Umum/Konseptual**, gunakan alur RAG:
            1.  Panggil `retrieving_data_rag()` dengan pertanyaan yang jelas yang disarikan dari permintaan pengguna. Anda bisa memanggil `retrieving_rag_info()` terlebih dahulu jika perlu konteks.

    3.  **Pembuatan Kueri SQL yang Tepat**:
        * **Filter Penipuan**: Hampir semua analisis akan memerlukan filter pada kolom `is_fraud`. Gunakan `WHERE is_fraud = 1` untuk data penipuan dan `WHERE is_fraud = 0` untuk data sah.
        * **Pencarian Teks**: Untuk kolom seperti `merchant` dan `job`, selalu gunakan `ILIKE` dengan pola wildcard (`%`) untuk pencarian yang tidak *case-sensitive* dan fleksibel. Contoh: `WHERE merchant ILIKE '%fraud_Donnelly LLC%'`.
        * **Agregasi**: Gunakan fungsi agregat SQL (`COUNT`, `SUM`, `AVG`, `MAX`, `MIN`) sesuai dengan pertanyaan pengguna.
        * **Tanggal**: Manfaatkan fungsi tanggal dan waktu SQL untuk memfilter berdasarkan periode tertentu pada kolom `trans_date_trans_time`.

    4.  **Bahasa**: Selalu deteksi bahasa yang digunakan pengguna (Bahasa Indonesia atau Inggris) dan berikan jawaban akhir Anda dalam bahasa yang sama.

    5.  **Sajikan Jawaban**:
        * Ubah hasil JSON dari `retrieving_data_db` atau hasil RAG dari `retrieving_data_rag` menjadi jawaban yang ringkas, jelas, dan mudah dipahami.
        * Jika tidak ada data yang ditemukan, informasikan kepada pengguna dengan sopan, misalnya, "Tidak ada data transaksi penipuan yang ditemukan untuk kriteria tersebut."
        * Jika memberikan informasi dari RAG, sebutkan sumbernya jika memungkinkan untuk kredibilitas.

    ---

    ### **Contoh Alur Kerja**

    **Contoh 1: Pertanyaan Data Kuantitatif (Database)**

    * **Pertanyaan Pengguna**: "Berapa total kerugian akibat penipuan di kategori *shopping*?"
    * **Proses Berpikir Anda**:
        1.  Tujuan: Menjumlahkan nilai transaksi (`amt`).
        2.  Kondisi: Transaksi harus penipuan (`is_fraud = 1`).
        3.  Filter: Kategori (`category`) harus berisi 'shopping'.
        4.  Alur: Ini adalah pertanyaan data, jadi saya harus menggunakan alur `retrieving_table_information()` -> `retrieving_data_db()`.
    * **Panggilan Fungsi yang Dihasilkan**:
        1.  `retrieving_table_information()`
        2.  `retrieving_data_db(query_syntax="SELECT SUM(amt) AS total_kerugian FROM `sandbox-project-471504.mekari_challenge_tabular_data.fraud_data` WHERE is_fraud = 1 AND category ILIKE '%shopping%'")`
    * **Hasil Fungsi (Contoh)**: `[{'total_kerugian': 54321.89}]`
    * **Bahasa yang digunakan user**: Bahasa Indonesia -> Jawab dengan bahasa indonesia
    * **Jawaban Akhir untuk Pengguna**: "Total kerugian akibat penipuan di kategori *shopping* adalah sebesar $54,321.89."

    **Contoh 2: Pertanyaan Pengetahuan Umum (RAG)**

    * **Pertanyaan Pengguna**: "What are the most common methods of credit card fraud?"
    * **Proses Berpikir Anda**:
        1.  Tujuan: Menjelaskan metode penipuan kartu kredit yang paling umum.
        2.  Sumber: Ini adalah pengetahuan konseptual, bukan data transaksi spesifik. Jawabannya pasti ada di dokumen PDF.
        3.  Alur: Saya akan menggunakan alur RAG dengan memanggil `retrieving_data_rag()`.
    * **Panggilan Fungsi yang Dihasilkan**:
        1.  `retrieving_data_rag(question="most common methods of credit card fraud")`
    * **Hasil Fungsi (Contoh)**: `[{"page_content": "The most common type of fraud is the use of a lost or stolen card, accounting for 48% of cases. Other methods include identity theft (15%), skimming (14%), and counterfeit cards (12%)...", "document_name": "Understanding Credit Card Frauds", ...}]`
    * **Bahasa yang digunakan user**: Bahasa Inggris -> Jawab dengan bahasa indonesia
    * **Jawaban Akhir untuk Pengguna**: "Based on the available documents, the most common methods of credit card fraud are the use of a lost or stolen card (accounting for 48% of cases), followed by identity theft (15%), skimming (14%), and counterfeit cards (12%)."""


# function for converting datetime from BQ
def date_converter(o):
    """
    A custom JSON serializer function to handle date and datetime objects.
    If the object is a date or datetime, it converts it to an ISO 8601 string.
    Otherwise, it raises a TypeError.
    """
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# function call to retrieve table information for query
def retrieving_table_information() -> str:
    """Retrieves table information that can be queried, please refer to this function before you create sql syntax to understand the table.

    Returns:
        Information regarding available table to help create a query syntax.
    """
    TABLE_DESCRIPTION = """
    Deskripsi Tabel: "**sandbox-project-471504.mekari_challenge_tabular_data.fraud_data**"

    Tabel ini berisi data transaksi kartu kredit yang dirancang untuk deteksi penipuan (*fraud detection*). Setiap baris merepresentasikan satu transaksi unik yang dilakukan oleh pemegang kartu. Tabel ini mencakup detail transaksi, informasi pribadi pemegang kartu, serta lokasi geografis terkait.

    ---

    ### **Kolom-kolom Tabel:**

    * **trans_date_trans_time**
        * **(Tipe Data: TIMESTAMP)**
        * **Deskripsi:** Waktu dan tanggal lengkap saat transaksi terjadi. Kolom ini mencatat momen pasti sebuah transaksi dilakukan.
        * **Contoh:** "2020-08-02 23:29:38.000000 UTC"

    * **cc_num**
        * **(Tipe Data: INTEGER)**
        * **Deskripsi:** Nomor kartu kredit unik yang digunakan untuk transaksi. Ini adalah pengenal utama untuk kartu yang terlibat.
        * **Contoh:** 6011399591920186

    * **merchant**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Nama pedagang atau toko tempat transaksi dilakukan. Untuk pencarian, sebaiknya gunakan `ILIKE` dengan pola `%` untuk menangani variasi nama dan tidak *case-sensitive*.
        * **Contoh:** "%fraud_Donnelly LLC%", "%fraud_Dooley Inc%"

    * **category**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Kategori atau jenis transaksi, seperti belanja, makanan, atau hiburan. Ini mengklasifikasikan sifat dari pengeluaran.
        * **Contoh:** "entertainment", "shopping_pos", "gas_transport"

    * **amt**
        * **(Tipe Data: FLOAT)**
        * **Deskripsi:** Jumlah atau nilai moneter dari transaksi.
        * **Contoh:** 19.44, 9.39, 60.71

    * **first**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Nama depan dari pemegang kartu kredit.
        * **Contoh:** "Maria"

    * **last**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Nama belakang dari pemegang kartu kredit.
        * **Contoh:** "Roy"

    * **gender**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Jenis kelamin pemegang kartu, biasanya diwakili oleh 'M' untuk pria (*Male*) atau 'F' untuk wanita (*Female*).
        * **Contoh:** "F"

    * **street**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Alamat jalan dari pemegang kartu kredit.
        * **Contoh:** "58665 Nicholas Ford Suite 348"

    * **city**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Kota tempat tinggal pemegang kartu kredit.
        * **Contoh:** "Sheffield"

    * **state**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Singkatan negara bagian dari alamat pemegang kartu kredit.
        * **Contoh:** "MA"

    * **zip**
        * **(Tipe Data: INTEGER)**
        * **Deskripsi:** Kode pos dari alamat pemegang kartu kredit.
        * **Contoh:** 1257

    * **lat**
        * **(Tipe Data: FLOAT)**
        * **Deskripsi:** Garis lintang (*latitude*) dari alamat pemegang kartu.
        * **Contoh:** 42.1001

    * **long**
        * **(Tipe Data: FLOAT)**
        * **Deskripsi:** Garis bujur (*longitude*) dari alamat pemegang kartu.
        * **Contoh:** -73.3611

    * **city_pop**
        * **(Tipe Data: INTEGER)**
        * **Deskripsi:** Populasi kota tempat tinggal pemegang kartu.
        * **Contoh:** 2121

    * **job**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Pekerjaan atau profesi dari pemegang kartu kredit. Gunakan `ILIKE` dengan pola `%` untuk pencarian yang fleksibel.
        * **Contoh:** "%Radio producer%"

    * **dob**
        * **(Tipe Data: DATE)**
        * **Deskripsi:** Tanggal lahir pemegang kartu.
        * **Format:** YYYY-MM-DD
        * **Contoh:** "1973-10-14"

    * **trans_num**
        * **(Tipe Data: STRING)**
        * **Deskripsi:** Pengenal unik untuk setiap transaksi. Ini adalah ID transaksi yang spesifik.
        * **Contoh:** "f40476d95acd240e32b37b4c4e34cf00"

    * **unix_time**
        * **(Tipe Data: INTEGER)**
        * **Deskripsi:** Waktu transaksi dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).
        * **Contoh:** 1375486178

    * **merch_lat**
        * **(Tipe Data: FLOAT)**
        * **Deskripsi:** Garis lintang (*latitude*) dari lokasi merchant.
        * **Contoh:** 42.256509

    * **merch_long**
        * **(Tipe Data: FLOAT)**
        * **Deskripsi:** Garis bujur (*longitude*) dari lokasi merchant.
        * **Contoh:** -72.465971

    * **is_fraud**
        * **(Tipe Data: INTEGER)**
        * **Deskripsi:** Sebuah *flag* atau penanda biner yang mengindikasikan apakah transaksi tersebut merupakan penipuan. Nilai `1` berarti transaksi adalah penipuan (*fraud*), dan `0` berarti transaksi sah.
        * **Contoh:** 0, 1

    ---

    ### **Relasi Penting:**

    * Setiap `trans_num` adalah unik untuk satu baris transaksi.
    * Seorang pemegang kartu (diidentifikasi oleh kombinasi `first` dan `last` atau `cc_num`) dapat memiliki banyak transaksi.
    * Kolom `lat` dan `long` merepresentasikan lokasi pemegang kartu, sedangkan `merch_lat` dan `merch_long` merepresentasikan lokasi *merchant*. Jarak antara kedua lokasi ini bisa menjadi indikator penting untuk analisis penipuan.
    """

    return TABLE_DESCRIPTION

# function call to retrieve pdf information for question formatting
def retrieving_rag_info() -> str:
    """Retrieves table information that can be queried, please refer to this function to have a sense whether the answer on the question will likely in the rag database or not

    Returns:
        Information regarding available information in the rag to help understanding the file available and to help create a good question for optimizing RAG.
    """
    FILE_INFO = """ 
    ** There is 2 pdf file available for RAG

    TITLE: Understanding Credit Card Frauds
    Authors: Tej Paul Bhatla, Vikram Prabhu & Amit Dua 
    This paper provides an overview of credit card fraud, detailing how it is committed, its impact on various stakeholders, and the technologies used for its prevention and management.
    Key Fraud Statistics and Methods
    •	Credit card fraud is defined as an individual using another person's credit card for personal reasons without the owner's or issuer's knowledge and with no intent to repay.
    •	Merchants are at a significantly higher risk from credit card fraud than cardholders. The rate of internet fraud is 12 to 15 times higher than in the "physical world".
    •	The most common type of fraud is the use of a lost or stolen card, accounting for 48% of cases. Other methods include identity theft (15%), skimming (14%), and counterfeit cards (12%).
    •	Fraud techniques are broadly classified into three categories: card-related, merchant-related, and internet-related frauds.
    o	Card-related: Includes application fraud, account takeover, and the creation of counterfeit cards through methods like skimming (electronically copying data from a card's magnetic stripe).
    o	Merchant-related: Involves merchant collusion, where owners or employees conspire to use customer information fraudulently.
    o	Internet-related: The internet provides an ideal environment for fraud through techniques like site cloning (copying legitimate websites) and creating false merchant sites to harvest card details.
    Impact of Fraud
    •	Cardholders: Generally the least impacted party, as consumer liability is often limited by law and bank policies.
    •	Merchants: The most affected party, especially in "card-not-present" transactions, as they must accept full liability for fraud losses. Costs include the value of the goods, shipping, chargeback fees from card associations, and damage to their reputation.
    •	Banks (Issuers/Acquirers): Incur administrative costs related to chargebacks and must make significant investments in sophisticated IT systems to prevent and detect fraud.
    Fraud Prevention and Management
    •	Various technologies exist to combat fraud, including Address Verification Systems (AVS), Card Verification Methods (CVM), and maintaining negative/positive lists of customers or card numbers.
    •	Recent developments in fraud management include rule-based systems, risk-scoring technologies, neural networks, biometrics, and smart cards with embedded chips (EMV).
    •	Effective fraud management aims to minimize the "total cost of fraud," which includes both the financial losses from fraud and the operational cost of prevention systems. This requires achieving a balance between insufficient screening (leading to high fraud losses) and excessive reviews (leading to high costs).

    TITLE: 2024 Report on Payment Fraud
    Authors: by the European Banking Authority (EBA) and the European Central Bank (ECB) 
    This report analyzes payment fraud data across the European Economic Area (EEA) for the periods H1 2022, H2 2022, and H1 2023. The analysis covers credit transfers, direct debits, card payments, cash withdrawals, and e-money transactions.
    Key Findings:
    •	Overall Fraud Levels: The total value of payment fraud across the EEA was €4.3 billion in 2022 and €2.0 billion in the first half of 2023.
    •	Fraud by Payment Type:
    o	In terms of value, credit transfers and card payments experienced the highest fraud levels. In H1 2023, fraudulent credit transfers amounted to €1.131 billion, while card fraud was €633 million.
    o	In terms of volume, card payments accounted for the largest number of fraudulent transactions, with 7.31 million in H1 2023.
    •	Primary Fraud Methods:
    o	For credit transfers, manipulation of the payer accounted for over half of the total fraud value.
    o	Card fraud was predominantly committed through the issuance of a payment order by a fraudster. For remote card fraud, the main cause was card details theft (64% by volume in H1 2023), while for non-remote fraud, it was the use of lost or stolen cards (over 50% by volume in H1 2023).
    •	Role of Strong Customer Authentication (SCA):
    o	SCA was applied to the majority of electronic payments, especially for credit transfers (around 77% by value).
    o	Transactions authenticated with SCA consistently showed lower fraud rates than those without SCA.
    o	Fraud rates for card payments are approximately ten times higher when the transaction counterpart is located outside the EEA, where SCA may not be required.
    •	Distribution of Losses:
    o	Payment service users (PSUs) bore over 80% of the total fraud losses for credit transfers.
    o	Losses from card payments and cash withdrawals were more evenly distributed, with PSUs bearing 45% and 51% of the losses, respectively, in H1 2023.
    •	Geographical Dimension:
    o	While most payment transactions were domestic, a majority of card payment fraud was cross-border (71% by value in H1 2023).
    o	A significant portion of fraudulent card payments (28% in H1 2023) involved transactions with counterparts outside the EEA.

    """

    return FILE_INFO

# function call to retrieve data from BQ
def retrieving_data_db(query_syntax: str) -> dict:
    """Retrieves data from the database by executing a SQL query. please consider the historical chat when creating query.

    Args:
        query_syntax (str): A valid Bigquery SQL query syntax string to execute against the database.

    Returns:
        A list of dictionaries, where each dictionary represents a row from the query results.
    """
    print(query_syntax)
    client = bigquery.Client()
    query_job = client.query(query_syntax)
    results_list = [dict(row) for row in query_job]
    json_output = json.dumps(results_list, indent=4, default=date_converter)

    return json_output

# function call to retrieve data from BQ
def retrieving_data_rag(question: str) -> list:
    """Retrieves data from the database by executing a SQL query. please consider the historical chat when creating query.

    Args:
        query_syntax (str): A valid Bigquery SQL query syntax string to execute against the database.

    Returns:
        A list of dictionary of retrieval result based on question
        consisting of
        result = [{
        "page_content" : information data,
        "document_name": source document on information data,
        "document_page": source page of document on information data,
        }
        ]
    """
    engine = connector.get_engine()
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="rag_data",
        connection=engine,
        use_jsonb=True
    )
    results = vector_store.similarity_search(query=question, k=4)
    final_res = []
    for result in results:
        result_temp = {
            "page_content" : result.page_content,
            "document_name": result.metadata['doc'],
            "document_page": result.metadata['page'],
        }
        final_res.append(result_temp)
    return final_res

class Chat_Data(BaseModel):
    session_id: str
    user_input: str

class Feedback_Data(BaseModel):
    session_id: str
    feedback_good_or_not: int # 0 means bad and 1 means good
    feedback_text: str    
     
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chatbot/ai-assistant")
def conversation(data_input:Chat_Data):
    # load metadata post from gcs
    storage_client = storage.Client()
    blob_name = f"gen-ai-memory/chat_history/{data_input.session_id}/history_{data_input.session_id}.json"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.get_blob(blob_name)
    if blob == None:
        history = []
    else:
        output = blob.download_as_text()
        chat_history_json = json.loads(output)
        history = []
        for history_temp in chat_history_json['chat_history']:
            history.append(
                types.Content(
                    role=history_temp['role'],
                    parts=[
                    types.Part.from_text(text=history_temp['chat'])
                    ]
                )
            )
    tools_query = [
        retrieving_data_db,
        retrieving_table_information,
        retrieving_rag_info,
        retrieving_data_rag
    ]
    model_name = "gemini-2.5-flash"  # @param ["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}

    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            tools=tools_query,
            system_instruction=system_instruction
        ),
        history=history
    )
    try:
        response = chat.send_message(data_input.user_input)

        # managing chat history into json for dump into gcs
        chat_history_list = []
        for chat_history_single in chat.get_history():
            if chat_history_single.parts[0].text != None:
                chat_history_temp ={"chat":chat_history_single.parts[0].text,
                                    "role":chat_history_single.role}
                chat_history_list.append(chat_history_temp)
            else:
                continue
        chat_history_json = {
            "session_id": data_input.session_id,
            "chat_history": chat_history_list
        }                        
        # dumping chat history into gcs
        chat_history_gcs_path = f"gen-ai-memory/chat_history/{data_input.session_id}/history_{data_input.session_id}.json"
        blob = bucket.blob(chat_history_gcs_path)
        blob.upload_from_string(json.dumps(chat_history_json), content_type="application/json")
        return {"ai_answer":response.text}
    except:
        logging.exception(str(traceback.format_exc()))
        return {"ai_answer":"Terdapat kesalahan pada AI, mohon tunggu beberapa saat"}

@app.post("/chatbot/feedback-user")
def feedback(data_input:Feedback_Data):
    # load metadata post from gcs
    storage_client = storage.Client()
    blob_name = f"gen-ai-memory/chat_history/{data_input.session_id}/history_{data_input.session_id}.json"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.get_blob(blob_name)
    if blob == None:
        return "There is no historical data"
    else:
        output = blob.download_as_text()
        chat_history_json = json.loads(output)
        # add feedback on the latest answer
        chat_history_json['chat_history'][-1]['feedback_good_or_not'] = data_input.feedback_good_or_not
        chat_history_json['chat_history'][-1]['feedback_text'] = data_input.feedback_text                       
        # replacing the history json with the one that have feedback
        chat_history_gcs_path = f"gen-ai-memory/chat_history/{data_input.session_id}/history_{data_input.session_id}.json"
        blob = bucket.blob(chat_history_gcs_path)
        blob.upload_from_string(json.dumps(chat_history_json), content_type="application/json")
        return "feedback is stored"
   

# @app.post("/chatbot/upload_to_sql")
# def upload_memory_to_db(data_input:Feedback_Data):
#     db_secret = access_secret()

#     # prepare vectorstore
#     connector = CloudSQLPostgresConnector(
#         instance_name=db_secret["INSTANCE_CONNECTION_NAME"],
#         user=db_secret['DB_USER'],
#         password=db_secret["DB_PASS"],
#         database=db_secret["DB_NAME"],
#         driver=driver
#     )
#     engine = connector.get_engine()
#     insert_data = sqlalchemy.text("""INSERT INTO demo_2_chat (session_id, user_chat, ai_chat) values (:session_id, :user_chat, :ai)""")
#     with engine.connect() as db_conn:
#         # update job information
#         bind_params = [
#             sqlalchemy.sql.bindparam(key="session_id", value=uuid.UUID(session_global), type_=UUID(as_uuid=True)),
#             sqlalchemy.sql.bindparam(key="user_chat", value=prompt),
#             sqlalchemy.sql.bindparam(key="ai", value=hasil),
#         ]
#         db_conn.execute(insert_data.bindparams(*bind_params))
#         db_conn.commit()
#     connector.close()