import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

pdf_path = r"YOUR FILE ADDRESS"  # مسیر دقیق فایل را وارد کنید

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"فایل در مسیر زیر یافت نشد: {pdf_path}")

print("درحال خواندن pdf")
loader = PyPDFium2Loader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

print("در حال بارگذاری مدل Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("در حال ساخت پایگاه داده برداری...")
db = FAISS.from_documents(
    chunks,
    embedding=embeddings
)

query = input("چطور میتونم کمکتون کنم؟ ")

related_chunks = db.similarity_search(query, k=5)

context = "\n\n--- بخش مرتبط ---\n".join([doc.page_content for doc in related_chunks])

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.avalai.ir/v1",
    api_key="your-api-key"  # کلید API خود را اینجا قرار دهید
)

prompt_template = f"""
شما یک دستیار هوشمند، دقیق و متعهد به متن هستید. وظیفه شما پاسخ به سوال کاربر صرفاً بر اساس "اطلاعات منبع" ارائه شده در زیر است.

### قوانین پاسخ‌دهی:
1. فقط و فقط از اطلاعات موجود در "اطلاعات منبع" برای پاسخ استفاده کنید.
2. از حدس زدن، تحلیل‌های خارج از متن یا استفاده از دانش عمومی خود اکیداً خودداری کنید.
3. پاسخ را کاملاً شفاف، روان، دقیق و به زبان فارسی بنویسید.
4. اگر پاسخ سوال به طور مستقیم یا غیرمستقیم در "اطلاعات منبع" وجود ندارد، دقیقاً عبارت زیر را بنویسید:
   "اطلاعات کافی در فایل مورد نظر برای پاسخ به این سوال یافت نشد."

====================
اطلاعات منبع:
{context}
====================

سوال کاربر:
{query}

پاسخ دقیق شما:
"""

messages = [
    {
        "role": "system",
        "content": "You are a precise Persian RAG assistant. Maintain absolute fidelity to the provided context."
    },
    {
        "role": "user",
        "content": prompt_template
    }
]

print("\nدر حال پردازش پاسخ...")
response = llm.invoke(messages)

print("\n" + "="*40)
print("پاسخ:")
print(response.content)
print("="*40)