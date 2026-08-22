# Software

## ساختار پروژه
- `app/models` — مدل‌های دیتابیس (SQLAlchemy)
- `app/schemas` — مدل‌های اعتبارسنجی ورودی/خروجی (Pydantic)
- `app/routers` — endpointها
- `app/auth` — منطق احراز هویت (JWT)

## راه‌اندازی
\`\`\`
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`