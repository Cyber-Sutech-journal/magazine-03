# CyberSutech — SQLite + HTTP/HTTPS Wireshark Lab


# CyberSootec

A small educational web application for demonstrating
HTTP vs HTTPS traffic using Wireshark.

## Ports

HTTP: 8000
HTTPS: 8443


## اجرا

Python 3.9+:

```bash
python server.py
```

سرویس‌ها:

- HTTP: http://127.0.0.1:8000
- HTTPS: https://127.0.0.1:8443

با اولین اجرا فایل `cybersutech.db` به‌صورت خودکار ساخته می‌شود.

## دیتابیس

SQLite است و جدول `users` دارد:

- `id`
- `username`
- `email`
- `password_hash`
- `created_at`

پسورد به صورت plaintext ذخیره نمی‌شود و با PBKDF2-HMAC-SHA256 hash می‌شود.

## تست

1. برو به `http://127.0.0.1:8000/register`
2. با داده ساختگی ثبت‌نام کن، مثلاً:
   - username: `demo-user`
   - email: `demo@example.com`
   - password: `demo-pass`
3. بعد از ثبت‌نام به Dashboard می‌روی.
4. از Dashboard برگرد و با همان مشخصات Login کن.

## سناریوی Wireshark

برای HTTP:

```text
http://127.0.0.1:8000/login
```

فیلتر:

```text
tcp.port == 8000
```

در POST درخواست HTTP، داده‌های فرم (از جمله password آزمایشی) در ترافیک HTTP قابل مشاهده است.

برای HTTPS:

```text
https://127.0.0.1:8443/login
```

فیلتر:

```text
tcp.port == 8443
```

در این حالت محتوای HTTP داخل TLS رمزنگاری می‌شود و password به صورت plaintext در packet دیده نمی‌شود.

### نکته مهم
HTTPS بودن به معنی این نیست که دیتابیس رمزنگاری شده است؛ HTTPS از مسیر Browser تا Server محافظت می‌کند. در این پروژه password در سمت Server نیز به صورت hash ذخیره می‌شود.

## اگر پورت اشغال بود

Windows:

```cmd
netstat -ano | findstr :8000
netstat -ano | findstr :8443
```

سپس PID را در صورت نیاز ببندید:

```cmd
taskkill /PID YOUR_PID /F
```

این پروژه برای لَب محلی/آموزشی است و برای Production طراحی نشده است.
