# NLM مولد صوتی موازی

سیستم تولید هم‌زمان Audio Overview با چند حساب Google NotebookLM، به‌صورت کاملاً لوکال.

---

## قابلیت‌های اصلی

- تقسیم PDF به بخش‌های کوچک (دستی یا خودکار)
- اجرای هم‌زمان چند حساب NotebookLM با سهم‌بندی مستقل
- آپلود منابع ثابت (PDF، TXT، MD، DOCX) به تمام Notebookها
- جلوگیری از آپلود تکراری با SHA-256
- دانلود خودکار Audio Overview
- تبدیل M4A به MP3 با FFmpeg
- تبدیل صوت به متن با Whisper Small (کاملاً لوکال)
- رابط وب Streamlit با ۱۳ صفحه مدیریتی
- Runner مستقل از UI با PID file
- ادامه پس از Restart (idempotent)
- دیتابیس SQLite با WAL mode

---

## نصب

### ۱. پیش‌نیاز

- Python 3.12+
- FFmpeg (برای تبدیل صوت)
- Git (اختیاری)

### ۲. نصب وابستگی‌های Python

```bash
pip install -r requirements.txt
```

وابستگی NotebookLM در `requirements.txt` قرار دارد. برای Login در ویندوز،
استفاده از Edge پیشنهاد می‌شود و نیازی به دانلود Chromium نیست:

```powershell
notebooklm profile create account_01
notebooklm -p account_01 login --browser msedge
```

### ۳. نصب FFmpeg در ویندوز

**روش ۱ – winget:**
```powershell
winget install ffmpeg
```

**روش ۲ – دستی:**
1. دانلود از [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. استخراج به `C:\ffmpeg`
3. افزودن `C:\ffmpeg\bin` به متغیر محیطی PATH
4. یا مسیر `ffmpeg.exe` را در صفحه تنظیمات برنامه وارد کنید

---

## اجرا

### Streamlit UI

```bash
streamlit run app.py
```

مرورگر به‌صورت خودکار باز می‌شود: `http://localhost:8501`

### Runner مستقل

```bash
python runner.py
```

یا با حالت آزمایشی:

```bash
python runner.py --fake
```

**نکته:** دکمه «اجرای Runner» در صفحه داشبورد یا تنظیمات، Runner را به‌صورت خودکار از طریق `subprocess.Popen` اجرا می‌کند.

---

## Login حساب‌های NotebookLM

هر حساب با یک Chrome Profile مستقل مدیریت می‌شود.  
برنامه هیچ‌گاه ایمیل، رمز عبور، Cookie یا Token را ذخیره نمی‌کند.

### مرحله ۱: ثبت حساب در برنامه

صفحه **حساب‌های NotebookLM** → «افزودن حساب جدید» را باز کنید.  
نام Profile را وارد کنید، مثلاً: `account_01`

### مرحله ۲: Login در مرورگر

دستورات زیر را در Terminal اجرا کنید:

```bash
notebooklm profile create account_01
notebooklm -p account_01 login
```

در ویندوز، از مرورگر Edge استفاده کنید اگر Chromium مشکل داشت:

```bash
notebooklm -p account_01 --browser edge login
```

### مرحله ۳: تأیید اتصال

در صفحه حساب‌ها، دکمه «بررسی اتصال» را بزنید.  
وضعیت باید `ACTIVE` شود.

---

## دانلود مدل Whisper

اولین اجرای Transcription، مدل را به‌صورت خودکار دانلود می‌کند.  
مدل پیش‌فرض `small` حدود ۴۶۰ مگابایت است.

برای پیش‌دانلود دستی:

```python
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu")
```

مدل در پوشه `~/.cache/huggingface/hub/` ذخیره می‌شود.

---

## ساختار فایل‌ها

```
app.py                   # Streamlit UI (13 صفحه)
runner.py                # Runner مستقل
database.py              # SQLite access layer (19+ جدول)
models.py                # Enums و dataclasses
settings.py              # تنظیمات مرکزی
pdf_service.py           # تقسیم PDF
notebook_service.py      # NotebookLM client (Fake + Real)
account_service.py       # مدیریت Login و حذف امن
allocation_service.py    # توزیع سهمی Jobها
shared_source_service.py # منابع ثابت + deduplication
audio_service.py         # FFmpeg تبدیل صوت
transcription_service.py # Whisper تبدیل به متن
file_service.py          # ZIP، مدیریت فایل
requirements.txt
README.md

data/
  database.sqlite3
  runtime/
    runner.pid
    runner.log
  accounts/
  shared_sources/
    global/
    projects/
  projects/
    <slug>/
      original/original.pdf
      chunks/001_pages_001_010.pdf
      audio_original/001_pages_001_010.m4a
      audio_mp3/001_pages_001_010.mp3
      transcripts/001_pages_001_010.txt
      exports/project_complete.zip
  tools/
    audio_conversion/
    transcriptions/
```

---

## تست‌ها

تأیید کامل نسخه در PowerShell:

```powershell
pip install -r requirements-dev.txt
powershell -ExecutionPolicy Bypass -File scripts\verify_release.ps1
```

این فرمان تست‌های pipeline با `FakeNotebookClient`، ماژول AI Folder، مجموعه
pytest، FFmpeg، importهای production و سازگاری dependencyها را بررسی می‌کند.
تمام مراحل باید با exit code صفر تمام شوند. تست واقعی NotebookLM به‌دلیل مصرف
سهمیه، جداگانه با یک پروژه کوچک انجام می‌شود.

پس از Login، تست واقعی یک‌chunk:

```powershell
python scripts\real_nlm_smoke.py --profile account_01
```

این تست یک shared source و یک PDF را آپلود می‌کند، آماده‌شدن هر دو را بررسی
می‌کند، ۶۰ ثانیه صبر می‌کند، صوت را تولید و دانلود می‌کند و notebook آزمایشی
را پس از موفقیت حذف می‌کند.

---

## معماری

```
Streamlit UI (app.py)
    │ reads DB
    │ subprocess.Popen → runner.py
    ↓
SQLite WAL (database.sqlite3)
    ↑ writes
runner.py
    ├── ParallelJobOrchestrator
    │   ├── AccountWorker (Process 1) → NotebookLM Profile 1
    │   │   ├── asyncio Slot 1
    │   │   ├── asyncio Slot 2
    │   │   └── asyncio Slot N
    │   └── AccountWorker (Process 2) → NotebookLM Profile 2
    ├── audio_service (FFmpeg)
    └── transcription_service (Whisper)
```

**اصل طراحی:** Streamlit تنظیمات و فرمان‌ها را در DB ثبت می‌کند؛ عملیات سنگین
NotebookLM در `runner.py` و subprocessهای آن اجرا می‌شود.

---

## محدودیت‌ها

- حداکثر منابع هر Notebook در NotebookLM: تقریباً ۵۰ فایل
- Whisper در حالت CPU کند است؛ GPU NVIDIA توصیه می‌شود
- Login اولیه NotebookLM به Edge/Chrome نیاز دارد؛ اجرای بعدی از profile ذخیره‌شده استفاده می‌کند
- سهمیه و محدودیت روزانه NotebookLM خارج از کنترل برنامه است
- برای جلوگیری از تولید ناقص، خرابی هر shared source همان job را متوقف می‌کند
- تبدیل صوت نیازمند FFmpeg نصب‌شده در سیستم است
- در ویندوز، `multiprocessing` نیاز به `if __name__ == "__main__":` دارد

---

## خطاهای رایج

| خطا | راه‌حل |
|-----|--------|
| `notebooklm not found` | `pip install "notebooklm-py[browser,cookies]"` |
| `FFmpeg not found` | نصب FFmpeg و افزودن به PATH |
| `AUTH_EXPIRED` | `notebooklm -p <name> login` دوباره اجرا کنید |
| `SQLite locked` | مطمئن شوید فقط یک Runner اجرا می‌شود |
| `CUDA not available` | از `device=cpu` استفاده کنید |

---

## پردازش پوشه با هوش مصنوعی (AI Folder Processor)

ماژول مستقلی برای تحلیل دسته‌ای فایل‌های یک پوشه با GapGPT / OpenAI-compatible API.

### نصب وابستگی‌های جدید

```bash
pip install openai python-dotenv charset-normalizer python-pptx openpyxl beautifulsoup4 filetype
```

در صورت اختلال اینترنت بین‌الملل (Mirror توسط Runflare، بدون ارتباط با GapGPT):

```bash
pip install openai -i https://mirror-pypi.runflare.com/simple
```

یا همه وابستگی‌ها از Mirror:

```bash
pip install -r requirements.txt -i https://mirror-pypi.runflare.com/simple
```

### تنظیم کلید API

روش ۱ – متغیر محیطی:
```bash
set GAPGPT_API_KEY=sk-...
```

روش ۲ – فایل `.env` (کپی از `.env.example`):
```env
GAPGPT_API_KEY=sk-...
```

روش ۳ – ورود موقت در رابط کاربری (تب «مدل و API»)

کلید API هرگز در دیتابیس، لاگ، کد یا فایل خروجی ذخیره نمی‌شود.

### استفاده

```bash
streamlit run app.py
```

از منوی کناری گزینه **🤖 AI Folder** را انتخاب کنید.

### فرمت‌های پشتیبانی‌شده

| فرمت | روش استخراج |
|------|------------|
| TXT, MD, RST, LOG, JSON, XML, YAML, INI, SQL, کد | Direct text |
| PDF | PyMuPDF |
| DOCX | python-docx |
| PPTX | python-pptx |
| XLSX | openpyxl (read-only) |
| HTML | BeautifulSoup |
| PNG, JPG, WEBP, BMP | Vision API (اختیاری) |
| MP3, WAV, M4A, AAC, FLAC, OGG | faster-whisper |
| MP4, MKV, MOV, WEBM | FFmpeg + faster-whisper |
| ZIP | Extract ایمن + پردازش داخلی (اختیاری) |

### فرمت‌های Skip شده

- فایل‌های باینری ناشناخته (`.bin`, `.exe`, `.dll`, ...)
- فایل‌های رمزدار PDF
- تصاویر در صورت غیرفعال بودن Vision
- صوت/ویدیو در صورت Skip بودن Audio Mode
- فایل‌های حساس (`.env`, `*.pem`, `id_rsa`, ...)

### ادامه پردازش قطع‌شده

از تب **اجرای صف** روی دکمه **ادامه** کنار Run قبلی کلیک کنید.  
Chunkهای تکمیل‌شده مجدداً ارسال نخواهند شد.

### تغییر مدل

در تب **مدل و API** نام هر مدل سازگار با OpenAI را وارد کنید:  
`gpt-5.2`, `claude-3-5-sonnet`, `gemini-2.0-flash`, ...

### تغییر هم‌زمانی

در تب **مدل و API** مقدار «هم‌زمانی (Concurrency)» را تنظیم کنید (۱ تا هر عدد دلخواه).

### اجرای تست‌ها

```bash
python test_ai_folder.py
```

تست‌های واقعی GapGPT (پولی) فقط با Flag دستی:

```bash
set RUN_GAPGPT_INTEGRATION_TESTS=true
python test_ai_folder.py
```

### فایل‌های جدید

| فایل | نقش |
|------|-----|
| `ai_folder_service.py` | اسکن پوشه و اعتبارسنجی |
| `file_extractor_service.py` | استخراج محتوا از انواع فایل |
| `prompt_service.py` | مدیریت Prompt و Placeholder |
| `chunking_service.py` | تقسیم متن و ادغام |
| `ai_api_service.py` | ارتباط با API (GapGPT/OpenAI) |
| `ai_batch_runner.py` | صف موازی، Retry، Resume |
| `secret_scanner.py` | شناسایی فایل‌ها و الگوهای حساس |
| `ai_folder_page.py` | رابط Streamlit (8 تب) |
| `test_ai_folder.py` | تست‌های کامل با FakeAIProvider |
| `.env.example` | نمونه تنظیمات API Key |

---

## جزوه‌ساز Word (Word Booklet Maker)

ماژول مستقل برای ساخت جزوه حرفه‌ای DOCX از فایل‌های متنی خروجی هوش مصنوعی.

### نصب وابستگی‌های جدید

```bash
pip install "python-docx>=1.1.0" "markdown-it-py>=3.0.0"
```

یا از requirements.txt:

```bash
pip install -r requirements.txt
```

### استفاده

```bash
streamlit run app.py
```

از منوی کناری گزینه **📖 جزوه‌ساز Word** را انتخاب کنید.

### ساختار خروجی

```
data/booklets/<slug>/
  output/               ← فایل نهایی booklet.docx
  manifests/            ← booklet_manifest.json
  previews/             ← merged_content.md (برای دیباگ)
  logs/                 ← build.log
```

### به‌روزرسانی فهرست مطالب (TOC)

فهرست مطالب به‌صورت یک **Field** واقعی Word درج می‌شود. برای به‌روزرسانی شماره صفحات:

1. فایل DOCX را در Microsoft Word باز کنید.
2. روی فهرست مطالب کلیک کنید.
3. کلید **F9** را بزنید یا **Update Field** را انتخاب کنید.
4. گزینه **Update entire table** را انتخاب کنید.

### فونت‌های پیشنهادی (فارسی)

| فونت | کاربرد | نصب |
|------|---------|-----|
| Vazirmatn | متن فارسی و Heading | [github.com/rastikerdar/vazirmatn](https://github.com/rastikerdar/vazirmatn) |
| B Nazanin | متن فارسی (جایگزین) | بسته فونت‌های فارسی |
| Calibri | متن انگلیسی | نصب پیش‌فرض Windows |
| Segoe UI Emoji | Emojiها | نصب پیش‌فرض Windows 10+ |

اگر فونت مورد نظر نصب نباشد، برنامه هشدار داده و از فونت جایگزین استفاده می‌کند.

### ویژگی‌های اصلی

- پشتیبانی از Markdown، Bold، Italic، فهرست، نقل‌قول و Code Block
- تشخیص و تبدیل JSON Table به جدول واقعی Word
- RTL کامل برای متن فارسی با Justify
- حفظ Medical Terminology انگلیسی (مثل STEMI، Troponin I)
- استایل‌های ویژه برای نکات: ⭐ 🚨 💡 💊 🔥 ✅ 📌 🏁 📊 🩺
- صفحه جلد با لوگو، عنوان، درس، دانشگاه و نام تهیه‌کننده
- فهرست مطالب (TOC) با Heading Style‌های واقعی Word
- Header، Footer و شماره صفحه (صفحه X از Y)
- مرتب‌سازی Natural، تاریخی، دستی و بر اساس شماره فایل
- ذخیره Manifest JSON با هش SHA-256 فایل‌ها
- سیستم Cache: اگر تغییری نداشته باشد، Rebuild نمی‌شود
- Preset برای ذخیره تنظیمات قالب

### فایل‌های ماژول

| فایل | مسئولیت |
|------|---------|
| `booklet_sort_service.py` | اسکن پوشه، مرتب‌سازی، استخراج Metadata |
| `booklet_parser.py` | Parse Markdown، تشخیص JSON Table، `ParsedChapter` |
| `docx_style_service.py` | Styleهای Word، RTL، فونت، رنگ، جدول |
| `docx_renderer.py` | ساخت جلد، TOC، فصل‌ها، جداول، Header/Footer |
| `booklet_manifest_service.py` | Manifest JSON، Snapshot Hash، Preview MD |
| `booklet_service.py` | هماهنگ‌کننده کل فرایند، خطاها، دیتابیس |
| `booklet_page.py` | رابط Streamlit (7 تب) |
| `tests/test_booklet_sort.py` | تست‌های مرتب‌سازی |
| `tests/test_booklet_parser.py` | تست‌های Parser و JSON Table |
| `tests/test_booklet_docx.py` | تست‌های ساخت DOCX |

### محدودیت‌های Parser

- فهرست‌های چندسطحی: تا حد امکان حفظ می‌شوند.
- Footnote: پشتیبانی نمی‌شود.
- جدول‌های Markdown (`| col | col |`): به جدول Word تبدیل نمی‌شوند (فقط JSON Table).
- HTML خام داخل Markdown: نادیده گرفته می‌شود.
- RTL/LTR ترکیبی در یک خط: با Best-effort مدیریت می‌شود.

### اجرای تست‌ها

```bash
python -m pytest tests/test_booklet_sort.py tests/test_booklet_parser.py tests/test_booklet_docx.py -v
```
