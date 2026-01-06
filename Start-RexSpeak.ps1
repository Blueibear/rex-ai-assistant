$env:REX_SPEAK_API_KEY = 'YOUR-SECRET-HERE'
$env:REX_SPEAK_STORAGE_URI = 'redis://127.0.0.1:6379/0'   # optional
$env:PYTHONPATH = (Resolve-Path .)

.\.venv\Scripts\python.exe -m waitress --listen=127.0.0.1:8000 wsgi:application
