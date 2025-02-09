# Translation Service Cheatsheet
Created by: kaxm23
Created on: 2025-02-09 10:07:52 UTC

## 1. API Endpoints

### Translation
```http
POST /api/v1/translate
{
  "text": "Hello world",
  "source_lang": "en",
  "target_lang": "es",
  "notify_email": true
}

GET /api/v1/translations/{translation_id}
GET /api/v1/translations/history