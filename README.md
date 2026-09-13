# 🎤 Luyện tập Shadowing Tiếng Anh

Streamlit + Supabase + Gemini 1.5 Flash.
**Đăng nhập: email + mật khẩu (Supabase Auth) cho mọi người.**
Role trong `profiles` tự quyết vào đâu:
- `admin` → Teacher Dashboard (soạn bài + thống kê)
- `student` → Student App (luyện shadowing)

## 1. Supabase (đã xong phần lớn)

1. SQL đã chạy: cột `gemini_api_key` + 2 bảng shadowing + RLS.
2. **Storage → New bucket** → `student_audios` → tick **Public**.
3. Lấy 2 key ở **Project Settings → API**:
   - `anon public` → `SUPABASE_ANON_KEY` (đã điền sẵn)
   - `service_role` → `SUPABASE_SERVICE_KEY` (**BÍ MẬT**, tự dán)
4. Tài khoản đăng nhập tạo ở **Authentication → Users** (hoặc app khác của bạn).
   Mỗi user cần 1 dòng trong `profiles` đúng `role`. HS còn phải có
   `assistantapp_students.vocab_user_id` trỏ về `profiles.id` và `trang_thai = 'Đang học'`.

## 2. Secrets

| Key | Ghi chú |
|---|---|
| SUPABASE_URL | đã điền |
| SUPABASE_ANON_KEY | public — chỉ để verify login. Đã điền |
| SUPABASE_SERVICE_KEY | service_role — BÍ MẬT, tự dán. Không commit |

File thật `.streamlit/secrets.toml` đã bị `.gitignore` → không lên GitHub.

## 3. Chạy local

    pip install -r requirements.txt
    # mở .streamlit/secrets.toml, dán service_role key vào
    streamlit run app.py

Cần ffmpeg + flac:
- macOS: brew install ffmpeg flac
- Ubuntu: sudo apt install ffmpeg flac

## 4. Deploy (Streamlit Community Cloud)

1. Push repo lên GitHub.
2. share.streamlit.io → New app → chọn repo → main file app.py.
3. Advanced → Secrets: dán 3 dòng secrets (điền service_role thật).
4. packages.txt đã có ffmpeg, flac.

## Ghi chú kỹ thuật
- 2 client: service_role cho data (bỏ qua RLS), anon chỉ để sign_in_with_password.
  Client anon tạo mới mỗi lần login → không lẫn session giữa người dùng.
- App chỉ đăng nhập, không tự đăng ký. Tạo user ở Supabase.
- Audio best/worst lưu path cố định → kỷ lục mới đè file cũ, không đẻ rác.
- Lớp lọc 1 (SpeechRecognition) chặn trước khi tốn quota Gemini; lỗi mạng → fail-open.
