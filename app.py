"""
Luyện tập Shadowing Tiếng Anh — Streamlit + Supabase + Gemini 1.5 Flash
=======================================================================
Login: email + mật khẩu qua Supabase Auth (mọi role). Role trong `profiles`
tự quyết vào đâu: admin -> Teacher Dashboard, student -> Student App.

Cấu trúc (modular):
  CONFIG/CLIENT  : secrets, client service_role (data) + client anon (chỉ để verify login)
  HELPERS        : youtube id, parse json, storage path
  PARSER         : transcript thô -> từng câu + start/end
  AUTH           : login Supabase Auth, rẽ theo role, gate 'Đang học'
  LAYER 1        : lọc nhanh SpeechRecognition (khỏi tốn Gemini)
  GEMINI         : chấm ngữ điệu + anti-cheat giọng máy
  RECORDS        : best/worst + tối ưu bộ nhớ storage
  UI STUDENT / UI TEACHER / MAIN
"""

import io
import re
import json
import difflib
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# CONFIG / CLIENT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Shadowing Tiếng Anh", page_icon="🎤", layout="wide")

AUDIO_BUCKET = "student_audios"
GEMINI_MODEL = "gemini-1.5-flash"
LAYER1_MIN_RATIO = 0.35
DEFAULT_TAIL_SEC = 5.0


@st.cache_resource
def get_service_client() -> Client:
    """Data client dùng service_role -> bỏ qua RLS. GIỮ BÍ MẬT, chỉ ở backend."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_KEY"])


def make_anon_client() -> Client:
    """Client anon MỚI mỗi lần verify login — không cache để tránh lẫn session giữa người dùng."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


sb = get_service_client()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def youtube_id(link: str) -> str:
    if not link:
        return ""
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", link)
    return m.group(1) if m else link.strip()


def extract_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("Không tìm thấy JSON trong output Gemini")
    return json.loads(text[s:e + 1])


def storage_path(student_id: str, lesson_id: str, sentence_id: int, kind: str) -> str:
    """Path CỐ ĐỊNH -> upsert đè chính nó, không đẻ file rác."""
    return f"{student_id}/{lesson_id}/{sentence_id}_{kind}.wav"


def public_url(path: str) -> str:
    return sb.storage.from_(AUDIO_BUCKET).get_public_url(path)


def audiosegment_to_wav_bytes(seg) -> bytes:
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TRANSCRIPT PARSER
# ---------------------------------------------------------------------------
TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,](\d{1,3}))?\]")


def _ts_to_sec(m: re.Match) -> float:
    g1, g2, g3, ms = m.group(1), m.group(2), m.group(3), m.group(4)
    if g3 is not None:
        h, mnt, s = int(g1), int(g2), int(g3)
    else:
        h, mnt, s = 0, int(g1), int(g2)
    total = h * 3600 + mnt * 60 + s
    if ms:
        total += int(ms) / (10 ** len(ms))
    return float(total)


def parse_transcript(raw: str) -> list[dict]:
    tokens = list(TS_RE.finditer(raw))
    if not tokens:
        return []

    frags = []
    for i, m in enumerate(tokens):
        start = _ts_to_sec(m)
        t_start = m.end()
        t_end = tokens[i + 1].start() if i + 1 < len(tokens) else len(raw)
        txt = re.sub(r"\s+", " ", raw[t_start:t_end].strip().replace("\n", " "))
        if txt:
            frags.append((start, txt))

    words = []
    for start, txt in frags:
        for w in txt.split():
            words.append((start, w))

    sentences, buf, buf_start = [], [], None
    for idx, (start, w) in enumerate(words):
        if not buf:
            buf_start = start
        buf.append(w)
        if re.search(r"[.!?][\"')\]]*$", w):
            nxt = words[idx + 1][0] if idx + 1 < len(words) else None
            end = nxt if nxt is not None else start + DEFAULT_TAIL_SEC
            est = max(1.5, len(buf) * 0.35)
            if end - buf_start < est:
                end = buf_start + est
            sentences.append({"text": " ".join(buf).strip(),
                              "start": round(buf_start, 2), "end": round(end, 2)})
            buf = []

    if buf:
        est = max(1.5, len(buf) * 0.35)
        sentences.append({"text": " ".join(buf).strip(),
                          "start": round(buf_start, 2), "end": round(buf_start + est, 2)})

    for i, s in enumerate(sentences):
        s["sentence_id"] = i
    return sentences


# ---------------------------------------------------------------------------
# AUTH  (Supabase Auth email + mật khẩu, rẽ theo role)
# ---------------------------------------------------------------------------
def do_login(email: str, password: str):
    """Trả (auth_dict, error). auth_dict gồm user_id, email, role, class_name, ho_ten."""
    email = email.strip().lower()
    if not email or not password:
        return None, "Nhập đủ email và mật khẩu."
    try:
        res = make_anon_client().auth.sign_in_with_password(
            {"email": email, "password": password})
    except Exception:
        return None, "Sai email hoặc mật khẩu."
    if not res or not res.user:
        return None, "Sai email hoặc mật khẩu."

    prof = (sb.table("profiles")
              .select("id, role, class_name, email, ho_ten")
              .eq("id", res.user.id).limit(1).execute())
    if not prof.data:
        return None, "Tài khoản chưa có hồ sơ trong profiles. Báo giáo viên."
    p = prof.data[0]
    return {"user_id": res.user.id, "email": res.user.email,
            "role": (p.get("role") or "").strip().lower(),
            "class_name": p.get("class_name"), "ho_ten": p.get("ho_ten")}, None


def find_student_by_profile(profile_id: str) -> dict | None:
    r = (sb.table("assistantapp_students").select("*")
           .eq("vocab_user_id", profile_id).limit(1).execute())
    return r.data[0] if r.data else None


def save_gemini_key(student_id: str, key: str):
    sb.table("assistantapp_students").update(
        {"gemini_api_key": key.strip()}).eq("id", student_id).execute()


def get_class_name(student: dict) -> str:
    if student.get("ten_lop"):
        return student["ten_lop"]
    return st.session_state.get("auth", {}).get("class_name") or "—"


def login_ui():
    st.subheader("🔐 Đăng nhập")
    email = st.text_input("Email")
    pw = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập", type="primary"):
        auth, err = do_login(email, pw)
        if err:
            st.error(err)
            return
        st.session_state.auth = auth

        if auth["role"] == "student":
            roster = find_student_by_profile(auth["user_id"])
            if not roster:
                st.session_state.clear()
                st.error("Tài khoản chưa gắn hồ sơ học sinh (vocab_user_id). Báo giáo viên.")
                return
            if (roster.get("trang_thai") or "").strip() != "Đang học":
                st.session_state.clear()
                st.error(f"Trạng thái '{roster.get('trang_thai')}' — chỉ HS 'Đang học' mới vào được.")
                return
            st.session_state.student = roster
            st.session_state.gemini_key = roster.get("gemini_api_key") or ""
        elif auth["role"] != "admin":
            st.session_state.clear()
            st.error(f"Role '{auth['role']}' không được hỗ trợ.")
            return
        st.rerun()


def require_gemini_key_ui() -> bool:
    stu = st.session_state.student
    if st.session_state.get("gemini_key"):
        with st.expander("🔑 Cập nhật API Key"):
            new_key = st.text_input("Nhập Key mới", type="password", key="upd_key")
            if st.button("Lưu Key mới"):
                if new_key.strip():
                    save_gemini_key(stu["id"], new_key)
                    st.session_state.gemini_key = new_key.strip()
                    st.success("Đã cập nhật Key.")
                    st.rerun()
        return True

    st.warning("Bạn chưa có Gemini API Key. Nhập 1 lần, hệ thống lưu lại cho buổi sau.")
    key = st.text_input("Gemini API Key", type="password")
    if st.button("Lưu Key"):
        if key.strip():
            save_gemini_key(stu["id"], key)
            st.session_state.gemini_key = key.strip()
            st.success("Đã lưu Key.")
            st.rerun()
        else:
            st.error("Key trống.")
    return False


# ---------------------------------------------------------------------------
# LAYER 1 FILTER
# ---------------------------------------------------------------------------
def layer1_check(wav_bytes: bytes, target_text: str):
    try:
        import speech_recognition as sr
    except Exception:
        return True, 1.0, ""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = r.record(source)
        heard = r.recognize_google(audio, language="en-US")
    except sr.UnknownValueError:
        return False, 0.0, ""
    except Exception:
        return True, 1.0, ""

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    ratio = difflib.SequenceMatcher(None, norm(heard), norm(target_text)).ratio()
    return (ratio >= LAYER1_MIN_RATIO), ratio, heard


# ---------------------------------------------------------------------------
# GEMINI SCORING
# ---------------------------------------------------------------------------
GEMINI_PROMPT = """Bạn là giám khảo IELTS chấm phát âm.
1. ANTI-CHEAT: Nghe xem đây là giọng người thật hay máy đọc (Google Translate/TTS).
   Dấu hiệu máy: không có tiếng thở, đều đều, mượt bất thường, không nhiễu nền tự nhiên.
   Nếu là máy đọc hộ, gán is_cheating = true.
2. CHẤM ĐIỂM: Đánh giá Word Stress (trọng âm từ) và Sentence Stress (ngữ điệu câu)
   so với câu gốc. Chỉ ra cụ thể từ nào đọc lướt, không nhấn đúng trọng âm.
3. CHỈ trả về JSON, không thêm chữ nào khác:
{"is_cheating": boolean, "score": number (0-100), "feedback": "lời khuyên chi tiết tiếng Việt"}

Câu gốc học sinh cần đọc: "%s"
"""


def score_with_gemini(api_key: str, wav_bytes: bytes, target_text: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content([
            GEMINI_PROMPT % target_text,
            {"mime_type": "audio/wav", "data": wav_bytes},
        ])
        data = extract_json(resp.text)
        data["score"] = int(round(float(data.get("score", 0))))
        data["is_cheating"] = bool(data.get("is_cheating", False))
        data["feedback"] = str(data.get("feedback", ""))
        return data
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("429", "quota", "resource", "exhaust",
                                   "403", "permission", "api key", "api_key")):
            raise ValueError("KEY_INVALID")
        raise


# ---------------------------------------------------------------------------
# RECORDS LOGIC
# ---------------------------------------------------------------------------
def get_record(student_id, lesson_id, sentence_id):
    r = (sb.table("assistantapp_shadowing_records").select("*")
           .eq("student_id", student_id).eq("lesson_id", lesson_id)
           .eq("sentence_id", sentence_id).limit(1).execute())
    return r.data[0] if r.data else None


def upload_audio(student_id, lesson_id, sentence_id, kind, wav_bytes) -> str:
    path = storage_path(student_id, lesson_id, sentence_id, kind)
    sb.storage.from_(AUDIO_BUCKET).upload(
        path, wav_bytes, {"content-type": "audio/wav", "upsert": "true"})
    return public_url(path)


def save_attempt(student_id, lesson_id, sentence_id, score, wav_bytes) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    rec = get_record(student_id, lesson_id, sentence_id)

    if rec is None:
        best_url = upload_audio(student_id, lesson_id, sentence_id, "best", wav_bytes)
        worst_url = upload_audio(student_id, lesson_id, sentence_id, "worst", wav_bytes)
        sb.table("assistantapp_shadowing_records").insert({
            "student_id": student_id, "lesson_id": lesson_id, "sentence_id": sentence_id,
            "total_attempts": 1, "best_score": score, "worst_score": score,
            "best_audio_url": best_url, "worst_audio_url": worst_url, "last_updated": now,
        }).execute()
        return {"new_best": True, "new_worst": True, "total": 1}

    upd = {"total_attempts": rec["total_attempts"] + 1, "last_updated": now}
    new_best = new_worst = False
    if rec["best_score"] is None or score > rec["best_score"]:
        upd["best_score"] = score
        upd["best_audio_url"] = upload_audio(student_id, lesson_id, sentence_id, "best", wav_bytes)
        new_best = True
    if rec["worst_score"] is None or score < rec["worst_score"]:
        upd["worst_score"] = score
        upd["worst_audio_url"] = upload_audio(student_id, lesson_id, sentence_id, "worst", wav_bytes)
        new_worst = True
    sb.table("assistantapp_shadowing_records").update(upd).eq("id", rec["id"]).execute()
    return {"new_best": new_best, "new_worst": new_worst, "total": upd["total_attempts"]}


# ---------------------------------------------------------------------------
# UI: YOUTUBE "NGHE MẪU"
# ---------------------------------------------------------------------------
def youtube_clip_component(video_id: str, start: float, end: float, key: str):
    import streamlit.components.v1 as components
    html = f"""
    <div id="player_{key}"></div>
    <script>
      function load_{key}() {{
        new YT.Player('player_{key}', {{
          height: '220', width: '100%',
          videoId: '{video_id}',
          playerVars: {{start: {int(start)}, autoplay: 1, controls: 1}},
          events: {{
            'onReady': function(e) {{
              e.target.seekTo({start}, true); e.target.playVideo();
              var iv = setInterval(function() {{
                if (e.target.getCurrentTime() >= {end}) {{
                  e.target.pauseVideo(); clearInterval(iv);
                }}
              }}, 200);
            }}
          }}
        }});
      }}
      if (window.YT && window.YT.Player) {{ load_{key}(); }}
      else {{
        var tag = document.createElement('script');
        tag.src = "https://www.youtube.com/iframe_api";
        document.head.appendChild(tag);
        window.onYouTubeIframeAPIReady = load_{key};
      }}
    </script>
    """
    components.html(html, height=250)


# ---------------------------------------------------------------------------
# UI STUDENT
# ---------------------------------------------------------------------------
def load_lessons() -> list[dict]:
    r = (sb.table("assistantapp_shadowing_lessons")
           .select("*").order("created_at", desc=True).execute())
    return r.data or []


def student_app():
    stu = st.session_state.student
    with st.sidebar:
        st.markdown(f"**HS:** {stu.get('ho_ten')}")
        st.markdown(f"**Lớp:** {get_class_name(stu)}")
        if st.button("Đăng xuất"):
            st.session_state.clear(); st.rerun()

    if not require_gemini_key_ui():
        return

    st.header("🎤 Luyện Shadowing")
    lessons = load_lessons()
    if not lessons:
        st.info("Chưa có bài học nào. Chờ giáo viên soạn bài nhé.")
        return

    titles = {l["id"]: l["title"] for l in lessons}
    lid = st.selectbox("Chọn bài học", options=list(titles), format_func=lambda x: titles[x])
    lesson = next(l for l in lessons if l["id"] == lid)
    vid = youtube_id(lesson["youtube_link"])
    sentences = lesson.get("sentences") or []

    for s in sentences:
        sid = s["sentence_id"]
        st.divider()
        st.markdown(f"**Câu {sid + 1}.** {s['text']}")
        st.caption(f"⏱ {s['start']}s → {s['end']}s")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("▶️ Nghe mẫu", key=f"play_{sid}"):
                youtube_clip_component(vid, s["start"], s["end"], key=f"{sid}")
        with c2:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴 Thu âm", "⏹ Dừng", key=f"rec_{sid}")
            if len(audio) > 0:
                wav = audiosegment_to_wav_bytes(audio)
                st.audio(wav, format="audio/wav")
                if st.button("📤 Nộp bài", key=f"submit_{sid}", type="primary"):
                    _handle_submit(stu, lesson, s, wav)


def _handle_submit(stu, lesson, sentence, wav):
    sid = sentence["sentence_id"]
    with st.spinner("Đang kiểm tra nhanh..."):
        ok1, ratio, heard = layer1_check(wav, sentence["text"])
    if not ok1:
        st.error(f"❌ Đọc linh tinh / sai nội dung (khớp {ratio:.0%}). "
                 f"KHÔNG tính lượt. (Máy nghe được: “{heard or 'không rõ'}”)")
        return

    try:
        with st.spinner("Đang chấm ngữ điệu bằng AI..."):
            result = score_with_gemini(st.session_state.gemini_key, wav, sentence["text"])
    except ValueError:
        st.error("🔑 Key hết lượt/Không hợp lệ, vui lòng cập nhật lại (mục 'Cập nhật API Key').")
        return
    except Exception as e:
        st.error(f"Lỗi khi chấm: {e}")
        return

    if result["is_cheating"]:
        st.error("🚨 Phát hiện gian lận dùng AI đọc hộ, hủy bài!")
        return

    info = save_attempt(stu["id"], lesson["id"], sid, result["score"], wav)
    score = result["score"]
    color = "green" if score >= 75 else ("orange" if score >= 50 else "red")
    st.markdown(f"### Điểm: :{color}[{score}/100]")
    st.write(result["feedback"])
    badges = []
    if info["new_best"]:
        badges.append("🏆 Kỷ lục cao mới!")
    if info["new_worst"]:
        badges.append("📉 Điểm thấp mới (đã lưu để đối chiếu)")
    if badges:
        st.info(" · ".join(badges))
    st.caption(f"Tổng số lần luyện câu này: {info['total']}")


# ---------------------------------------------------------------------------
# UI TEACHER
# ---------------------------------------------------------------------------
def load_assignments() -> list[dict]:
    r = (sb.table("assistantapp_homework_assignment")
           .select("id, noi_dung, buoi_hoc_id, doi_tuong, ngay_giao")
           .order("created_at", desc=True).limit(200).execute())
    return r.data or []


def teacher_compose():
    st.subheader("📝 Soạn bài Shadowing")
    assignments = load_assignments()
    opt = {"(không gắn bài tập)": None}
    for a in assignments:
        label = f"{(a.get('noi_dung') or '')[:50]} — {a.get('doi_tuong') or ''} ({a.get('ngay_giao') or ''})"
        opt[label] = a["id"]
    pick = st.selectbox("Gắn vào bài tập về nhà (tùy chọn)", list(opt))
    assignment_id = opt[pick]

    title = st.text_input("Tên bài")
    yt = st.text_input("Link YouTube")
    raw = st.text_area("Transcript thô (có timestamp [00:00:00] Text...)", height=220)

    if st.button("🔍 Xem trước câu"):
        st.session_state.preview = parse_transcript(raw)

    preview = st.session_state.get("preview")
    if preview:
        st.success(f"Gom được {len(preview)} câu:")
        st.dataframe(pd.DataFrame(preview)[["sentence_id", "start", "end", "text"]],
                     use_container_width=True, hide_index=True)
        if st.button("💾 Lưu bài học", type="primary"):
            if not (title and yt and preview):
                st.error("Thiếu tên bài / link / transcript.")
            else:
                sb.table("assistantapp_shadowing_lessons").insert({
                    "assignment_id": assignment_id, "title": title,
                    "youtube_link": yt, "sentences": preview,
                }).execute()
                st.session_state.preview = None
                st.success("Đã lưu bài học!")


def teacher_stats():
    st.subheader("📊 Thống kê tiến độ")
    recs = sb.table("assistantapp_shadowing_records").select("*").execute().data or []
    if not recs:
        st.info("Chưa có dữ liệu luyện tập.")
        return

    students = {s["id"]: s for s in
                (sb.table("assistantapp_students").select("id, ho_ten, nick_name").execute().data or [])}
    lessons = {l["id"]: l for l in
               (sb.table("assistantapp_shadowing_lessons").select("id, title").execute().data or [])}

    rows = []
    for r in recs:
        rows.append({
            "Học sinh": (students.get(r["student_id"], {}).get("ho_ten")
                         or students.get(r["student_id"], {}).get("nick_name") or r["student_id"]),
            "Bài": lessons.get(r["lesson_id"], {}).get("title", r["lesson_id"]),
            "Câu": r["sentence_id"] + 1,
            "Số lần": r["total_attempts"],
            "Cao nhất": r["best_score"],
            "Thấp nhất": r["worst_score"],
            "best_url": r["best_audio_url"],
            "worst_url": r["worst_audio_url"],
            "Cập nhật": r["last_updated"],
        })
    df = pd.DataFrame(rows).sort_values(["Học sinh", "Bài", "Câu"]).reset_index(drop=True)
    st.dataframe(df.drop(columns=["best_url", "worst_url"]),
                 use_container_width=True, hide_index=True)

    st.markdown("#### 🔊 Nghe audio Tốt nhất / Tệ nhất")
    who = st.selectbox("Chọn dòng để nghe", df.index,
                       format_func=lambda i: f"{df.loc[i,'Học sinh']} · {df.loc[i,'Bài']} · Câu {df.loc[i,'Câu']}")
    a, b = st.columns(2)
    with a:
        st.caption(f"🏆 Tốt nhất ({df.loc[who,'Cao nhất']})")
        if df.loc[who, "best_url"]:
            st.audio(df.loc[who, "best_url"])
    with b:
        st.caption(f"📉 Tệ nhất ({df.loc[who,'Thấp nhất']})")
        if df.loc[who, "worst_url"]:
            st.audio(df.loc[who, "worst_url"])


def teacher_dashboard():
    with st.sidebar:
        st.markdown(f"**Admin:** {st.session_state.get('auth', {}).get('email', '')}")
        if st.button("Đăng xuất"):
            st.session_state.clear(); st.rerun()
    st.header("👩‍🏫 Teacher Dashboard")
    tab1, tab2 = st.tabs(["Soạn bài", "Thống kê"])
    with tab1:
        teacher_compose()
    with tab2:
        teacher_stats()


# ---------------------------------------------------------------------------
# MAIN  (role tự quyết, không có lựa chọn phân hệ)
# ---------------------------------------------------------------------------
def main():
    st.title("🎤 Luyện tập Shadowing Tiếng Anh")

    if not st.session_state.get("auth"):
        login_ui()
        return

    role = st.session_state.auth.get("role")
    if role == "admin":
        teacher_dashboard()
    elif role == "student":
        if not st.session_state.get("student"):
            st.session_state.clear(); st.rerun()
        student_app()
    else:
        st.error("Role không hợp lệ.")
        if st.button("Đăng xuất"):
            st.session_state.clear(); st.rerun()


if __name__ == "__main__":
    main()
