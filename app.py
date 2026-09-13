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
import hashlib
from datetime import datetime, timezone, date

import streamlit as st
import pandas as pd
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# CONFIG / CLIENT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Shadowing Tiếng Anh", page_icon="🎤", layout="wide")

AUDIO_BUCKET = "student_audios"
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]
LAYER1_MIN_RATIO = 0.50        # đọc đúng nội dung tối thiểu 50%
MIN_DURATION_RATIO = 0.60      # bản ghi phải dài ≥ 60% đoạn mẫu
CAP_PER_DAY = 5                # tối đa số lần CHẤM mỗi câu / ngày
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


GEMINI_KEY_URL = "https://aistudio.google.com/app/apikey"


def parse_keys(raw: str) -> list[str]:
    """Tách nhiều key: mỗi dòng / phẩy / chấm phẩy đều được."""
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[\n,;]+", raw) if p.strip()]


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


def save_gemini_keys(student_id: str, raw: str):
    """Lưu nguyên khối (nhiều dòng) vào cột gemini_api_key."""
    sb.table("assistantapp_students").update(
        {"gemini_api_key": raw.strip()}).eq("id", student_id).execute()


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
            st.session_state.gemini_keys = parse_keys(roster.get("gemini_api_key") or "")
        elif auth["role"] != "admin":
            st.session_state.clear()
            st.error(f"Role '{auth['role']}' không được hỗ trợ.")
            return
        st.rerun()


def _key_help():
    st.link_button("🔗 Mở Google AI Studio để lấy Key", GEMINI_KEY_URL,
                   use_container_width=True)
    st.markdown(
        """
##### 📌 Cách lấy Gemini API Key (miễn phí)
1. Bấm nút xanh ở trên → **đăng nhập bằng Gmail** của em.
2. Bấm **Create API key** (Tạo khoá API).
3. Bấm **Copy** để sao chép mã — mã bắt đầu bằng `AIza...`.
4. Quay lại đây, **dán vào ô bên dưới**.

> 💡 **Nên tạo 2–3 key từ 2–3 Gmail khác nhau**, dán cả 2–3 vào — **mỗi key một dòng**.
> App sẽ tự xoay vòng để em **không bị hết lượt giữa buổi** (lỗi 429).
> Key này miễn phí và chỉ của riêng em.
        """
    )


def _save_keys_form(stu, existing: list[str], btn_label: str):
    raw = st.text_area("Mỗi API Key một dòng", value="\n".join(existing),
                       height=120, key="keys_area",
                       placeholder="AIzaSy...key1\nAIzaSy...key2\nAIzaSy...key3")
    if st.button(btn_label, type="primary"):
        new_keys = parse_keys(raw)
        if new_keys:
            save_gemini_keys(stu["id"], "\n".join(new_keys))
            st.session_state.gemini_keys = new_keys
            st.success(f"Đã lưu {len(new_keys)} key.")
            st.rerun()
        else:
            st.error("Chưa có key hợp lệ.")


def require_gemini_key_ui() -> bool:
    stu = st.session_state.student
    keys = st.session_state.get("gemini_keys", [])
    if keys:
        with st.expander(f"🔑 Cập nhật API Key (đang có {len(keys)} key)"):
            _key_help()
            _save_keys_form(stu, keys, "Lưu Key")
        return True

    st.warning("Bạn chưa có Gemini API Key. Nhập 1 lần, hệ thống lưu lại cho buổi sau.")
    _key_help()
    _save_keys_form(stu, [], "Lưu Key")
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
   so với câu gốc.
3. issues: tối đa 2 TỪ (lấy đúng từ trong câu gốc) mà học sinh đọc SAI RÕ nhất
   (đọc phẳng, không nhấn trọng âm, sai âm). Mỗi cái:
   {"word": "<từ trong câu gốc>", "problem_vi": "<lỗi ngắn gọn tiếng Việt>"}.
   Nếu đọc tốt thì để [].
4. CHỈ trả về JSON, không thêm chữ nào khác:
{"is_cheating": boolean, "score": number (0-100), "feedback": "lời khuyên ngắn tiếng Việt",
"issues": [{"word": "...", "problem_vi": "..."}]}

Câu gốc học sinh cần đọc: "%s"
"""


def _list_models_str(genai) -> str:
    try:
        names = [m.name for m in genai.list_models()
                 if "generateContent" in getattr(m, "supported_generation_methods", [])]
        return ", ".join(names) if names else "(không có model nào hỗ trợ generateContent)"
    except Exception as e:
        return f"(không liệt kê được: {e})"


def score_with_gemini(api_key: str, wav_bytes: bytes, target_text: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    prompt = GEMINI_PROMPT % target_text
    audio = {"mime_type": "audio/wav", "data": wav_bytes}

    cached = st.session_state.get("gemini_model")
    candidates = ([cached] if cached else []) + \
                 [m for m in GEMINI_MODEL_CANDIDATES if m != cached]

    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content([prompt, audio])
            data = extract_json(resp.text)
            data["score"] = int(round(float(data.get("score", 0))))
            data["is_cheating"] = bool(data.get("is_cheating", False))
            data["feedback"] = str(data.get("feedback", ""))
            data["issues"] = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
            st.session_state.gemini_model = name          # nhớ model chạy được
            return data
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("429", "quota", "resource", "exhaust",
                                       "403", "permission", "api key", "api_key")):
                raise ValueError("KEY_INVALID")
            if any(k in msg for k in ("404", "not found", "not supported")):
                continue                                   # model này không có -> thử model kế
            raise
    # không model nào chạy được
    raise RuntimeError("Không có model Gemini nào dùng được với key này. "
                       "Model khả dụng: " + _list_models_str(genai))


def score_rotating(keys: list[str], wav_bytes: bytes, target_text: str) -> dict:
    """
    Xoay vòng nhiều key để chia tải. Mỗi lượt nộp bắt đầu từ key kế tiếp
    (round-robin). Key nào dính 429/403 -> tự nhảy sang key sau trong CÙNG lượt.
    Ném ValueError('ALL_KEYS_DEAD') nếu tất cả key đều hỏng.
    """
    if not keys:
        raise ValueError("ALL_KEYS_DEAD")
    n = len(keys)
    start = st.session_state.get("key_rr", 0) % n
    st.session_state.key_rr = (start + 1) % n          # lượt sau bắt đầu key khác
    order = [keys[(start + i) % n] for i in range(n)]
    for k in order:
        try:
            return score_with_gemini(k, wav_bytes, target_text)
        except ValueError:                              # KEY_INVALID -> thử key kế
            continue
    raise ValueError("ALL_KEYS_DEAD")


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


def scored_today(rec) -> int:
    """Số lần đã chấm HÔM NAY cho câu này (0 nếu chưa có / khác ngày)."""
    if not rec:
        return 0
    if rec.get("daily_date") == date.today().isoformat():
        return rec.get("daily_count") or 0
    return 0


def save_attempt(student_id, lesson_id, sentence_id, score, wav_bytes, feedback) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    today = date.today().isoformat()
    rec = get_record(student_id, lesson_id, sentence_id)

    if rec is None:
        best_url = upload_audio(student_id, lesson_id, sentence_id, "best", wav_bytes)
        worst_url = upload_audio(student_id, lesson_id, sentence_id, "worst", wav_bytes)
        sb.table("assistantapp_shadowing_records").insert({
            "student_id": student_id, "lesson_id": lesson_id, "sentence_id": sentence_id,
            "total_attempts": 1, "best_score": score, "worst_score": score,
            "best_audio_url": best_url, "worst_audio_url": worst_url,
            "best_feedback": feedback, "worst_feedback": feedback,
            "daily_count": 1, "daily_date": today, "last_updated": now,
        }).execute()
        return {"new_best": True, "new_worst": True, "total": 1, "today": 1}

    new_daily = (rec.get("daily_count") or 0) + 1 if rec.get("daily_date") == today else 1
    upd = {"total_attempts": rec["total_attempts"] + 1, "last_updated": now,
           "daily_count": new_daily, "daily_date": today}
    new_best = new_worst = False
    if rec["best_score"] is None or score > rec["best_score"]:
        upd["best_score"] = score
        upd["best_audio_url"] = upload_audio(student_id, lesson_id, sentence_id, "best", wav_bytes)
        upd["best_feedback"] = feedback
        new_best = True
    if rec["worst_score"] is None or score < rec["worst_score"]:
        upd["worst_score"] = score
        upd["worst_audio_url"] = upload_audio(student_id, lesson_id, sentence_id, "worst", wav_bytes)
        upd["worst_feedback"] = feedback
        new_worst = True
    sb.table("assistantapp_shadowing_records").update(upd).eq("id", rec["id"]).execute()
    return {"new_best": new_best, "new_worst": new_worst,
            "total": upd["total_attempts"], "today": new_daily}


def record_practice(student_id, lesson_id, sentence_id, seconds, match):
    """Ghi 1 lần LUYỆN hợp lệ (miễn phí, không chấm). Tăng practice_count + log."""
    now = datetime.now(timezone.utc).isoformat()
    entry = {"t": now, "sec": round(seconds, 1), "match": round(match, 2)}
    rec = get_record(student_id, lesson_id, sentence_id)
    if rec is None:
        sb.table("assistantapp_shadowing_records").insert({
            "student_id": student_id, "lesson_id": lesson_id, "sentence_id": sentence_id,
            "total_attempts": 0, "practice_count": 1, "practice_log": [entry],
            "last_updated": now,
        }).execute()
        return
    log = (rec.get("practice_log") or [])[-19:] + [entry]      # giữ 20 mục gần nhất
    sb.table("assistantapp_shadowing_records").update({
        "practice_count": (rec.get("practice_count") or 0) + 1,
        "practice_log": log, "last_updated": now,
    }).eq("id", rec["id"]).execute()


# ---------------------------------------------------------------------------
# UI: YOUTUBE "NGHE MẪU"
# ---------------------------------------------------------------------------
def youtube_clip_component(video_id: str, start: float, end: float, key: str):
    import streamlit.components.v1 as components
    html = f"""
    <div id="player_{key}"></div>
    <div style="margin-top:8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <button id="replay_{key}" style="padding:8px 14px; border-radius:8px;
        border:1px solid #ccc; background:#f6f6f6; cursor:pointer; font-size:15px;">
        🔁 Nghe lại đoạn</button>
      <label style="font-size:14px; cursor:pointer;">
        <input type="checkbox" id="loop_{key}"> Lặp tự động</label>
      <span style="font-size:13px; color:#888;">({start}s → {end}s)</span>
    </div>
    <script>
      var seg_{key} = {{start: {start}, end: {end}, player: null, loop: false}};
      function playSeg_{key}() {{
        var p = seg_{key}.player;
        if (p) {{ p.seekTo(seg_{key}.start, true); p.playVideo(); }}
      }}
      function load_{key}() {{
        new YT.Player('player_{key}', {{
          height: '220', width: '100%',
          videoId: '{video_id}',
          playerVars: {{start: {int(start)}, controls: 1}},
          events: {{
            'onReady': function(e) {{
              seg_{key}.player = e.target;
              playSeg_{key}();
              setInterval(function() {{
                var p = seg_{key}.player;
                if (!p || !p.getCurrentTime) return;
                if (p.getCurrentTime() >= seg_{key}.end) {{
                  if (seg_{key}.loop) {{ p.seekTo(seg_{key}.start, true); p.playVideo(); }}
                  else {{ p.pauseVideo(); }}
                }}
              }}, 150);
            }}
          }}
        }});
        document.getElementById('replay_{key}').onclick = playSeg_{key};
        document.getElementById('loop_{key}').onchange = function(ev) {{
          seg_{key}.loop = ev.target.checked;
          if (seg_{key}.loop) playSeg_{key}();
        }};
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
    components.html(html, height=310)


# ---------------------------------------------------------------------------
# PRONUNCIATION MAP  (render bản đồ ngữ điệu câu mẫu)
# ---------------------------------------------------------------------------
TONE_ARROW = {"level": "→", "rise": "↗", "fall": "↘", "fall-rise": "↘↗"}
CS_LABEL = {"linking": "nối âm", "elision": "nuốt âm", "assimilation": "đổi âm",
            "intrusion": "chèn âm", "gemination": "gộp âm"}


def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (w or "").lower())


def _bold_frag_in_word(word: str, frags: list[str]) -> str:
    """Bôi đậm âm tiết trọng âm bên trong 1 từ (frag đầu tiên khớp)."""
    for frag in frags:
        f = (frag or "").strip()
        if not f:
            continue
        m = re.search(re.escape(f), word, flags=re.I)
        if m:
            s, e = m.start(), m.end()
            return word[:s] + "**" + word[s:e] + "**" + word[e:]
    return word


def _render_chunk_words(chunk_text: str, stress: list[str], focus: str) -> str:
    fnorm = _norm_word(focus) if focus else None
    out = []
    for w in (chunk_text or "").split():
        core, trail = w, ""
        m = re.search(r"[.,!?;:]+$", w)
        if m:
            core, trail = w[:m.start()], w[m.start():]
        if fnorm and _norm_word(core) == fnorm:
            disp = f":red[**{core}**]"                       # sentence stress
        else:
            frags = [f for f in (stress or []) if re.search(re.escape(f), core, flags=re.I)]
            disp = _bold_frag_in_word(core, frags) if frags else core
        out.append(disp + trail)
    return " ".join(out)


def render_map_basic(m: dict) -> str:
    parts = [_render_chunk_words(c.get("text", ""), c.get("stress", []), m.get("focus"))
             for c in m.get("chunks", [])]
    return " ".join(parts)


def render_map_full(m: dict) -> str:
    segs = []
    for c in m.get("chunks", []):
        txt = _render_chunk_words(c.get("text", ""), c.get("stress", []), m.get("focus"))
        arrow = TONE_ARROW.get(c.get("tone", ""), "")
        segs.append(f"{txt} {arrow}".strip())
    return "  |  ".join(segs)


def render_connected_speech(m: dict) -> str:
    cs = m.get("connected_speech") or []
    if not cs:
        return ""
    items = [f'{c.get("span","")} → *{c.get("sounds_like","")}* '
             f'({CS_LABEL.get(c.get("type",""), c.get("type",""))})' for c in cs]
    return "🔗 " + "  ·  ".join(items)


def highlight_issue_words(text: str, words: list[str]) -> str:
    out = text
    for w in words:
        w = (w or "").strip()
        if not w:
            continue
        out = re.sub(r"(?i)\b(" + re.escape(w) + r")\b", r":red[**\1**]", out, count=1)
    return out


def extract_json_array(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        raise ValueError("Không tìm thấy JSON array")
    return json.loads(text[s:e + 1])


TEACHER_MAP_PROMPT = """Bạn là chuyên gia phát âm tiếng Anh cho học sinh Việt Nam (A1–B1).
Tôi ĐÍNH KÈM một file audio (mp3) và danh sách câu, mỗi dòng: ID | start–end (giây) | câu.
Hãy NGHE đúng đoạn từ start đến end của TỪNG câu, rồi mô tả CÁCH NGƯỜI NÓI TRONG AUDIO
thực sự đọc (ngắt nhịp ở đâu, nhấn từ nào, cuối mỗi nhịp lên hay xuống giọng).
DỰA TRÊN AUDIO — không suy đoán theo lý thuyết sách vở.

Với MỖI câu, trả về "bản đồ phát âm":
1. chunks: chia câu theo NHÓM THỞ mà người nói NGẮT thực tế trong audio. Mỗi chunk gồm:
   - text: nguyên văn phần chunk (giữ dấu câu).
   - stress: các ÂM TIẾT mà người nói NHẤN RÕ trong đoạn đó (vd "por" cho "important",
     "bought" cho từ một âm tiết).
   - tone: hướng giọng CUỐI chunk NGHE ĐƯỢC trong audio, chọn đúng 1:
       "level" = giữ ngang, "rise" = lên, "fall" = xuống, "fall-rise" = xuống-lên.
2. focus: từ mà người nói NHẤN MẠNH NHẤT trong câu (nghe rõ nhất).
3. connected_speech: chỗ NGHE THẤY nối/nuốt âm, mỗi cái {type, span, sounds_like, note_vi}.
   type ∈ linking, elision, assimilation, intrusion, gemination. Không rõ thì để [].
4. tip_vi: 1 câu tiếng Việt NGẮN — điểm quan trọng nhất khi nhại theo audio này.

Nếu đoạn nào nghe không rõ, cứ dựa trên phần nghe được, ĐỪNG bịa.
CHỈ trả về JSON array, không thêm chữ nào khác. Mẫu 1 phần tử:
[{"sentence_id":0,"chunks":[{"text":"He bought","stress":["bought"],"tone":"level"},
{"text":"and oranges.","stress":["or"],"tone":"fall"}],
"focus":"oranges","connected_speech":[],"tip_vi":"Nhại theo nhịp trong audio."}]

DANH SÁCH CÂU (ID | start–end giây | câu):
"""


def build_teacher_prompt(sentences: list[dict]) -> str:
    lines = [f'{s["sentence_id"]} | {s["start"]}–{s["end"]}s | {s["text"]}'
             for s in sentences]
    return TEACHER_MAP_PROMPT + "\n".join(lines)


def attach_maps(sentences: list[dict], json_text: str):
    """Trả (sentences, n_gắn). n = -1 nếu JSON lỗi."""
    if not (json_text or "").strip():
        return sentences, 0
    try:
        arr = extract_json_array(json_text)
    except Exception:
        return sentences, -1
    by_id = {}
    for item in arr:
        sid = item.get("sentence_id")
        if isinstance(sid, int):
            by_id[sid] = {
                "chunks": item.get("chunks", []),
                "focus": item.get("focus"),
                "connected_speech": item.get("connected_speech", []),
                "tip_vi": item.get("tip_vi", ""),
            }
    n = 0
    for s in sentences:
        if s["sentence_id"] in by_id:
            s["map"] = by_id[s["sentence_id"]]
            n += 1
    return sentences, n


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

    mode = st.radio("Chế độ hiển thị", ["Cơ bản", "Đầy đủ"],
                    horizontal=True, key="disp_mode")
    if mode == "Đầy đủ":
        st.caption("→ giữ ngang · ↗ lên · ↘ xuống · ↘↗ lửng lơ · "
                   "**đậm** = trọng âm từ · :red[đỏ] = nhấn mạnh nhất câu")

    for s in sentences:
        sid = s["sentence_id"]
        st.divider()
        st.markdown(f"**Câu {sid + 1}.**")
        m = s.get("map")
        if m and m.get("chunks"):
            st.markdown(render_map_full(m) if mode == "Đầy đủ" else render_map_basic(m))
            if mode == "Đầy đủ":
                cs = render_connected_speech(m)
                if cs:
                    st.caption(cs)
            if m.get("tip_vi"):
                st.caption("💡 " + m["tip_vi"])
        else:
            st.markdown(s["text"])
        st.caption(f"⏱ {s['start']}s → {s['end']}s")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("▶️ Nghe mẫu", key=f"play_{sid}"):
                st.session_state[f"show_{sid}"] = True
        with c2:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴 Thu âm", "⏹ Dừng", key=f"rec_{sid}")

        if st.session_state.get(f"show_{sid}"):
            youtube_clip_component(vid, s["start"], s["end"], key=f"{sid}")

        takes_key = f"takes_{sid}"
        takes = st.session_state.setdefault(takes_key, [])
        KEEP_VISIBLE = 5   # chỉ giữ vài bản gần nhất trên màn hình để chọn chấm

        # Thu xong bản MỚI -> TỰ lưu (không cần bấm Giữ). Nhận diện bản mới bằng chữ ký.
        if len(audio) > 0:
            wav = audiosegment_to_wav_bytes(audio)
            sig = hashlib.md5(wav).hexdigest()
            if st.session_state.get(f"lastsig_{sid}") != sig:
                st.session_state[f"lastsig_{sid}"] = sig
                rec_seconds = len(audio) / 1000.0
                ok, ratio, reason = free_gates(wav, rec_seconds, s)
                if ok:
                    record_practice(stu["id"], lesson["id"], sid, rec_seconds, ratio)
                    takes.append({"wav": wav, "sec": rec_seconds, "match": ratio})
                    del takes[:-KEEP_VISIBLE]
                    st.success("✅ Bản này hợp lệ — đã tự lưu, Luyện +1.")
                else:
                    st.error(reason + " (Thu lại nhé — bản này không được tính.)")

        # Danh sách bản gần đây -> chọn 1 bản để chấm
        if takes:
            st.markdown(f"**{len(takes)} bản gần nhất** (bản cũ vẫn được tính, chỉ ẩn bớt cho gọn):")
            for idx, t in enumerate(list(takes)):
                cc1, cc2 = st.columns([4, 1])
                cc1.audio(t["wav"], format="audio/wav")
                cc1.caption(f"Bản {idx + 1} · {t['sec']:.1f}s · khớp {t['match']:.0%}")
                if cc2.button("🗑 Xoá", key=f"del_{sid}_{idx}"):
                    takes.pop(idx); st.rerun()

            pick = st.radio("Chọn bản để chấm", range(len(takes)),
                            format_func=lambda i: f"Bản {i + 1}", horizontal=True,
                            key=f"pick_{sid}")
            if st.button("✅ Chấm bản đã chọn", key=f"grade_{sid}", type="primary"):
                _handle_grade(stu, lesson, s, takes[pick])


def free_gates(wav, rec_seconds, sentence):
    """Cửa miễn phí: đủ dài + đọc đúng nội dung. Trả (ok, ratio, lý_do_lỗi)."""
    sample_len = float(sentence["end"]) - float(sentence["start"])
    need = max(1.0, MIN_DURATION_RATIO * sample_len)
    if rec_seconds < need:
        return False, 0.0, (f"❌ Quá ngắn ({rec_seconds:.1f}s, cần ≥ {need:.1f}s). "
                            "Đọc trọn cả câu nhé — bản này không được tính.")
    ok1, ratio, heard = layer1_check(wav, sentence["text"])
    if not ok1:
        return False, ratio, (f"❌ Đọc sai nội dung / nói linh tinh (khớp {ratio:.0%}). "
                              f"Không tính. (Máy nghe được: “{heard or 'không rõ'}”)")
    return True, ratio, ""


def _handle_grade(stu, lesson, sentence, take):
    sid = sentence["sentence_id"]
    rec = get_record(stu["id"], lesson["id"], sid)
    if scored_today(rec) >= CAP_PER_DAY:
        st.warning(f"⏳ Câu này đã chấm đủ {CAP_PER_DAY} lần hôm nay. "
                   "Cứ luyện thêm, để dành lượt cho mai nhé.")
        return

    try:
        with st.spinner("Đang chấm ngữ điệu bằng AI..."):
            result = score_rotating(st.session_state.gemini_keys, take["wav"], sentence["text"])
    except ValueError:
        st.error("🔑 Tất cả API Key đều hết lượt/không hợp lệ. "
                 "Vào '🔑 Cập nhật API Key' thêm key mới (nên 2–3 key).")
        return
    except Exception as e:
        st.error(f"Lỗi khi chấm: {e}")
        return

    if result["is_cheating"]:
        st.error("🚨 Phát hiện gian lận dùng AI/máy đọc hộ — hủy, KHÔNG tính!")
        return

    info = save_attempt(stu["id"], lesson["id"], sid, result["score"], take["wav"], result["feedback"])
    score = result["score"]
    color = "green" if score >= 75 else ("orange" if score >= 50 else "red")
    st.markdown(f"### Điểm: :{color}[{score}/100]")
    st.write(result["feedback"])

    issues = result.get("issues") or []
    if issues:
        st.markdown("**Chỗ cần sửa:** " +
                    highlight_issue_words(sentence["text"], [i.get("word", "") for i in issues]))
        for i in issues:
            st.caption(f"• **{i.get('word','')}** — {i.get('problem_vi','')}")

    badges = []
    if info["new_best"]:
        badges.append("🏆 Kỷ lục cao mới!")
    if info["new_worst"]:
        badges.append("📉 Điểm thấp mới")
    if badges:
        st.info(" · ".join(badges))
    st.caption(f"Đã chấm câu này: {info['total']} lần · Hôm nay: {info['today']}/{CAP_PER_DAY}")


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

        st.markdown("---")
        st.markdown("#### 🗺️ Bản đồ phát âm (tùy chọn)")
        st.markdown(
            "1. **Đính kèm file mp3** của bài vào ChatGPT / Gemini.\n"
            "2. Bấm **Copy** khối dưới (đã có sẵn mốc giây từng câu) → dán vào cùng chỗ đó.\n"
            "3. AI nghe từng đoạn → trả JSON → **dán ngược** vào ô bên dưới → bấm Lưu.\n"
            "*(Bỏ trống cũng được — câu sẽ hiện trơn, không có bản đồ.)*"
        )
        with st.expander("📋 Khối để copy (prompt + danh sách câu)"):
            st.code(build_teacher_prompt(preview), language="text")

        map_json = st.text_area("Dán JSON bản đồ AI trả về vào đây", height=160,
                                key="map_json")

        if st.button("💾 Lưu bài học", type="primary"):
            if not (title and yt and preview):
                st.error("Thiếu tên bài / link / transcript.")
            else:
                sentences, n = attach_maps(preview, map_json)
                if n == -1:
                    st.warning("JSON bản đồ lỗi định dạng → lưu bài KHÔNG kèm bản đồ. "
                               "Kiểm tra lại JSON nếu muốn có bản đồ.")
                    sentences, n = preview, 0
                sb.table("assistantapp_shadowing_lessons").insert({
                    "assignment_id": assignment_id, "title": title,
                    "youtube_link": yt, "sentences": sentences,
                }).execute()
                st.session_state.preview = None
                msg = f"Đã lưu bài học! Gắn bản đồ cho {n}/{len(sentences)} câu."
                if n < len(sentences):
                    msg += " (Câu thiếu bản đồ sẽ hiện trơn.)"
                st.success(msg)


def humanize_since(iso: str) -> str:
    if not iso:
        return "—"
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - t
        s = int(delta.total_seconds())
        if s < 3600:
            return f"{max(1, s // 60)} phút trước"
        if s < 86400:
            return f"{s // 3600} giờ trước"
        return f"{s // 86400} ngày trước"
    except Exception:
        return iso[:10]


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
            "Luyện": r.get("practice_count") or 0,
            "Đã chấm": r["total_attempts"],
            "Cao nhất": r["best_score"],
            "Thấp nhất": r["worst_score"],
            "Lần gần nhất": humanize_since(r.get("last_updated")),
            "best_url": r["best_audio_url"],
            "worst_url": r["worst_audio_url"],
            "best_fb": r.get("best_feedback") or "",
            "worst_fb": r.get("worst_feedback") or "",
        })
    df = pd.DataFrame(rows).sort_values(["Học sinh", "Bài", "Câu"]).reset_index(drop=True)
    st.dataframe(df.drop(columns=["best_url", "worst_url", "best_fb", "worst_fb"]),
                 use_container_width=True, hide_index=True)

    st.markdown("#### 🔊 Nghe + xem AI chấm (Tốt nhất / Tệ nhất)")
    who = st.selectbox("Chọn dòng", df.index,
                       format_func=lambda i: f"{df.loc[i,'Học sinh']} · {df.loc[i,'Bài']} · Câu {df.loc[i,'Câu']}")
    a, b = st.columns(2)
    with a:
        st.markdown(f"🏆 **Tốt nhất — {df.loc[who,'Cao nhất']}/100**")
        if df.loc[who, "best_url"]:
            st.audio(df.loc[who, "best_url"])
        if df.loc[who, "best_fb"]:
            st.caption(df.loc[who, "best_fb"])
    with b:
        st.markdown(f"📉 **Tệ nhất — {df.loc[who,'Thấp nhất']}/100**")
        if df.loc[who, "worst_url"]:
            st.audio(df.loc[who, "worst_url"])
        if df.loc[who, "worst_fb"]:
            st.caption(df.loc[who, "worst_fb"])


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
