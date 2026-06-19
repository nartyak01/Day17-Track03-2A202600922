# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Tài liệu này phản tư kết quả benchmark sau khi hoàn thiện lab trong `src/`. Chạy lệnh:

```bash
cd src
python -m pytest test_agents.py -v
python benchmark.py
```

---

## 1. Kết quả benchmark

### Standard benchmark (`data/conversations.json`)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2056 | 17196 | 0.11 | 0.21 | 0 | 0 |
| Advanced | 1949 | 24967 | 0.71 | 0.73 | 413 | 1 |

### Long-context stress benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 576 | 24713 | 0.00 | 0.10 | 0 | 0 |
| Advanced | 777 | 27351 | 1.00 | 1.00 | 296 | 28 |

**Tests:** 4/4 passed (`User.md`, compact trigger, cross-session recall, prompt load trên thread dài).

---

## 2. Vì sao Advanced có recall tốt hơn Baseline?

### Baseline — chỉ nhớ trong cùng thread

`BaselineAgent` lưu message trong `SessionState` theo `thread_id`. Khi benchmark hỏi recall ở thread mới (`{conv_id}-recall`), agent không có dữ liệu từ các cuộc trước → recall **0.11** (standard) và **0.00** (stress).

Đây là hành vi **đúng thiết kế**: baseline là mốc so sánh “không có long-term memory”.

### Advanced — ba lớp memory

`AdvancedAgent` kết hợp:

1. **Short-term (compact):** `CompactMemoryManager` — message gần + summary message cũ trong cùng thread.
2. **Persistent:** `UserProfileStore` → `state/profiles/<user>/User.md` — facts ổn định (tên, nghề, nơi ở, style…).
3. **Fact extraction:** `extract_profile_updates()` trích fact từ message tiếng Việt và `upsert_facts()`.

Khi sang thread recall mới, Advanced vẫn đọc `User.md` → recall **0.71** và **1.00** trên stress test (tên `DũngCT Stress`, `MLOps engineer`, `Đà Nẵng`, `3 bullet`).

### Câu chuyện tổng quát (theo Rubric)

1. Baseline không nhớ dài hạn → recall thấp.
2. Advanced thêm `User.md` → recall tăng mạnh.
3. Hội thoại dài làm prompt cost tăng.
4. Compact memory giúp kiểm soát context trong thread dài (28 compactions trên stress).
5. Hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail.

---

## 3. Vì sao Advanced có thể tốn hơn Baseline ở hội thoại ngắn?

Trên **standard benchmark**, Advanced có **prompt tokens cao hơn** (24967 vs 17196) dù recall tốt hơn nhiều.

Nguyên nhân chính:

| Nguồn chi phí | Baseline | Advanced |
|---------------|----------|----------|
| History trong thread | Cộng dồn toàn bộ messages | Chỉ `keep_messages` + summary sau compact |
| `User.md` | Không có | **Mỗi lượt** đọc profile (~100+ tokens) |
| Recall thread | Không có profile | Vẫn load `User.md` |

Ở benchmark standard (10 cuộc ngắn, nhiều recall question), **chi phí cố định của `User.md` + nhiều thread** có thể lớn hơn lợi ích compact (chỉ **1 compaction** trên toàn bộ standard set).

Điều này khớp với README lab: *ở hội thoại ngắn, Advanced có thể tốn hơn Baseline về token usage* — trade-off **recall vs token** là có chủ đích.

**Agent tokens** gần tương đương (1949 vs 2056) vì cả hai đều dùng offline path với câu trả lời deterministic.

---

## 4. Compact memory hoạt động ra sao trên stress test?

Stress benchmark (`advanced_long_context.json`) có **16 turn rất dài** về tin tức + preference + correction (Huế → Đà Nẵng, bỏ nhiễu PM/Hà Nội).

### Baseline trên stress thread

- Không compact → mỗi lượt tính prompt trên **toàn bộ** history.
- Prompt tokens: **24713** cho một thread dài.
- Recall: **0.00** — thread recall mới không có context cũ (và baseline không có `User.md`).

### Advanced trên stress thread

- **28 compactions** — compact kích hoạt nhiều lần khi vượt `compact_threshold_tokens` (mặc định 400).
- Message cũ được gom vào `summary`; chỉ giữ `compact_keep_messages` (mặc định 4) message gần nhất.
- Recall: **1.00** — facts ổn định nằm trong `User.md`, không phụ thuộc toàn bộ raw history.

### Test `test_compact_reduces_prompt_load_on_long_thread`

Test dùng `compact_threshold_tokens=60` và `keep_messages=2` → compact xảy ra sớm → Advanced **prompt tokens < Baseline** trên cùng 30 turn synthetic.

Benchmark stress dùng ngưỡng mặc định (400) và message thực tế rất dài → Advanced vẫn có prompt **27351 vs 24713**. Compact **không luôn** thắng Baseline về tổng prompt khi:

- `User.md` + summary vẫn được load mỗi lượt;
- ngưỡng compact cao → vài lượt đầu chưa nén;
- summary heuristic (nối chuỗi) vẫn có thể dài.

**Ý chính của compact:** giới hạn **độ phình của context trong thread**, tránh kéo nguyên văn mọi turn cũ — không phải luôn giảm tổng prompt so với baseline khi đã cộng thêm persistent memory.

---

## 5. Memory file tăng trưởng và rủi ro

### Kích thước file sau benchmark

| User | File | Bytes (approx.) |
|------|------|-----------------|
| `dungct` | `state/profiles/dungct/User.md` | 413 |
| `dungct_stress` | `state/profiles/dungct_stress/User.md` | 296 |

Memory growth là **chi phí thật**: mỗi fact ổn định thêm dòng trong markdown, tăng prompt mỗi lượt.

### Rủi ro đã thấy trong thực tế

1. **Regex extraction sai:** Một số câu hỏi recall (ví dụ chứa “tên gì”, “ở đâu”) có thể bị `extract_profile_updates()` match nhầm → fact lỗi trong `User.md` (ví dụ `name`/`location` bị ghi từ nội dung câu hỏi thay vì fact thật). Cần **confidence threshold** hoặc chỉ extract từ turn khai báo, không từ turn hỏi.

2. **Fact cũ vs correction:** Lab có case đính chính (Đà Nẵng → Huế, backend → MLOps). `upsert_facts()` ghi đè theo key — đúng hướng **conflict handling**, nhưng regex yếu vẫn có thể giữ fact sai nếu không match pattern correction.

3. **Summary mất chi tiết:** Compact dùng heuristic `summarize_messages()` (nối chuỗi), không phải LLM summary → tin tức dài (Artemis, X-59…) có thể bị cắt cụt trong thread; facts quan trọng nên ở `User.md` hoặc summary có cấu trúc.

4. **File phình theo thời gian:** Production cần rotation, decay, hoặc giới hạn số field trong profile.

---

## 6. Phân tách ba lớp memory

| Lớp | Thành phần | Mục đích | Baseline | Advanced |
|-----|------------|----------|----------|----------|
| Short-term | Messages trong thread | Follow-up trong cùng cuộc | Toàn bộ history | Recent + summary |
| Persistent | `User.md` | Cross-session recall | Không | Có |
| Compact | `CompactMemoryManager` | Giảm raw context thread dài | Không | Có |

---

## 7. Kết luận

Lab đạt mục tiêu thiết kế:

- **Baseline** quên qua session → recall thấp, không memory file, không compaction.
- **Advanced** nhớ qua `User.md` → recall cao, có memory growth, compact hoạt động trên thread dài.
- Trade-off rõ: **đổi token/complexity để có recall và kiểm soát context dài**.

Hướng cải thiện (bonus):

- Confidence threshold trước khi ghi `User.md`
- Memory decay cho field ít dùng
- Entity extraction có cấu trúc (JSON schema thay vì regex)
- Live agent với LangChain tools + summarization middleware

---

## 8. Lệnh tái chạy

```powershell
cd src
python -m pytest test_agents.py -v
python benchmark.py
```

Kết quả benchmark có thể thay đổi nhẹ giữa các lần chạy nếu logic offline hoặc dataset được cập nhật; số liệu trong bảng trên là snapshot tại thời điểm hoàn thành lab.
