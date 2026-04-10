# Ngày 6 — Hackathon: SPEC → Prototype → Demo

> Không có lecture mới. Hôm nay = chứng minh. SPEC là hypothesis, Prototype là evidence, Demo là convince.

---

## Tổng quan

```text
SÁNG (9:00-13:00)              CHIỀU (14:00-16:00)            DEMO DAY (16:00-18:00)
┌────────────────────┐        ┌────────────────────┐         ┌────────────────────┐
│   BUILD PROTOTYPE   │        │   POLISH + PREP     │         │  DEMO ROUND          │
│                     │        │                     │         │  (lần lượt/zone)    │
│  M1: Canvas check   │        │  Polish prototype    │         │  10 phút/nhóm        │
│  Build              │   →    │  Viết demo script    │    →    ├────────────────────┤
│  M2: Show 1 thứ     │        │  M4: Dry run + setup │         │  TOP TEAMS PRESENT  │
│  Build tiếp         │        │                     │         │  + TRAO GIẢI         │
│  M3: SPEC final     │        │                     │         │                     │
└────────────────────┘        └────────────────────┘         └────────────────────┘
```

---

## Timeline

| Giờ | Milestone | Nội dung | Ghi chú |
|-----|-----------|----------|---------|
| 9:00 | **M1** | Canvas check + SPEC draft review | Chưa có Canvas → bổ sung trước 9:30 |
| 9:15 | | **Build Prototype** | GV đi vòng: "Đang build gì? Stuck ở đâu?" |
| 11:00 | **M2** | **"Show ít nhất mock prototype"** — GV check từng nhóm | Chưa có → simplify scope / dùng tool mock nhanh |
| 11:15 | | Build tiếp | |
| 13:00 | **M3** | **SPEC final + demo flow draft** | Trước nghỉ trưa — chiều chỉ polish |
| 13:00–14:00 | | Nghỉ trưa | |
| 14:00 | | Polish + chuẩn bị demo | 15:00: "1h nữa Demo. Freeze code, focus narrative." |
| 15:30 | **M4** | Demo prep done + dry run | Mỗi nhóm 1 bàn: laptop + poster/slides |
| **16:00** | **M5** | **Demo round (10 phút/nhóm)** | Lần lượt trình bày trong zone, các nhóm khác nghe + feedback |
| **17:00** | **M6** | **Top teams present + trao giải + closing** | |

---

## Deliverables

| # | Deliverable | Loại | Deadline |
|---|-------------|------|----------|
| 1 | **SPEC final** — Canvas + 6 phần đầy đủ | Nhóm | 23:59 09/04/2026 |
| 2 | **Prototype** — link/file + mô tả + supported links | Nhóm | 23:59 09/04/2026 |
| 3 | **Demo** — poster/slides | Nhóm | 23:59 09/04/2026 |
| 4 | **Reflection** — role + đóng góp + reflection | Cá nhân | 23:59 09/04/2026 |

---

## Prototype — 3 levels

| Level | Mô tả | Ví dụ | Điểm |
|-------|-------|-------|------|
| **Sketch** | Vẽ/draft flow trên giấy, slides, whiteboard. Chưa build gì. | Vẽ user journey trên giấy: user mở app → nhập triệu chứng → AI gợi ý khoa → user chọn → đặt lịch | Đủ điểm |
| **Mock prototype** | UI/flow build được (HTML, app) nhưng chưa gắn AI thật. Dùng tools: Claude Artifacts, v0, Figma... | App HTML có form nhập triệu chứng, hiện kết quả mẫu, flow click được — nhưng chưa gọi API AI | **Bonus** |
| **Working prototype** | Có AI chạy thật. Input → AI xử lý → output. Demo live được. | App gọi Gemini API: user nhập triệu chứng → AI phân tích → trả gợi ý khoa + confidence score | **Bonus ++** |

**Checkpoint M2 (11:00):** mỗi nhóm phải có ít nhất **mock prototype**. Chưa có → simplify scope hoặc dùng tool mock nhanh (Antigravity, Claude, Figma).

**Lưu ý:**
- Sketch + Mock: vẫn phải có ít nhất 1 prompt/AI call chạy thật kèm theo (chạy riêng, show bên cạnh)
- Working prototype: dùng vibe-coding tools (Cursor, Claude Code, Replit Agent...) hoàn toàn OK
- Mọi level: mỗi người giải thích được phần mình làm. Không hiểu = 0 điểm demo cá nhân

Xem chi tiết tools: [`03-tools-guide/prototyping-tools.md`](03-tools-guide/prototyping-tools.md)

---

## SPEC — 6 phần

Dùng template: [`02-templates/01-spec-template.md`](02-templates/01-spec-template.md)

| # | Phần | Yêu cầu |
|---|------|---------|
| 1 | **AI Product Canvas** | 3 cột Value / Trust / Feasibility + learning signal. Auto hay aug? Data gì, loại gì, có marginal value? |
| 2 | **User Stories 4 paths** | 2–3 features × 4 paths: happy / low-confidence / failure / correction |
| 3 | **Eval metrics + threshold** | 3 metrics + threshold + red flag. Precision hay recall? Tại sao? |
| 4 | **Top 3 failure modes** | Mỗi failure: trigger → hậu quả → mitigation |
| 5 | **ROI 3 kịch bản** | Conservative / realistic / optimistic + kill criteria |
| 6 | **Mini AI spec** | 1 trang tóm tắt: giải gì, cho ai, auto/aug, quality, risk, data flywheel |

---

## Demo round (M5, 10 phút/nhóm)

**Cách chơi:**

1. Các nhóm trong cùng zone **lần lượt trình bày**, các nhóm còn lại ngồi nghe + feedback
2. Mỗi nhóm: **5 phút trình bày** + **5 phút Q&A và feedback** từ peer + GV
3. Sau mỗi nhóm trình bày, các nhóm khác **điền feedback form**

**4 tiêu chí feedback (chấm 1-5, mỗi mục 25%):**

| # | Tiêu chí | 1 = yếu | 3 = ổn | 5 = xuất sắc |
|---|----------|---------|--------|--------------|
| 1 | Demo quality | Crash/không demo được | Chạy được, Q&A cơ bản | Mượt + Q&A sắc bén |
| 2 | Presentation | Rời rạc, khó theo | Rõ ràng, đủ ý | Cuốn hút, mạch lạc, ấn tượng |
| 3 | Problem-solution fit | Bài toán mơ hồ | Rõ, giải pháp hợp lý | Insight sắc, muốn dùng ngay |
| 4 | AI product thinking | AI gắn vào vô nghĩa | AI hữu ích, UX ổn | AI là lợi thế cốt lõi, UX xuất sắc |

Thêm: điểm thưởng đặc biệt (technical excellence / UX / tích hợp thực tế / tác động xã hội) + góp ý cải thiện.

**Nộp đủ feedback** — điền form ngay sau mỗi nhóm, không để dồn cuối.

Xem chi tiết flow + form mẫu: [`hackathon-rules-E403.md`](hackathon-rules-E403.md)

---

## Checklist trước demo (M4, 15:30)

- [ ] Prototype ready (ít nhất mock)
- [ ] Demo script 5 phút: ai nói gì, show gì, thứ tự nào
- [ ] Poster/slides tóm tắt: Problem → Solution → Auto/Aug → Demo
- [ ] Mỗi người trả lời được: "Auto hay aug?", "Failure mode chính?", "Phần mình làm gì?"

---

## Scoring (chung Day 5 + Day 6 = 100 điểm)

| Hạng mục | Điểm | Loại | Khi nào |
|----------|------|------|---------|
| SPEC milestone | 25 | Nhóm (20) + cá nhân (5) | 23h59 09/04/2026 |
| Prototype milestone | 15 | Nhóm (10) + cá nhân (5) | 23h59 09/04/2026 |
| Demo Day | 25 | Nhóm (peer feedback 15 + feedback quality 7 + bonus 3) | Present 16:00, nộp file 23h59 09/04 |
| UX exercise | 10 | Cá nhân + bonus | UX workshop sáng D5 |
| Individual reflection | 25 | Cá nhân (role 10 + reflection 15) | 23h59 09/04/2026 |
| **Tổng** | **100** | | |

### Binary gates và penalties

| Tình huống | Hậu quả | Áp dụng cho |
|-----------|---------|-------------|
| Không nộp SPEC draft trước 23:59 D5 | -5 điểm từ tổng SPEC | Cả nhóm |
| Prototype không có AI call thật | Cap Prototype ở 4/10 | Cả nhóm |
| Không giải thích được phần mình trong demo | 0 điểm Prototype cá nhân (5 điểm) | Cá nhân |
| Không có commit trên group repo | 0 tất cả điểm cá nhân — SPEC + Prototype (10 điểm) | Cá nhân |
| Không nộp đủ feedback forms | 0 điểm "nộp đủ" (2 điểm Demo) | Cá nhân |
| Copy-paste reflection giữa thành viên | 0 toàn bộ reflection (15 điểm) | Tất cả người liên quan |

---

## Hướng dẫn nộp bài

**Deadline:** 23h59 09/04/2026 | **Nộp lên:** LMS | **Nộp link 2 public GitHub repo (cá nhân + nhóm)**

Mỗi người nộp **2 link repo public** lên LMS: 1 repo cá nhân + 1 repo nhóm.

### Repo cá nhân

```
MaHocVien-HoTen-Day06
```

Ví dụ: `AI20K001-NguyenVanA-Day06`

```
AI20K001-NguyenVanA-Day06/
│
├── reflection.md                  ← Role + đóng góp + reflection cá nhân
│
└── extras/                        ← Tùy chọn — nộp thêm để lấy bonus
    └── ...                        ← Prompt test logs, research notes, screenshot AI conversation,
                                      design iterations, hoặc bất kỳ output cá nhân nào khác
```

### Repo nhóm

```
NhomXX-Lop-Day06
```

Ví dụ: `Nhom01-403-Day06`

```
Nhom01-403-Day06/
│
├── spec-final.md                  ← SPEC 6 phần
├── prototype-readme.md            ← Mô tả prototype + link + phân công
└── demo-slides.pdf                ← Poster hoặc slides dùng khi present
```

- Cả 2 repo phải **public** — GV cần truy cập được để chấm
- **Repo cá nhân** chứa reflection — đây là căn cứ chính cho điểm cá nhân
- **Folder `extras/`** không bắt buộc — nộp thêm các output cá nhân khác (prompt test logs, design sketches, research notes, screenshot AI conversation...) sẽ được xét **bonus điểm** nếu nội dung có chất lượng và thể hiện quá trình tư duy
- **Repo nhóm** do nhóm cùng quản lý — mọi thành viên đều nộp link repo nhóm này lên LMS

### Chi tiết từng file

**spec-final.md** — SPEC 6 phần hoàn chỉnh:
1. AI Product Canvas (Value / Trust / Feasibility + learning signal)
2. User Stories 4 paths (happy / low-confidence / failure / correction)
3. Eval metrics + threshold (3 metrics, precision hay recall, red flag)
4. Top 3 failure modes (trigger → hậu quả → mitigation)
5. ROI 3 kịch bản (conservative / realistic / optimistic + kill criteria)
6. Mini AI spec (1 trang tóm tắt)

**prototype-readme.md** gồm:
- Mô tả prototype (2-3 câu)
- Level: Sketch / Mock / Working
- Link prototype (GitHub repo / Figma / deployed app / video nếu có)
- Tools và API đã dùng
- Phân công: ai làm gì

**demo-slides.pdf** — poster hoặc slides tóm tắt:
- Problem → Solution → Auto/Aug → Demo flow
- Đọc được từ xa 1-2 mét nếu in poster

**reflection.md** gồm 7 phần:
1. Role cụ thể trong nhóm (không viết chung chung "thành viên")
2. Phần phụ trách cụ thể (liệt kê 2-3 đóng góp có output rõ)
3. SPEC phần nào mạnh nhất, phần nào yếu nhất? Vì sao?
4. Đóng góp cụ thể khác với mục 2 (ví dụ: test prompt, debug, support)
5. 1 điều học được trong hackathon mà trước đó chưa biết
6. Nếu làm lại, đổi gì? (phải cụ thể, không viết "cố gắng hơn")
7. AI giúp gì? AI sai/mislead ở đâu? (bắt buộc nêu cả hai mặt)

---

## Ví dụ bài nộp tốt

### prototype-readme.md

```markdown
# Prototype — AI triage Vinmec

## Mô tả
Chatbot hỏi bệnh nhân 3-5 câu về triệu chứng, gợi ý top 3 chuyên khoa phù hợp
kèm confidence score. Bệnh nhân chọn hoặc gặp lễ tân.

## Level: Mock prototype
- UI build bằng Claude Artifacts (HTML/CSS/JS)
- 1 flow chính chạy thật với Gemini API: nhập triệu chứng → nhận gợi ý khoa

## Links
- Prototype: https://claude.site/artifacts/xxx
- Prompt test log: xem file `prototype/prompt-tests.md`
- Video demo (backup): https://drive.google.com/xxx

## Tools
- UI: Claude Artifacts
- AI: Google Gemini 2.0 Flash (via Google AI Studio)
- Prompt: system prompt + few-shot examples cho 10 triệu chứng phổ biến

## Phân công
| Thành viên | Phần | Output |
|-----------|------|--------|
| An | Canvas + failure modes | spec/spec-final.md phần 1, 4 |
| Bình | User stories 4 paths + prompt engineering | spec/spec-final.md phần 2, prototype/prompt-tests.md |
| Châu | Eval metrics + ROI + demo slides | spec/spec-final.md phần 3, 5, demo/slides.pdf |
| Dũng | UI prototype + demo script | prototype/, demo/demo-script.md |
```

### reflection.md

```markdown
# Individual reflection — Nguyễn Văn A (AI20K001)

## 1. Role
UX designer + prompt engineer. Phụ trách thiết kế flow chatbot và viết system prompt.

## 2. Đóng góp cụ thể
- Thiết kế conversation flow 5 bước (hỏi vị trí đau → thời gian → mức độ → tiền sử → gợi ý)
- Viết và test 3 phiên bản system prompt, chọn v3 vì recall tốt nhất trên 10 test cases
- Vẽ poster layout cho demo

## 3. SPEC mạnh/yếu
- Mạnh nhất: failure modes — nhóm nghĩ ra được case "triệu chứng chung chung"
  mà AI gợi ý quá rộng, và có mitigation cụ thể (hỏi thêm câu follow-up)
- Yếu nhất: ROI — 3 kịch bản thực ra chỉ khác số user, assumption gần giống nhau.
  Nên tách assumption rõ hơn (VD: conservative = chỉ dùng ở 1 chi nhánh,
  optimistic = rollout toàn hệ thống)

## 4. Đóng góp khác
- Test prompt với 10 triệu chứng khác nhau, ghi log kết quả vào prompt-tests.md
- Giúp Châu debug eval metrics — ban đầu chỉ có "accuracy" chung,
  sau tách ra precision cho từng khoa

## 5. Điều học được
Trước hackathon nghĩ precision và recall chỉ là metric kỹ thuật.
Sau khi thiết kế AI triage mới hiểu: chọn recall cao hơn cho khoa cấp cứu
(bỏ sót nguy hiểm hơn false alarm) nhưng precision cao hơn cho khoa chuyên sâu
(gợi ý sai gây lãng phí thời gian bệnh nhân). Metric là product decision,
không chỉ engineering decision.

## 6. Nếu làm lại
Sẽ test prompt sớm hơn — ngày đầu chỉ viết SPEC, đến trưa D6 mới bắt đầu test prompt.
Nếu test sớm từ tối D5 thì có thể iterate thêm 2-3 vòng, prompt sẽ tốt hơn nhiều.

## 7. AI giúp gì / AI sai gì
- **Giúp:** dùng Claude để brainstorm failure modes — nó gợi ý được "drug interaction"
  mà nhóm không nghĩ ra. Dùng Gemini để test prompt nhanh qua AI Studio.
- **Sai/mislead:** Claude gợi ý thêm feature "đặt lịch khám" vào chatbot —
  nghe hay nhưng scope quá lớn cho hackathon. Suýt bị scope creep nếu không dừng lại.
  Bài học: AI brainstorm tốt nhưng không biết giới hạn scope.
```

---

## Checklist trước khi nộp

- [ ] Repo cá nhân đặt tên đúng: `MaHocVien-HoTen-Day06`
- [ ] Repo nhóm đặt tên đúng: `NhomXX-Lop-Day06`
- [ ] Repo cá nhân có: `reflection.md`
- [ ] Repo nhóm có: `spec-final.md` + `prototype-readme.md` + `demo-slides.pdf`
- [ ] SPEC đủ 6 phần
- [ ] Prototype readme có link + mô tả + phân công
- [ ] Reflection đủ 7 phần, không copy-paste với bạn cùng nhóm
- [ ] Có ít nhất 1 commit trên group repo

---

## Tài liệu trong repo này

### Templates — [`02-templates/`](02-templates/)

| File | Dùng cho |
|------|----------|
| [`01-spec-template.md`](02-templates/01-spec-template.md) | SPEC 6 phần — template chính |
| [`02-canvas-template.md`](02-templates/02-canvas-template.md) | AI Product Canvas |
| [`03-user-stories-4path.md`](02-templates/03-user-stories-4path.md) | User stories × 4 paths |
| [`04-eval-metrics.md`](02-templates/04-eval-metrics.md) | Eval metrics + threshold |
| [`05-failure-modes.md`](02-templates/05-failure-modes.md) | Top 3 failure modes |
| [`06-roi-3-scenarios.md`](02-templates/06-roi-3-scenarios.md) | ROI 3 kịch bản |
| [`07-poster-layout.md`](02-templates/07-poster-layout.md) | Poster/slides layout |
| [`08-demo-script.md`](02-templates/08-demo-script.md) | Demo script 5 phút + 5 phút Q&A |

### Hướng dẫn công cụ — [`03-tools-guide/`](03-tools-guide/)

| File | Nội dung |
|------|----------|
| [`api-cheatsheet.md`](03-tools-guide/api-cheatsheet.md) | API key setup, model nào cho gì |
| [`prototyping-tools.md`](03-tools-guide/prototyping-tools.md) | Tools theo level prototype + hướng dẫn từng tool |
| [`prompt-engineering-tips.md`](03-tools-guide/prompt-engineering-tips.md) | Prompt tips cho hackathon |

### Tham khảo — [`04-reference/`](04-reference/)

| File | Nội dung |
|------|----------|
| [`day5-cheatsheet.md`](04-reference/day5-cheatsheet.md) | Recap 1 trang các framework Day 5 |
| [`canvas-example.md`](04-reference/canvas-example.md) | Ví dụ Canvas hoàn chỉnh (AI Email Triage) |

### Luật chơi

| File | Nội dung |
|------|----------|
| [`hackathon-rules.md`](hackathon-rules.md) | Rules, timeline, milestones, scoring |
| [`hackathon-rules-E403.md`](hackathon-rules-E403.md) | Rules lớp E403 — demo round flow, Google Form, scoring |

---

*Ngày 6 — VinUni A20 — AI Thực Chiến · 2026*
