# Ngày 2 — Tìm Đúng Bài Toán — Updated for v2 Metrics

> Từ brief mơ hồ hoặc cơ hội thực tế → Problem Statement rõ ràng, testable, có boundary.

---

## Tổng quan buổi học

```text
SÁNG — Lecture (4h)                     CHIỀU — Lab (4h)
┌──────────────────────────┐            ┌──────────────────────────┐
│ Business-to-AI           │            │ Scan bài toán từ chính   │
│ Translation Framework    │     →      │ mình → Filter → Deep-dive│
│ AI-Fit Matrix            │            │ → AI Fit → Go/No-Go      │
│ Go / No-Go / Not Yet     │            │ → Pitch                  │
└──────────────────────────┘            └──────────────────────────┘
```

**Lecture** dạy frameworks: cách dịch yêu cầu mơ hồ thành bài toán AI, chọn mức giải pháp (Rule / LLM Feature / Agent), viết Problem Statement, và ra quyết định Go / No-Go / Not Yet.

**Lab** cho học viên thực hành toàn bộ quy trình trên bài toán thật từ công việc/cuộc sống của mình — từ scan 5+ ý tưởng → chọn 1 → deep-dive → quyết định.

---

## Nguyên tắc chung (áp dụng xuyên suốt Ngày 2, 5, 6)

1. **AI không thay học viên ra quyết định** — AI brainstorm/draft, học viên chọn/sửa/chốt.
2. **"Không build AI" vẫn có thể điểm cao** — điểm dựa trên chất lượng tư duy, không phải độ “ngầu” của giải pháp.
3. **Bắt buộc có AI Support Log** — ghi AI giúp gì, sai gì, nhóm sửa gì.
4. **Gate giữa các ngày** — Ngày 2: PS + metric + architecture đủ rõ → qua Ngày 5.
5. **Rubric mới tách rõ nhóm / cá nhân** — nhóm mạnh không tự động kéo toàn bộ thành viên lên điểm cao.

---

## Tài liệu

| File | Mô tả |
|------|-------|
| `01-worksheet.md` | Worksheet chính — self-check theo rubric |
| `02-deliverable-example.md` | Bài nộp mẫu hoàn chỉnh + giải thích vì sao bài này đạt điểm cao |
| `03-inspiration-kit.md` | Gợi ý bài toán cho ai chưa nghĩ ra |

---

## Flow buổi lab (4h)

```text
Phase 0  WORKED EXAMPLE         15 min   GV demo bài nộp mẫu — học viên xem
Phase 1  SCAN                   20 min   Cá nhân: liệt kê 5+ bài toán → nhóm: share top 3
Phase 2  QUICK-ASSESS           40 min   Cá nhân: điền 3 Quick Problem Cards → nhóm: gallery vote
         ─── Break ───          10 min
Phase 3  PITCH-CHALLENGE-VOTE   30 min   Nhóm pitch + challenge + chọn 1 bài toán
Phase 4  DEEP-DIVE              85 min   Nhóm: vẽ workflow + viết PS + research + future flow
         ─── Break ───          10 min
Phase 5  EVALUATE               20 min   Nhóm: AI Readiness + Go/Not Yet/No-Go
Phase 6  REFLECTION             10 min   Cá nhân: AI Support Log + reflection
```

---

## Deliverables học viên

Mỗi người nộp 1 zip: `MaHocVien-HoTen-day02.zip`

| File | Loại | Mô tả |
|------|------|-------|
| `01-problem-scan.md` | Cá nhân | 5+ bài toán qua 4 Lenses + 3 Quick Cards + kill rationale |
| `02-deep-dive-report.md` | Nhóm | Workflow + PS 6-field + metrics + research + AI fit + Go/No-Go |
| `03-ai-log.md` | Cá nhân | Log cho Phase 6 — Reflection: ghi lại AI hỗ trợ gì trong từng phase, sai/hời hợt ở đâu, học viên sửa gì bằng tay — và điều gì thay đổi trong tư duy sau buổi lab |
| `04-workflow-diagram.png/pdf` | Nhóm | Current-state + future-state flow |
| `extras/` | Tùy chọn | Research notes, screenshots, tài liệu bổ sung |

**Lưu ý vận hành**
- `01` và `03` là căn cứ chính cho **điểm cá nhân**.
- `02` và `04` là căn cứ chính cho **điểm nhóm**.
- Mỗi người vẫn nộp **1 zip riêng**, kể cả khi file nhóm giống nhau.

---

## Chấm điểm (100 điểm) — v2 metrics

### A. Điểm nhóm — 60 điểm

| Thành phần | Điểm | Focus |
|-----------|------|-------|
| Workflow Mapping | 15 | Flow vẽ rõ, bottleneck, thời gian, handoff |
| Problem Statement + Metrics + Boundary | 20 | 6 field đủ, metric có ngưỡng, boundary rõ |
| AI Fit + Alternatives + Future Flow | 15 | Justify so sánh rule / LLM / agent, future flow có AI / Human / Boundary / Fallback |
| Decision Quality | 10 | Go / Not Yet / No-Go justify bằng evidence |

### B. Điểm cá nhân — 40 điểm

| Thành phần | Điểm | Focus |
|-----------|------|-------|
| Scan Breadth + Quick Problem Cards | 12 | Breadth, dùng 4 lenses, cards đủ chất lượng |
| Pitch + Challenge Participation | 12 | Pitch rõ, challenge đúng trọng tâm, không đi nhờ nhóm |
| AI Support Log + Reflection | 10 | Log Phase 6: ghi trung thực AI hỗ trợ gì, sai ở đâu, học viên sửa gì — và rút ra được bài học gì cho bản thân |
| Individual Understanding Check | 6 | Trả lời được problem → metric → boundary → AI fit |

### C. Mức xếp loại

| Mức | Điểm | Ý nghĩa |
|-----|------|---------|
| **Không pass** | < 50 | Chưa nắm được problem-first, metric, AI fit |
| **Vừa đủ pass** | 50–64 | Có hiểu flow cơ bản nhưng còn mơ hồ ở phần chính |
| **Hiểu khá** | 65–79 | Làm được đa số yêu cầu, logic tương đối rõ |
| **Hiểu đầy đủ** | 80–89 | Workflow, PS, AI fit, decision đều nhất quán |
| **Hiểu xuất sắc** | 90–100 | Tư duy rất chặt, biết phản biện alternatives, quyết định trưởng thành |

### Nguyên tắc chấm

- Chất lượng tư duy > mức phức tạp giải pháp
- "Không cần AI" vẫn có thể điểm cao
- "Not Yet" với justify tốt > "Go" với lập luận yếu
- Điểm cá nhân dùng để **phân biệt mức độ hiểu thực**, không chỉ mức độ ngồi trong nhóm tốt

---

## Mục tiêu thực sự của Ngày 2

Ngày 2 không nhằm kiểm tra “ai nghĩ ra ý tưởng AI hay nhất”, mà nhằm kiểm tra xem học viên có thể:

- tìm pain thật,
- mô tả workflow thật,
- viết metric đo được,
- chọn đúng mức giải pháp,
- và ra quyết định trưởng thành hay không.

**Problem first, not AI first.**

---

*README — Ngày 2: Tìm Đúng Bài Toán*  
*Bản cập nhật theo v2 metrics*  
*VinUni A20 — AI Thực Chiến*
