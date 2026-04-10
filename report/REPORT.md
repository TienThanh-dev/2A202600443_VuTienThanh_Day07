# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Tiến Thành
**Nhóm:** C401-C1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity đo góc giữa hai vector (text embeddings). Giá trị cao (gần 1) nghĩa là hai vector có hướng gần giống nhau, tức nội dung hai câu mang ý nghĩa tương đồng cao, bất kể độ dài vector khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Hôm nay trời nắng đẹp, tôi muốn đi dạo công viên.
- Sentence B: Thời tiết hôm nay rất đẹp nên tôi dự định ra ngoài dạo chơi.
- Tại sao tương đồng: Hai câu có từ khóa và ngữ nghĩa gần như giống nhau (thời tiết đẹp + đi dạo), nên vector embedding chỉ cùng hướng (cosine ~ 0.95)

**Ví dụ LOW similarity:**
- Sentence A: Hôm nay trời nắng đẹp, tôi muốn đi dạo công viên.
- Sentence B: Công ty tôi vừa ra mắt mẫu điện thoại mới.
- Tại sao khác: Hai câu hoàn toàn khác chủ đề (thời tiết vs. công nghệ), vector embedding chỉ theo hai hướng gần như vuông góc (cosine ~ 0.05)

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc (hướng) giữa hai vector, nên loại bỏ ảnh hưởng của độ dài (magnitude) – điều thường khác nhau giữa các câu ngắn/dài hoặc do cách model nhúng. Euclidean distance lại bị ảnh hưởng mạnh bởi magnitude, dẫn đến hai câu tương đồng về nghĩa nhưng khác độ dài sẽ bị đánh giá “khác xa” một cách sai lệch.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*$N = \left\lceil \frac{L - C}{C - O} \right\rceil + 1$
> $$   L = 10000   $$ (độ dài document), $$   C = 500   $$ (chunk size), $$   O = 50   $$ (overlap).  
> Bước nhảy giữa các chunk: $$   C - O = 450   $$.  
> $$   
> \frac{10000 - 500}{450} = \frac{9500}{450} \approx 21.111 \quad \Rightarrow \quad \lceil 21.111 \rceil = 22
> 22+1 =23

> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap = 100 thì bước nhảy giảm còn 500 - 100 = 400, số chunk tăng lên 25 (tăng 2 chunk). Muốn overlap nhiều hơn vì giúp giữ lại nhiều ngữ cảnh hơn ở ranh giới giữa các chunk, giảm nguy cơ mất thông tin quan trọng khi chia tài liệu, từ đó cải thiện chất lượng retrieval và xử lý downstream (dù tốn thêm tài nguyên lưu trữ/tính toán).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hệ thống hỗ trợ học tập và tra cứu quy trình khóa học "AI Thực Chiến" (VinUni A20).

**Tại sao nhóm chọn domain này?**
Nhóm chọn bộ tài liệu bài tập AI này vì đây là nguồn dữ liệu thực tế, có cấu trúc rõ ràng (gồm mục tiêu, timeline, tiêu chí chấm điểm và hướng dẫn kỹ thuật). Việc xây dựng RAG trên bộ dữ liệu này giúp học viên nhanh chóng tra cứu các yêu cầu bài tập (deliverables), thời hạn (deadlines) và các bước cài đặt môi trường mà không cần đọc thủ công toàn bộ các file Markdown.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự (Thực tế) | Metadata đã gán |
|---|---|---|---|---|
| 1 | day02.md | Tài liệu Lab Ngày 2 | 6,666 | day: "02", topic: "problem_statement" |
| 2 | day03.md | Tài liệu Lab Ngày 3 | 2,352 | day: "03", topic: "agent_implementation" |
| 3 | day05.md | Tài liệu Lab Ngày 5 | 12,393 | day: "05", topic: "product_design" |
| 4 | day06.md | Tài liệu Lab Ngày 6 | 17,091 | day: "06", topic: "hackathon" |
| 5 | day07.md | Tài liệu Lab Ngày 7 | 7,609 | day: "07", topic: "embedding_rag" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|---|---|---|---|
| day | String | "02", "06", "07" | Giúp giới hạn phạm vi tìm kiếm khi người dùng hỏi về một ngày học cụ thể (ví dụ: "Deadline nộp bài Ngày 5 là khi nào?"). |
| topic | String | "hackathon", "logic" | Giúp phân loại nội dung giữa phần lý thuyết thiết kế và phần thực hành lập trình, giúp hàm `search_with_filter` trả về kết quả chính xác hơn theo ngữ cảnh. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy         | Chunk Count | Avg Length | Preserves Context? |
| -------- | ---------------- | ----------- | ---------- | ------------------ |
| day02.md | FixedSizeChunker | 37          | 198.0      |  Low              |
| day02.md | SentenceChunker  | 7           | 787.57     | High             |
| day02.md | RecursiveChunker | 45          | 121.09     | Medium          |
| day03.md | FixedSizeChunker | 16          | 192.87     |  Low              |
| day03.md | SentenceChunker  | 9           | 257.33     | High             |
| day03.md | RecursiveChunker | 17          | 135.82     | Medium          |
| day05.md | FixedSizeChunker | 71          | 198.46     |  Low              |
| day05.md | SentenceChunker  | 14          | 752.36     | High             |
| day05.md | RecursiveChunker | 79          | 132.41     | Medium          |
| day06.md | FixedSizeChunker | 100         | 199.04     |  Low              |
| day06.md | SentenceChunker  | 23          | 647.61     | High             |
| day06.md | RecursiveChunker | 115         | 128.45     | Medium          |
| day07.md | FixedSizeChunker | 44          | 199.5      |  Low              |
| day07.md | SentenceChunker  | 13          | 505.69     | High             |
| day07.md | RecursiveChunker | 48          | 136.38     | Medium          |


### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Chiến lược này hoạt động theo cơ chế đệ quy, sử dụng một danh sách các dấu phân cách (separators) ưu tiên từ lớn đến nhỏ như \n#, \n##, \n\n, và. Đầu tiên, nó cố gắng chia văn bản ở các cấp độ Header để giữ các khối thông tin lớn đi cùng nhau; nếu một khối vẫn vượt quá chunk_size, nó sẽ tiếp tục chia nhỏ bằng các dấu phân cách thấp hơn như xuống dòng hoặc dấu chấm. Điều này đảm bảo văn bản được chia nhỏ nhưng không làm vỡ các cấu trúc logic quan trọng

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Bộ dữ liệu của nhómcó cấu trúc Markdown rất rõ ràng với các tiêu đề (Headers), bảng và danh sách liệt kê nhiệm. RecursiveChunker khai thác các pattern này để đảm bảo một yêu cầu bài tập hoặc một tiêu chí chấm điểm không bị cắt đôi giữa hai chunk, giúp Agent có đủ ngữ cảnh (context) để trả lời chính xác các câu hỏi về quy trình khóa học

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy             | Chunk Count | Avg Length | Retrieval Quality? |
| -------- | -------------------- | ----------- | ---------- | ------------------ |
| day02.md | FixedSizeChunker     | 37          | 198.0      |  Low              |
| day02.md | **RecursiveChunker** | 46          | 119.08    |  Good             |
| day03.md | FixedSizeChunker     | 16          | 192.87     |  Low              |
| day03.md | **RecursiveChunker** | 19          | 121.95    |  Good             |
| day05.md | FixedSizeChunker     | 71          | 198.46     |  Low              |
| day05.md | **RecursiveChunker** | 90          | 116.45     |  Good             |
| day06.md | FixedSizeChunker     | 100         | 199.04     |  Low              |
| day06.md | **RecursiveChunker** | 119         | 124.42     |  Good             |
| day07.md | FixedSizeChunker     | 44          | 199.5      |  Low              |
| day07.md | **RecursiveChunker** | 55          | 119.49     |  Good             |


### So Sánh Với Thành Viên Khác

| Thành viên | Strategy         | Retrieval Score (/10) | Điểm mạnh                                             | Điểm yếu                             |
| ---------- | ---------------- | --------------------- | ----------------------------------------------------- | ------------------------------------ |
| Tôi        | RecursiveChunker | 8.5                   | Giữ context tốt, chunk size hợp lý, retrieval ổn định | Số chunk nhiều hơn → tốn compute     |
| A          | SentenceChunker  | 8.0                   | Context rất tốt (theo câu)                            | Chunk quá dài → noise, dễ vượt token |
| B          | FixedSizeChunker | 6.5                   | Đơn giản, nhanh                                       | Mất context, retrieval kém chính xác |


**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* RecursiveChunker là lựa chọn tốt nhất vì cân bằng giữa việc giữ ngữ cảnh và kiểm soát kích thước chunk. Trong khi SentenceChunker giữ context tốt nhưng chunk quá dài, và FixedSizeChunker làm mất cấu trúc semantic, thì RecursiveChunker tận dụng cấu trúc Markdown để tối ưu retrieval accuracy trong hệ thống RAG.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex pattern `r''(?<=[.!?])\s+'` kết hợp negative lookbehind để detect dấu câu kết thúc, tránh tách nhầm abbreviation.
> Hàm chính (`chunk`) sẽ split text thành list câu, sau đó merge các câu ngắn lại để đảm bảo chunk_size không bị vi phạm.  
> Edge case được xử lý: câu không dấu chấm (toàn bộ paragraph), multiple whitespace, dấu chấm trong quotes, và câu cuối cùng không có dấu chấm.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán recursive: bắt đầu với danh sách separators theo thứ tự ưu tiên (`["\n\n", "\n", ". ", " ", ""]`), thử split text theo separator hiện tại.  
> Nếu bất kỳ phần nào > `chunk_size` thì gọi đệ quy `_split` với separator tiếp theo (nhỏ hơn). Base case là khi separator rỗng hoặc toàn bộ phần ≤ `chunk_size`.  
> Mỗi chunk được trả về kèm overlap bằng cách giữ lại `overlap` ký tự của chunk trước (nếu có), đảm bảo ngữ cảnh liên tục giữa các chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` embed toàn bộ documents (sử dụng `embedding_fn`), sau đó chia nhỏ thành chunks bằng `RecursiveChunker` (hoặc SentenceChunker tùy config), lưu vào Chroma collection với metadata (`source`, `chunk_index`).  
> `search` embed query → gọi `collection.query` với `similarity_metric="cosine"` (hoặc dot product sau khi normalize), trả về top-k kết quả kèm score và metadata.  
> Toàn bộ quá trình được wrap trong `try-except` để fallback graceful khi embedding lỗi.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` áp dụng filter metadata (`where` clause của Chroma) ngay trong query vector DB (filter trước khi tính similarity) để giảm scope tìm kiếm.  
> `delete_document` xóa theo `document_id` bằng cách query tất cả ids có metadata khớp rồi gọi `collection.delete(ids=...)`, hỗ trợ cả delete single doc và delete theo filter.  
> Cả hai hàm đều log số lượng items bị ảnh hưởng và raise informative error nếu collection không tồn tại.

### KnowledgeBaseAgent

**`answer`** — approach:
> Hàm `answer` nhận query từ user, sau đó gọi `_store.search` để retrieve top-k chunks liên quan nhất.  
> Các chunk này được concatenate thành một context block và inject vào prompt theo format:
> "Context: ... \n\n Question: ... \n\n Answer:"  
> Prompt được thiết kế theo hướng grounded QA, yêu cầu model chỉ trả lời dựa trên context được cung cấp nhằm giảm hallucination.  
> Nếu không tìm thấy context phù hợp, agent sẽ fallback trả lời "Không tìm thấy thông tin trong tài liệu".

### Test Results

```
=================================================== test session starts ====================================================
platform win32 -- Python 3.11.0, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\vutie\.conda\envs\ai_in_action\python.exe
cachedir: .pytest_cache
rootdir: C:\D\AI_in_action\Day_7\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0, langsmith-0.7.26
collected 42 items                                                                                                          

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                 [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                          [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                   [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                    [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                         [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                         [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                               [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                              [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                           [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                       [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                 [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                        [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                            [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                      [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                            [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                  [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                    [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                          [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                               [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                 [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                     [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                  [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                           [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                          [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                     [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                 [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                            [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                      [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED             [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                           [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                          [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED              [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                         [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                  [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED        [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED            [100%]

==================================================== 42 passed in 1.97s ====================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Mục tiêu của Lab Ngày 7 là tìm hiểu về Vector Store" | "Học cách triển khai RAG pattern trong Lab 7" | Cao | -0.0398 | Sai |
| 2 | "Cách cài đặt Local Embedder" | "Sử dụng sentence-transformers để chạy embedding" | Cao | -0.0909 | Sai |
| 3 | "RecursiveChunker chia nhỏ văn bản đệ quy" | "Chiến lược chunking dựa trên câu" | Trung bình | -0.0355 | Đúng |
| 4 | "Hệ thống RAG giúp chatbot tra cứu tài liệu" | "Thời tiết hôm nay có nắng nhẹ" | Thấp | -0.0926 | Đúng |
| 5 | "Nộp bài vào thư mục report" | "Hoàn thành các TODO trong src package" | Trung bình | 0.1969 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các cặp câu mang tính kỹ thuật rất gần nhau (Pair 1, 2) nhưng score lại rất thấp hoặc âm. Điều này khẳng định `MockEmbedder` (sử dụng random/hash) không thể xử lý ngữ nghĩa thực sự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Cách tính điểm cho bài tập UX Ngày 5 là gì? | Dựa trên tiêu chí trải nghiệm người dùng và tính khả thi (chi tiết trong day05.md). |
| 2 | Các giai đoạn chính của Lab Ngày 7 gồm những gì? | Gồm 2 Phase: Cá nhân (implement src) và Nhóm (benchmark strategy). |
| 3 | Deadline nộp SPEC draft là lúc mấy giờ? | Thường được quy định vào cuối ngày hoặc theo timeline trong day05.md. |
| 4 | Sự khác biệt giữa Mock prototype và Working prototype là gì? | Mock là bản mô phỏng giao diện, Working là bản có chức năng thực tế (day06.md). |
| 5 | Cấu trúc thư mục của Phase 3 yêu cầu gì? | Yêu cầu các folder src, tests và notebook rõ ràng (day03.md). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Cách tính điểm cho bài tập UX Ngày 5 là gì? | Bài tập UX chiếm 10 điểm, chấm dựa trên: phân tích 4 paths (4đ), sketch as-is + to-be (4đ) và nhận xét gap marketing/thực tế (2đ)| 0.89 | Yes| Điểm được tính dựa trên khả năng phân tích đủ 4 luồng trải nghiệm, chất lượng bản vẽ sketch và nhận xét về sự khác biệt giữa quảng cáo và sử dụng thực tế|
| 2 | Các giai đoạn chính của Lab Ngày 7 gồm những gì? | Lab chia làm 2 Phase: Phase 1 (Cá nhân) hoàn thành src package và Phase 2 (Nhóm) so sánh các chiến lược Retrieval |0.94 | Yes| Lab 7 gồm hai giai đoạn chính: đầu tiên học viên tự lập trình các hàm TODO trong gói src, sau đó nhóm phối hợp thực hiện benchmark để so sánh hiệu quả các chiến lược chunking|
| 3 | Deadline nộp SPEC draft là lúc mấy giờ?| Thời hạn cuối cùng để nộp Topic + SPEC draft là 23:59 ngày 08/04 (Day 5) | 0.91 | Yes | Hạn chót để các nhóm hoàn thành và nộp bản thảo SPEC lên hệ thống là 23:59 tối ngày 08/04/2026 |
| 4 |  Sự khác biệt giữa Mock prototype và Working prototype là gì?| Mock là UI/flow chưa gắn AI thật (dùng tool như Figma); Working là bản có AI chạy thật, có input/output thực tế | 0.92 | Yes | Mock prototype chỉ mô phỏng giao diện và luồng click, trong khi Working prototype bắt buộc phải thực hiện các lệnh gọi API AI để xử lý dữ liệu thật |
| 5 | Cấu trúc thư mục của Phase 3 yêu cầu gì? | Yêu cầu thiết lập môi trường với cấu trúc gồm thư mục src/tools/ để mở rộng công cụ và thư mục models/ để chứa file .gguf | 0.85 | Yes| Theo hướng dẫn Lab 3, cấu trúc thư mục cần đảm bảo có nơi lưu trữ các công cụ tùy chỉnh (tools) và các mô hình ngôn ngữ chạy local |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi đã học được cách tối ưu hóa việc sử dụng metadata filtering để thu hẹp phạm vi tìm kiếm. Thay vì chỉ dựa vào độ tương đồng văn bản, việc một thành viên trong nhóm áp dụng thêm trường importance đã giúp hệ thống ưu tiên các tài liệu chính thống (như quy chế chính thức) so với các bản thảo, giúp giảm thiểu nhiễu và tăng độ chính xác của câu trả lời từ Agent.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Họ đã visualize lại kết quả của các chiến lược mà họ thực hiện từ đó có thể so sánh dễ dàng hơn sự tương quan tốt xấu giữa các chiến lược được sử dụng đối với data của họ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Không sử dụng một chiến lược có sẵn hay chỉ 1 chiến lược mà sử dụng hybrid chunking, kết hợp nhiều chiến lược lại với nhau để chunking tốt hơn nữa. Phân tích sâu hơn về cấu trúc, nội dung data để thêm nhiều trường metadata hơn một cách rõ ràng hơn, filter dễ dàng. 


---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | 84/ 90** |
