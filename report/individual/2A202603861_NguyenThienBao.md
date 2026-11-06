# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603861**
- **Họ và tên: Nguyễn Thiên Bảo**
- **SEED** (in ở ô đầu notebook): 4309

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra: **15/15 đạt** (gồm 5/5 câu tính tay). Đã ghi đủ 10 dự đoán.

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.922 | 0.919 | 0.05 | 0 |
| 2. BoW + MLP | 0.950 | 0.947 | 6.37 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.906 | 0.976 | 83.97 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.961 | 0.974 | 37.37 | 0 |
| 3c. Encoder (MiniLM) + MLP head | **0.966** | 0.974 | 7.20 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 0.000 | 1219.70 | 0.008 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 0.000 | 1649.30 | 0.084 |
| 4c. LLM nhỏ few-shot | — | 0.000 | 1171.99 | 0.023 |

- Hàng LLM có F1 = 0 vì invalid = 40/40: model trả về chuỗi rỗng (chỉ tốn tiền token input, 0 token output) với `max_completion_tokens = 3`. Đây là lỗi ngân sách token, không phản ánh năng lực phân loại của LLM. (đề viết sẵn và không nói có thể sửa)
- Pareto: BoW + LogReg → BoW + MLP → 3c → 3a. Đạt SLA p95 < 50 ms: BoW + LogReg, BoW + MLP, 3b, 3c.

## 3. Dự đoán và thực tế

1. **`llm_models`** — Đoán: model lớn F1 cao hơn model nhỏ ~0.02–0.05, ngang 3b. Thực tế: **cả hai F1 = 0, invalid 40/40**. Lệch vì 3 token output không đủ để model in ra nhãn nên trả về rỗng; `parse_label` ra None → bị tính là ham. Phần chi phí thì đoán đúng: model lớn đắt gấp ~10 lần ($0.00335 so với $0.00033).
2. **`ml_lr`** — Đoán: lr = 50 thì loss dao động hoặc phân kỳ. Thực tế: loss vọt lên ~2.5 ở epoch đầu rồi giảm về **0.005**, thấp nhất trong 3 mức lr. Lệch vì BCE của logistic regression là hàm lồi, feature 0/1 nên mô hình vẫn hội tụ. Nhưng loss train thấp chỉ có nghĩa là trọng số rất lớn, mô hình quá tự tin — **loss train thấp ≠ mô hình tốt**.
3. **`tradeoff`** — Đoán: BoW + MLP và 3a kNN bị lấn át. Thực tế: **cả hai đều nằm trên đường Pareto** (3a có F1 SUB cao nhất), còn 3b mới là cách bị lấn át (bởi 3c). Lệch vì trên 40 tin, chênh lệch 0.002 F1 chưa tới 1 tin. Trên toàn bộ test, kNN lại kém nhất (0.906) → thứ hạng trên SUB không đáng tin.

## 4. Phân tích 50 case khó

| Model | F1 SUB → F1 50 case khó | Acc 50 case khó |
| --- | --- | --- |
| BoW + LogReg | 0.919 → **0.333** | 0.52 |
| BoW + MLP | 0.947 → 0.432 | 0.58 |
| Emb + kNN | 0.976 → 0.724 | 0.68 |
| Emb + LogReg | 0.974 → 0.640 | 0.64 |
| Emb + MLP head | 0.974 → 0.653 | 0.66 |

**BoW + LogReg tụt nhiều nhất** , kém nhất ở các nhóm *Spam giọng tự nhiên* (10%), *Thao túng / lừa đảo* (12%) và *Né từ khoá* (17%). LLM (4a, 4d) trả về rỗng nên luôn ra ham → 0% ở mọi nhóm spam, không đánh giá được.

Hai case cụ thể:
- **#29** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd...` (Né từ khoá): BoW tách thành các token như `c0ngr4tulat10ns`, `w0n`, `g1ft` — không có trong vocab → vector gần như rỗng → bias kéo về ham. Embedding vẫn hiểu được nghĩa nên các model Encoder đúng 83% ở nhóm này.
- **#12** `hi there, sorry to bother you, my friend gave me your number...` (Spam giọng tự nhiên): **tất cả model đều sai**. Câu lịch sự, đầy từ cá nhân (i, my, you) mà BoW gán trọng số ham, và về nghĩa cũng gần tin nhắn bình thường nên embedding đặt nó vào vùng ham. Muốn nhận ra cần hiểu **ý đồ** (dụ vào trang hẹn hò), không chỉ từ ngữ hay chủ đề.

Bài học: test thường cùng phân phối với train nên điểm quá lạc quan; mô hình học thuộc "khuôn" spam cũ chứ chưa hiểu khái niệm spam.

## 5. Thử thách cá nhân

**Đề #2:** giảm vocab còn 100 từ và tăng lên 5000 từ, train lại, báo F1 test.
**Cách làm:** với mỗi `max_size` ∈ {100, 2000, 5000}: `build_vocab` trên train → `vectorize` train/test → `train_logreg` (lr = 0.5, 300 epoch) → `metrics` trên test.

| Vocab | Recall | F1 | Train (s) |
| --- | --- | --- | --- |
| 100 | 0.760 | 0.863 | 0.1 |
| 2000 | 0.856 | 0.922 | 3.2 |
| 5000 | 0.856 | 0.922 | 6.9 |

**Giải thích:** với 100 từ, vocab gần như chỉ còn từ phổ biến và mất các từ đặc trưng của spam → bỏ sót nhiều spam (recall giảm), precision vẫn 1.0. Từ 2000 lên 5000 chỉ thêm từ hiếm (train có 7136 từ khác nhau), trọng số gần 0 → F1 không đổi mà thời gian train tăng gấp đôi. ⇒ Lợi ích giảm dần; 2000 là mức hợp lý.

## 6. Phương án cho sản phẩm thật

**Chọn 3c — Encoder (MiniLM) + MLP head.**
- Đạt SLA: p95 < 50 ms; xử lý xấp xỉ 1125 tin/s, gấp ~10 lần mức cần (116 tin/s).
- Chất lượng cao nhất: F1 test **0.966**, cũng tốt nhất trong nhóm khó ngoài kNN (F1 50 case khó 0.653).
- Loại bỏ: 3a kNN (p95 84 ms, vượt SLA, chậm dần khi bộ nhớ lớn); LLM (p95 1.2–1.6 s; chi phí model nhỏ $0.008/1k tin × 10M tin/ngày ≈ **$80/ngày ≈ $2.400/tháng**, model lớn ≈ $25.200/tháng).
- **Đánh đổi chấp nhận:** cần tài nguyên tính toán cho encoder (đắt hơn BoW), và vẫn yếu với spam giọng tự nhiên. Khắc phục bằng **router**: tin có p ∈ [0.3, 0.7] (vài % lưu lượng) gửi sang LLM với prompt phòng thủ để xử lý bất đồng bộ; thêm bước chuẩn hoá chống né từ khoá và rule chặn prompt injection.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | Nhận gì | Mình sửa / kiểm chứng gì |
| --- | --- | --- | --- |
|---|---|---|---|
| 4 — TODO 4.2–4.4 | Viết prompt phân loại, `parse_label`, `llm_classify_batch` | trả về 3 hàm | Đọc từng dòng; chạy 2 ô kiểm tra (API giả lập) ✅ rồi mới gọi API thật |
| 4 — Lỗi invalid | Vì sao LLM ra invalid 40/40, F1 = 0? | Chỉ ra model trả về rỗng do `max_tokens` quá nhỏ; gợi ý in `finish_reason` / `usage` | Đối chiếu chi phí (chỉ có token input) |
| 4–6 — Câu hỏi | Gợi ý trả lời Câu 4, 5, 6 | Bản nháp câu trả lời | Đối chiếu với bảng kết quả, biểu đồ Pareto, bảng % theo nhóm của mình; rút gọn lại |
| 5 — Phản ánh | Gợi ý thiết kế router, ước tính chi phí | Ý tưởng router 2 tầng | Tính lại chi phí theo cột USD/1k trong bảng của mình |

**Điều đã học:**
- Với LLM, chi tiết kỹ thuật (ngân sách token, tham số reasoning, cách parse, đếm invalid) quyết định hệ thống có chạy được hay không. Phải xem output thô chứ không chỉ nhìn F1.
- Điểm trên tập test cùng phân phối dễ quá lạc quan; cần bộ case khó để biết mô hình thật sự hiểu hay chỉ học thuộc từ khoá.
- Mô hình phức tạp hơn chưa chắc tốt hơn: chất lượng biểu diễn (embedding) quan trọng hơn độ sâu của đầu phân loại, và với sản phẩm thật phải cân cả độ trễ lẫn chi phí, không chỉ F1.

