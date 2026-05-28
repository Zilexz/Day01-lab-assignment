# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp (0.0–0.5), các phản hồi gần như giống hệt nhau qua nhiều lần gọi, dùng từ ngữ an toàn, ổn định và thường lặp lại cùng một sự thật quen thuộc (ví dụ Việt Nam có hơn 3.000 km bờ biển). Khi temperature tăng lên 1.0 rồi 1.5, mô hình bắt đầu chọn những sự thật đa dạng hơn, cách diễn đạt sáng tạo và bất ngờ hơn; nhưng ở 1.5 thì câu chữ đôi khi trở nên rời rạc, lan man hoặc kém chính xác về mặt dữ kiện. Nói cách khác, temperature càng cao thì độ ngẫu nhiên và sáng tạo càng tăng, đổi lại tính nhất quán và độ tin cậy giảm đi.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature thấp, khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần câu trả lời nhất quán, chính xác và bám sát chính sách của doanh nghiệp; độ ngẫu nhiên thấp giúp giảm nguy cơ "bịa" thông tin (hallucination) và đảm bảo cùng một câu hỏi sẽ nhận được câu trả lời đáng tin cậy như nhau, điều quan trọng hơn nhiều so với sự đa dạng hay sáng tạo trong ngữ cảnh này.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Tổng token mỗi ngày = 10.000 người × 3 lần gọi × 350 token = 10.500.000 token ≈ 10.500 nghìn-token (K-token).
> - GPT-4o: 10.500 K-token × $0,010 = **$105 / ngày** (≈ $3.150 / tháng).
> - GPT-4o-mini: 10.500 K-token × $0,0006 = **$6,30 / ngày** (≈ $189 / tháng).
>
> Tỷ lệ chi phí = $0,010 / $0,0006 ≈ **16,7 lần**. Nghĩa là với cùng khối lượng công việc này, GPT-4o đắt hơn GPT-4o-mini khoảng 16–17 lần. (Đây là ước tính theo giá token đầu ra để minh hoạ; chi phí thực tế còn cộng thêm token đầu vào nhưng tỷ lệ chênh lệch giữa hai model vẫn ở mức tương tự.)

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> **GPT-4o xứng đáng:** các tác vụ đòi hỏi suy luận phức tạp, nhiều bước và độ chính xác cao như phân tích hợp đồng pháp lý, tư vấn tài chính/y tế, sinh code đáng tin cậy hoặc tóm tắt tài liệu chuyên ngành — nơi một câu trả lời sai có thể gây thiệt hại lớn hơn nhiều so với phần chi phí token tăng thêm.
>
> **GPT-4o-mini tốt hơn:** các tác vụ khối lượng lớn, đơn giản và lặp lại như phân loại văn bản, phát hiện ý định (intent), định tuyến yêu cầu, trả lời FAQ ngắn hoặc gắn nhãn dữ liệu — nơi chất lượng giữa hai model gần như không khác biệt nhưng việc tiết kiệm ~16 lần chi phí và độ trễ thấp hơn lại mang lại lợi ích rõ rệt khi scale.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với các giao diện tương tác hướng tới người dùng và những phản hồi dài, ví dụ chatbot hội thoại hay khi mô hình viết một đoạn văn/đoạn code dài: việc hiển thị token ngay khi chúng được sinh ra giúp giảm mạnh "thời gian chờ đến token đầu tiên" (time-to-first-token), khiến hệ thống cảm giác phản hồi nhanh và giữ người dùng tập trung thay vì nhìn màn hình trống. Ngược lại, non-streaming phù hợp hơn khi đầu ra được tiêu thụ bởi máy chứ không phải con người, hoặc khi cần toàn bộ kết quả trước khi xử lý — chẳng hạn phải parse một JSON hoàn chỉnh, chạy tác vụ nền/batch, kiểm tra và biến đổi toàn bộ output trước khi dùng, hoặc với các phản hồi rất ngắn mà streaming chỉ làm tăng độ phức tạp mà không cải thiện trải nghiệm.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định
