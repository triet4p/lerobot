## Plan: Đồng Bộ JPEG + Cadence RTC (Giữ Nguyên Kích Thước Ảnh)

Mục tiêu là giảm khựng bằng 2 trục: monotonic observation timestep độc lập và nén JPEG có bật/tắt rõ ràng, đồng bộ cấu hình client -> server.

Ràng buộc bắt buộc:
- Chỉ nén khi truyền.
- Không thay đổi kích thước ảnh trong pipeline nén/giải nén.
- Ảnh sau giải nén ở server phải đúng kích thước gốc trước khi đưa vào model.
- Kích thước sau giải nén phải khớp đúng kích thước cấu hình camera của phiên chạy hiện tại.

Lưu ý thiết kế cũ: observation timestep từng được bám theo action index để giảm tần suất infer dư thừa và đồng bộ gần với nhịp action rollout. Cách này giúp tránh flood ở điều kiện mạng tốt, nhưng trong bối cảnh có interpolation/VPN và polling GetActions theo cụm thì dễ gây trùng timestep hoặc thưa observation enqueue, dẫn tới khựng. Vì vậy chuyển sang timestep observation monotonic độc lập là thay đổi bắt buộc để ổn định cadence.

**Steps**
1. Chốt baseline trước sửa trên cùng cấu hình chạy hiện tại: chạy 2-3 phút và lưu 2 CSV audit server làm mốc so sánh. (blocking cho các bước đánh giá sau)
2. Xác nhận server đang chạy đúng phiên bản code mới nhất (tránh benchmark sai do binary/commit cũ). (blocking cho 3-11)
3. Sửa monotonic observation timestep ở client RTC: tạo bộ đếm observation riêng, tăng dần mỗi lần gửi; không phụ thuộc action index. (blocking cho 4-11)
4. Thêm tham số bật/tắt rõ ràng cho timestamp mode độc lập ở config/CLI: obs_timestep_independent (bool, default True). (depends on 3)
5. Thiết kế bộ tham số JPEG ở client config/CLI: image_compress_enable (bool), image_compress_quality (int). Không có resize parameter để tránh đổi kích thước. (blocking cho 6-9)
6. Triển khai nén JPEG ở client trước khi pickle hóa observation khi image_compress_enable=True; khi False giữ nguyên payload cũ. (depends on 5)
7. Trong payload nén, kèm metadata kích thước gốc theo từng camera (orig_h, orig_w) để server kiểm chứng kích thước sau giải nén. (depends on 6)
8. Mở rộng RemotePolicyConfig để truyền đầy đủ chế độ nén từ client sang server (ít nhất enable + quality), đồng bộ theo từng session. (depends on 5, parallel with 6-7)
9. Server bắt buộc đọc mode nén từ policy setup và decode theo mode nhận được; ảnh sau decode phải được kiểm tra đúng kích thước gốc từng camera trước khi vào processor/model. Sai kích thước thì log lỗi rõ và bỏ frame lỗi. (depends on 6-8)
10. Ghi log mode rõ ràng để audit ở cả 2 phía:
- Client log đầu phiên: compress_enable/quality + payload bytes + shape từng camera trước nén
- Server log đầu phiên: compress_mode nhận từ policy setup + decode path active
- CSV audit thêm cột mode/compress_ratio/decode_ms + decoded_shape_cam1/decoded_shape_cam2. (depends on 6-9)
11. Chạy benchmark A/B theo 3 profile: no-compress, JPEG q90, JPEG q85; giữ cùng task/camera/fps/chunk settings. (depends on 3-10)
12. Chốt preset vận hành ổn định nhất (chunk_size_threshold, fps, interpolation_multiplier, jpeg quality). (depends on 11)
13. Tùy chọn sau khi RTC ổn định: mirror cơ chế tương tự sang async_inference. (parallel independent, không chặn rollout RTC)

**Relevant files**
- src/lerobot/rtc_inference/configs.py: thêm tham số bật/tắt timestep độc lập và JPEG encode.
- src/lerobot/rtc_inference/helpers.py: helper nén/giải nén ảnh observation + metadata kích thước gốc.
- src/lerobot/rtc_inference/robot_client.py: gán obs timestep monotonic, encode JPEG, log mode client, ghi shape trước nén.
- src/lerobot/rtc_inference/policy_server.py: nhận mode nén từ policy setup, decode theo mode, verify shape sau decode, log mode server + audit decode metrics.
- src/lerobot/rtc_inference/helpers.py (RemotePolicyConfig): trường config đồng bộ mode nén client -> server.
- scripts/orchestrator/orchestrator_rtc_xvla_client_only.py: expose đầy đủ cờ JPEG + timestep mode ra CLI.
- scripts/utils/orchestrator-xvla-client-only.sh: command mẫu có cờ bật/tắt JPEG rõ ràng.
- src/lerobot/async_inference/robot_client.py: mirror nếu mở rộng async.
- src/lerobot/async_inference/policy_server.py: mirror nếu mở rộng async.

**Verification**
1. Kiểm tra log đầu phiên ở client và server có cùng mode nén (enable/quality) cho đúng một session.
2. Xác nhận shape tại 3 điểm luôn đồng nhất theo camera:
- Lúc capture ở client
- Sau decode ở server
- Trước khi đưa vào model
3. Với mỗi camera, kích thước ở 3 điểm (capture client, decode server, trước model) phải trùng với kích thước đã cấu hình cho camera đó trong command/robot config của phiên chạy.
4. So sánh trước/sau theo CSV:
- inter_arrival_ms: p50/p95/max
- recv_to_dequeue_ms: p50/p95
- enqueue_to_dequeue_ms: p50/p95
- enqueued_ratio
- decode_ms (khi bật JPEG)
- compression_ratio hoặc payload_bytes
- decoded_shape_cam1/decoded_shape_cam2
5. Tiêu chí đạt:
- enqueued_ratio tăng rõ rệt
- inter_arrival_ms giảm và ít spike >1s
- enqueue_to_dequeue_ms vẫn thấp (không tạo nghẽn queue)
- decode_ms không trở thành bottleneck mới
- không có frame nào sai shape ở server audit
6. Chạy thực robot 3-5 phút để xác nhận giảm khựng theo cảm nhận và queue plot.

**Decisions**
- In scope: RTC path end-to-end trước, bao gồm đồng bộ config nén client -> server và audit mode/shape.
- Out of scope tạm thời: thay đổi thuật toán policy hoặc model horizon.
- Giữ tương thích ngược: tắt JPEG thì payload cũ hoạt động như hiện tại.

**Further Considerations**
1. Bắt đầu bằng JPEG quality 90, sau đó hạ 85 nếu vẫn nghẽn mạng.
2. Nếu decode_ms tăng cao, cân nhắc decode song song phía server sau khi có số liệu thực.
3. Chỉ mở rộng sang async_inference sau khi preset RTC ổn định.
