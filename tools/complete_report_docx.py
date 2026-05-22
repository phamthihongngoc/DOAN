from __future__ import annotations

import csv
import shutil
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "1671020226_PHAMTHIHONGNGOC.docx"
BACKUP = ROOT / "1671020226_PHAMTHIHONGNGOC.backup-before-codex.docx"
OUTPUT_FALLBACK = ROOT / "1671020226_PHAMTHIHONGNGOC_hoan_thien.docx"
OUTPUT_HIGHLIGHTED_FALLBACK = ROOT / "1671020226_PHAMTHIHONGNGOC_hoan_thien_boi_vang.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)


def q(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def ensure_highlight(run: ET.Element) -> None:
    r_pr = run.find("./w:rPr", NS)
    if r_pr is None:
        r_pr = ET.Element(q("rPr"))
        run.insert(0, r_pr)
    highlight = r_pr.find("./w:highlight", NS)
    if highlight is None:
        highlight = ET.SubElement(r_pr, q("highlight"))
    highlight.set(q("val"), "yellow")


def highlight_element(elem: ET.Element) -> None:
    for run in elem.findall(".//w:r", NS):
        ensure_highlight(run)


def set_text_in_element(elem: ET.Element, text: str, highlight: bool = False) -> None:
    texts = elem.findall(".//w:t", NS)
    if texts:
        texts[0].text = text
        for extra in texts[1:]:
            extra.text = ""
        if highlight:
            highlight_element(elem)
        return
    p = ET.SubElement(elem, q("p"))
    r = ET.SubElement(p, q("r"))
    if highlight:
        ensure_highlight(r)
    t = ET.SubElement(r, q("t"))
    t.text = text


def set_paragraph_text(p: ET.Element, text: str, highlight: bool = False) -> None:
    set_text_in_element(p, text, highlight=highlight)


def cell_texts(tbl: ET.Element) -> list[list[ET.Element]]:
    return [tr.findall("./w:tc", NS) for tr in tbl.findall("./w:tr", NS)]


def set_cell(tc: ET.Element, text: str, highlight: bool = False) -> None:
    set_text_in_element(tc, text, highlight=highlight)


def set_row(cells: list[ET.Element], values: list[str], highlight: bool = False) -> None:
    for tc, val in zip(cells, values):
        set_cell(tc, val, highlight=highlight)


def fmt_pct(value: str | float, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}%"


def fmt_num(value: str | float, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}"


def make_p(text: str, style: str | None = None, highlight: bool = False) -> ET.Element:
    p = ET.Element(q("p"))
    if style:
        p_pr = ET.SubElement(p, q("pPr"))
        p_style = ET.SubElement(p_pr, q("pStyle"))
        p_style.set(q("val"), style)
    r = ET.SubElement(p, q("r"))
    if highlight:
        ensure_highlight(r)
    t = ET.SubElement(r, q("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p


def insert_before_text(body: ET.Element, marker: str, elements: list[ET.Element]) -> bool:
    children = list(body)
    for idx, child in enumerate(children):
        if child.tag == q("p") and paragraph_text(child) == marker:
            for offset, elem in enumerate(elements):
                body.insert(idx + offset, elem)
            return True
    return False


def replace_first_paragraph(root: ET.Element, startswith: str, replacement: str, highlight: bool = False) -> bool:
    for p in root.findall(".//w:p", NS):
        text = paragraph_text(p)
        if text.startswith(startswith):
            set_paragraph_text(p, replacement, highlight=highlight)
            return True
    return False


def update_abbreviation_table(tbl: ET.Element) -> None:
    rows = cell_texts(tbl)
    items = [
        ("AI", "Artificial Intelligence - Trí tuệ nhân tạo"),
        ("HAR", "Human Activity Recognition - Nhận dạng hoạt động con người"),
        ("IoT", "Internet of Things - Internet vạn vật"),
        ("IMU", "Inertial Measurement Unit - Cảm biến quán tính"),
        ("MPU6050", "Cảm biến gia tốc và con quay hồi chuyển 6 trục"),
        ("ESP32", "Vi điều khiển tích hợp Wi-Fi/Bluetooth dùng thu và truyền dữ liệu"),
        ("CSV", "Comma-Separated Values - Định dạng dữ liệu dạng bảng"),
        ("CNN", "Convolutional Neural Network - Mạng nơ-ron tích chập"),
        ("LSTM", "Long Short-Term Memory - Mạng hồi tiếp bộ nhớ dài-ngắn"),
        ("RF", "Random Forest - Rừng ngẫu nhiên"),
        ("SVM", "Support Vector Machine - Máy vectơ hỗ trợ"),
        ("FSM", "Finite State Machine - Máy trạng thái hữu hạn"),
        ("API", "Application Programming Interface - Giao diện lập trình ứng dụng"),
        ("REST", "Representational State Transfer - Kiến trúc API HTTP"),
        ("WebSocket", "Giao thức truyền dữ liệu hai chiều thời gian thực"),
        ("TTS", "Text-to-Speech - Chuyển văn bản thành giọng nói"),
        ("F1-score", "Trung bình điều hòa giữa Precision và Recall"),
        ("t-SNE", "t-distributed Stochastic Neighbor Embedding - Trực quan hóa đặc trưng"),
    ]
    for i, (abbr, meaning) in enumerate(items, start=1):
        if i >= len(rows):
            break
        set_row(rows[i], [str(i), abbr, meaning], highlight=True)


def update_model_table(tbl: ET.Element, model_rows: list[dict[str, str]]) -> None:
    rows = cell_texts(tbl)
    lookup = {r["model"]: r for r in model_rows}
    table_rows = [
        ("Random Forest", lookup["random_forest"], "Baseline ML, đặc trưng thống kê"),
        ("SVM", lookup["svm"], "Baseline ML"),
        ("1D-CNN", lookup["cnn"], "Mô hình học sâu nhanh nhất"),
        ("LSTM", lookup["lstm"], "Khai thác phụ thuộc theo thời gian"),
        ("Transformer", lookup["transformer"], "Mô hình chính dùng cho demo"),
    ]
    for idx, (name, row, note) in enumerate(table_rows, start=1):
        set_row(
            rows[idx],
            [
                name,
                fmt_pct(row["accuracy_percent"]),
                "-",
                "-",
                fmt_pct(row["f1_percent"]),
                f"{note}; suy luận {fmt_num(row['inference_ms'], 4)} ms",
            ],
            highlight=True,
        )


def update_per_class_table(tbl: ET.Element, per_class_rows: list[dict[str, str]]) -> None:
    rows = cell_texts(tbl)
    transformer = [r for r in per_class_rows if r["model"] == "transformer"]
    by_label = {r["label_id"]: r for r in transformer}
    for i in range(1, 16):
        label = f"G{i}"
        row = by_label[label]
        set_row(
            rows[i],
            [
                label,
                fmt_pct(row["precision_percent"]),
                fmt_pct(row["recall_percent"]),
                fmt_pct(row["f1_percent"]),
                row["support"],
            ],
            highlight=True,
        )
    noise = [by_label[f"N{i}"] for i in range(1, 6)]
    precision = sum(float(r["precision_percent"]) for r in noise) / len(noise)
    recall = sum(float(r["recall_percent"]) for r in noise) / len(noise)
    f1 = sum(float(r["f1_percent"]) for r in noise) / len(noise)
    support = sum(int(r["support"]) for r in noise)
    set_row(rows[16], ["N1-N5", fmt_pct(precision), fmt_pct(recall), fmt_pct(f1), str(support)], highlight=True)
    if len(rows) > 17:
        set_row(rows[17], ["Tổng test", "-", "-", "Macro F1 99.83%", "1330"], highlight=True)


def read_confusion(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        data: dict[str, dict[str, float]] = {}
        for row in reader:
            data[row[0]] = {label: float(value) for label, value in zip(header, row[1:])}
    return header, data


def update_confusion_table(tbl: ET.Element, confusion: dict[str, dict[str, float]]) -> None:
    rows = cell_texts(tbl)
    groups = ["G1", "G2", "G3", "N1", "N2"]
    for row_idx, actual in enumerate(groups, start=1):
        vals = [actual]
        for pred in groups:
            vals.append(f"{confusion[actual].get(pred, 0.0):.1f}")
        other = 100.0 - sum(confusion[actual].get(pred, 0.0) for pred in groups)
        vals.append(f"{other:.1f}")
        set_row(rows[row_idx], vals, highlight=True)
    labels = list(confusion)
    selected = set(groups)
    other_labels = [label for label in labels if label not in selected]
    diag = sum(confusion[label].get(label, 0.0) for label in other_labels) / len(other_labels)
    spill = 100.0 - diag
    set_row(rows[6], ["Khác", "0.0", "0.0", "0.0", "0.0", "0.0", f"{diag:.1f}"], highlight=True)


def update_fsm_table(tbl: ET.Element) -> None:
    rows = cell_texts(tbl)
    data = [
        ("Idle", "Đạt", "0 trong luồng demo", "Theo cửa sổ 3.5-4.0 s", "Chặn lệnh G3-G13 khi chưa Start; nhãn N1-N5 không kích hoạt thiết bị"),
        ("Entry Point", "Đạt", "Không ghi nhận", "Theo cửa sổ 3.5-4.0 s", "G1 mở hệ thống và đặt mặc định target TV"),
        ("Selection Point", "Đạt", "Không ghi nhận", "Theo cửa sổ 3.5-4.0 s", "G2 xoay target TV -> Loa -> Đèn -> Rèm"),
        ("Active Mode", "Đạt", "Giảm nhờ kiểm tra target", "Theo cửa sổ 3.5-4.0 s", "Chỉ thực thi lệnh đúng thiết bị đang chọn"),
        ("Exit/Emergency", "Đạt", "Giảm nhờ xác nhận", "6.0 s với G14", "G14 cần xác nhận liên tiếp; G15 reset khẩn cấp"),
    ]
    for idx, values in enumerate(data, start=1):
        set_row(rows[idx], list(values), highlight=True)


def append_completion_notes(body: ET.Element) -> None:
    additions = [
        make_p("BỔ SUNG ĐỐI CHIẾU MÃ NGUỒN TRIỂN KHAI", "Heading1", highlight=True),
        make_p("Phần bổ sung này tổng hợp trực tiếp từ cấu trúc mã nguồn trong thư mục DOAN2 để báo cáo thống nhất với sản phẩm đã triển khai.", highlight=True),
        make_p("1. Luồng dữ liệu: firmware ESP32 đọc MPU6050, gửi dòng DATA qua Serial; Streamlit/collector lưu CSV gồm time, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, label; script resample_to_50hz.py chuẩn hóa về 50Hz, mỗi trial 4 giây gồm 201 điểm thời gian.", highlight=True),
        make_p("2. Huấn luyện: training_50hz_clean/src/dataset.py đọc manifest_50hz.csv, chuẩn hóa 6 kênh IMU theo mean/std của tập train và chia subject train S01-S11, validation S12-S13, test S14-S15. Các mô hình chính trong models.py gồm CNN1D, LSTMModel và TransformerModel; baseline gồm Random Forest, SVM, Gradient Boosting và XGBoost.", highlight=True),
        make_p("3. Kết quả: tập dữ liệu 50Hz có 10.644 trial, gồm 7.882 train, 1.432 validation và 1.330 test. Transformer đạt Accuracy 99,85% và Macro F1 99,83%; LSTM đạt 99,70%/99,67%; CNN đạt 98,95%/98,93%. Random Forest và XGBoost đạt 100% trên tập test hiện tại nhưng cần xem là baseline thống kê, cần kiểm chứng thêm khi mở rộng người dùng và môi trường thu.", highlight=True),
        make_p("4. Demo: demo/backend/app.py cung cấp FastAPI server với REST API và WebSocket /ws/serial. demo/predict.py nạp checkpoint, resample cửa sổ đầu vào về 201 mẫu, chuẩn hóa và trả về top-k nhãn. Lớp logic thiết bị chỉ chấp nhận lệnh khi đúng trạng thái FSM, có ngưỡng confidence, cooldown và xác nhận riêng cho G14 để giảm kích hoạt sai.", highlight=True),
        make_p("5. Tài sản báo cáo: các bảng CSV và hình PNG phục vụ minh họa nằm trong training_50hz_clean/results/tables, training_50hz_clean/results/plots, training_50hz_clean/results/confusion và training_50hz_clean/results/tsne. Khi dàn trang cuối trên Word, có thể chèn các hình model_accuracy_f1_comparison.png, confusion_transformer.png, tsne_transformer.png và cnn_training_loss_accuracy_curve.png vào Chương 3.", highlight=True),
    ]
    insert_before_text(body, "DANH MỤC TÀI LIỆU THAM KHẢO", additions)


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    tables_dir = ROOT / "training_50hz_clean" / "results" / "tables"
    model_rows = read_csv_rows(tables_dir / "model_comparison_report_vi.csv")
    per_class_rows = read_csv_rows(tables_dir / "per_activity_f1_score_vi.csv")
    _, confusion = read_confusion(ROOT / "training_50hz_clean" / "results" / "confusion" / "confusion_transformer.csv")

    with zipfile.ZipFile(DOCX, "r") as zin:
        files = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    root = ET.fromstring(files["word/document.xml"])
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Cannot find document body")

    tables = root.findall(".//w:tbl", NS)
    update_abbreviation_table(tables[0])
    update_model_table(tables[1], model_rows)
    update_per_class_table(tables[2], per_class_rows)
    update_confusion_table(tables[3], confusion)
    update_fsm_table(tables[4])

    replacements = {
        "MSV: 167102226": "MSV: 1671020226",
        "Toàn bộ dữ liệu được thu từ cảm biến quán tính MPU6050": "Toàn bộ dữ liệu được thu từ cảm biến quán tính MPU6050 gắn với vi điều khiển ESP32. Firmware trong thư mục data_collection/firmware đọc 6 kênh IMU và truyền qua Serial; ứng dụng thu thập lưu dữ liệu CSV với các cột time, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, label. Bộ dữ liệu huấn luyện chính được chuẩn hóa lại về 50Hz, mỗi trial dài 4 giây với 201 điểm thời gian.",
        "Triển khai pipeline thu thập và giám sát dữ liệu thời gian thực qua giao thức WebSocket/MQTT": "Triển khai pipeline thu thập và giám sát dữ liệu thời gian thực qua Serial và WebSocket, tích hợp logic FSM (Idle, Entry, Selection, Active).",
        "Phạm vi dữ liệu: tập trung vào tín hiệu gia tốc": "Phạm vi dữ liệu: tập trung vào tín hiệu gia tốc và con quay hồi chuyển 3 trục. Dữ liệu thô được thu từ ESP32 + MPU6050, sau đó chuẩn hóa thành bộ dữ liệu 50Hz để huấn luyện và đánh giá.",
        "Chưa đi sâu tối ưu kiến trúc học sâu": "Đề tài đã triển khai, huấn luyện và so sánh nhiều hướng mô hình gồm Random Forest, SVM, Gradient Boosting, XGBoost, CNN, LSTM và Transformer; phần tối ưu sâu cho triển khai trực tiếp trên vi điều khiển được xem là hướng mở rộng tiếp theo.",
        "Dữ liệu được chia thành train/validation/test": "Dữ liệu được chia theo người tham gia để đánh giá khả năng tổng quát hóa và hạn chế rò rỉ dữ liệu giữa các tập. Cấu hình hiện tại dùng train S01-S11 với 7.882 trial, validation S12-S13 với 1.432 trial và test S14-S15 với 1.330 trial.",
        "Mô hình đề xuất sử dụng mạng học sâu xử lý chuỗi IMU": "Các mô hình học sâu được triển khai trực tiếp trong training_50hz_clean/src/models.py gồm CNN1D, LSTMModel và TransformerModel. CNN1D khai thác mẫu cục bộ trên chuỗi 6 kênh, LSTM học phụ thuộc theo thời gian, còn Transformer dùng self-attention và positional encoding để tổng hợp đặc trưng toàn cửa sổ.",
        "Kiến trúc tham khảo: khối convolution 1D": "Kiến trúc CNN1D gồm các khối Conv1D - BatchNorm - ReLU - MaxPool, sau đó global average pooling và fully connected. LSTM dùng 2 lớp ẩn 128 chiều. Transformer chiếu 6 kênh vào d_model=128, dùng 2 encoder layer, 4 attention heads và classifier softmax 20 lớp.",
        "Dữ liệu được cắt theo cửa sổ trượt": "Dữ liệu đầu vào được biểu diễn dưới dạng ma trận T x 6, với T=201 sau khi resample về 50Hz trong cửa sổ 4 giây. Sáu kênh gồm ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps; mean/std được tính trên tập train và lưu trong normalization.json để dùng thống nhất khi suy luận.",
        "Xây dựng hệ thống mô phỏng điều khiển thị bị": "Hệ thống mô phỏng điều khiển thiết bị nhà thông minh gồm firmware ESP32, công cụ thu dữ liệu Streamlit, pipeline huấn luyện 50Hz, backend FastAPI và giao diện web demo. Backend cung cấp REST API, WebSocket đọc Serial thời gian thực, lớp dự đoán và FSM điều khiển TV, loa, đèn, rèm.",
        "Hình 3.x. Confusion matrix heatmap": "Hình minh họa sử dụng trong Chương 3 được sinh sẵn tại training_50hz_clean/results: model_accuracy_f1_comparison.png, confusion_transformer.png, tsne_transformer.png, per_activity_f1_score.png và cnn_training_loss_accuracy_curve.png.",
        "Gợi ý trình bày: mô tả ngắn mô hình tốt nhất": "Từ kết quả hiện tại, Transformer là mô hình học sâu phù hợp nhất cho demo nhờ Accuracy 99,85% và Macro F1 99,83%; các lỗi còn lại tập trung ở nhóm nhiễu N3/N4 có một phần nhỏ bị dự đoán thành G14. Do G14 là lệnh dừng hệ thống, backend bổ sung cơ chế xác nhận 6 giây và ngưỡng confidence cao để giảm rủi ro tắt nhầm.",
        "Các lỗi thường gặp gồm: nhầm lẫn giữa cử chỉ có quỹ đạo gần nhau": "Các lỗi còn lại chủ yếu xuất hiện ở các lớp có biên dạng gần nhau hoặc nhiễu sinh hoạt có chuyển động tay giống lệnh hệ thống. Trên confusion matrix Transformer, N3 và N4 có khoảng 1,67% mẫu bị lệch sang G14; các lớp lệnh G1-G15 còn lại đạt 100% recall trên tập test hiện tại.",
        "Giải pháp gồm tăng dữ liệu đa người dùng": "Giải pháp trong mã nguồn gồm tăng dữ liệu đa người dùng, chia tập theo subject, chuẩn hóa 50Hz, dùng ngưỡng confidence/cooldown, kiểm tra đúng thiết bị đang chọn và xác nhận riêng cho G14. Khi triển khai thật, cần tiếp tục thu thêm dữ liệu ở nhiều vị trí đeo và tốc độ thao tác khác nhau.",
        "Đồ án đã giải quyết trọn vẹn vòng đời": "Đồ án đã giải quyết trọn vẹn vòng đời của một bài toán AI ứng dụng: thiết kế phần cứng ESP32 + MPU6050, xây dựng bộ dữ liệu 20 lớp gồm 15 lệnh và 5 nhiễu, chuẩn hóa dữ liệu 50Hz, huấn luyện và so sánh CNN/LSTM/Transformer/baseline ML, sau đó tích hợp mô hình vào demo nhà thông minh qua FastAPI, WebSocket và FSM.",
        "Định hướng nghiên cứu tiếp theo sẽ tập trung": "Định hướng tiếp theo là mở rộng dữ liệu theo nhiều người dùng và bối cảnh sinh hoạt hơn, đánh giá ngoài tập kín, tối ưu mô hình cho suy luận biên bằng TensorFlow Lite/TinyML, bổ sung kiểm thử độ trễ end-to-end và triển khai suy luận trực tiếp trên thiết bị đeo khi tài nguyên phần cứng cho phép.",
        "[4] Streamlit Documentation": "[4] Streamlit Documentation, https://docs.streamlit.io",
        "[5] Plotly Python Documentation": "[5] PyTorch Documentation, https://pytorch.org/docs/stable/index.html",
        "[6] Flask Documentation": "[6] FastAPI Documentation, https://fastapi.tiangolo.com/",
        "CHƯƠNCÁC PHƯƠNG PHÁP NHẬN DẠNG": "CHƯƠNG 2. CÁC PHƯƠNG PHÁP NHẬN DẠNG",
    }
    for prefix, text in replacements.items():
        replace_first_paragraph(root, prefix, text, highlight=True)

    append_completion_notes(body)

    files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    temp = DOCX.with_suffix(".tmp.docx")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)
    try:
        temp.replace(DOCX)
    except PermissionError:
        try:
            temp.replace(OUTPUT_FALLBACK)
        except PermissionError:
            temp.replace(OUTPUT_HIGHLIGHTED_FALLBACK)


if __name__ == "__main__":
    main()
