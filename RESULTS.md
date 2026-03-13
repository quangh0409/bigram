# Ket qua bai tap N-gram tieng Viet

## Thong tin mo hinh

**Dataset:** Wikipedia tieng Viet (Hugging Face: `tdtunlp/wikipedia_vi`)

## 1) Mo hinh ngon ngu N-gram muc am tiet

Mo hinh duoc cai dat trong:
- `ngram_vi.py` - Lop `BaseNgramVietnameseLM`
- `BigramVietnameseLM` - Lop con cho bigram (n=2)

Dac diem:
- **Token dac biet:** `<s>` (bat dau cau), `</s>` (ket thuc cau)

P(w_i | context) = (count(context, w_i) + alpha) / (count(context) + alpha * |V|)

Script chay:
- `run_experiment.py`

## 2) Xac suat cau

Da tinh tren mo hinh 4-gram huan luyen tu 1,000 bai Wikipedia tieng Viet:

### Cau 1: "Hôm nay trời đẹp lắm"
- P(sentence) = `2.538456937992e-28`
- log P(sentence) = `-63.540826`

## 3) Cau sinh ra tu mo hinh 4-gram

10 cau mau sinh ra:


01. liên xô đã khiến quân nhật trở nên tự đắc và phân tán ; và từ các cửa ô của

02. great american wiknic là một buổi chiều đầu thu năm 1792 , pháp khai quật được rất nhiều nhà thờ

03. tối cao pháp viện hoa kỳ đã và đang hình thành một bộ gọi là tạp a hàm . những

04. hoài niệm liên xô vẫn duy trì được vị thế quốc tế của liên xô tại balkan , trận không

05. quy hoạch tứ giác kinh tế . gdp tăng 11 % so với cùng kỳ năm 2016 , 5 ,

06. theo quyết định số 44 - hđbt ngày 1 tháng 2 năm 1876 , các thôn đổi thành làng .

07. hai mươi mục trong phán quyết của quan tòa jack weinstein , 7 công ty hóa chất mỹ , nhưng

08. khu du lịch văn hóa .

09. định nghĩa . hầu hết các chất rắn trong mạng , được đề ra hồi tháng 9 , 1964 của

10. tháng 6 năm 1951 , ông được thủ tướng chính phủ ban hành nghị quyết về thí điểm cơ chế


## Link du lieu

**Dataset:** Wikipedia tieng Viet
- Nguon: Hugging Face Datasets
- Dataset ID: `tdtunlp/wikipedia_vi`
- URL: https://huggingface.co/datasets/tdtunlp/wikipedia_vi


