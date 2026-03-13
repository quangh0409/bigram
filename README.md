# Bigram Language Model cho tieng Viet (muc am tiet)

Du an nay xay dung mo hinh ngon ngu bigram cho tieng Viet o muc am tiet, su dung token dac biet `<s>` va `</s>`.

## Nguon du lieu

Corpus duoc tai tu bo:

**Dataset:** Wikipedia tieng Viet (Hugging Face: `tdtunlp/wikipedia_vi`)

Script tai va trich xuat cau:

- data/download_ud_vtb.py

Corpus sau khi trich xuat cau:

- data/vi_sentences_ud_vtb.txt

## Cach chay

```bash
python run_experiment.py
```

Neu file corpus chua ton tai, script se tu dong tai ve.

## Noi dung thuc hien

1. Xay dung mo hinh bigram o muc am tiet (tach token theo khoang trang va dau cau).
2. Tinh xac suat cau "Hom nay troi dep lam".
3. Sinh mot so cau tu mo hinh da huan luyen.

## Tep ma nguon chinh

- ngram_vi.py
- run_experiment.py
- data/download_ud_vtb.py
