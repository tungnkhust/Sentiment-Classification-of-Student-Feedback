# BTL_NLP

## Set up
- Cài đặt các package và thư viện liên quan
```bash
pip install -r requirement.txt
```

## Training

### 1 Training word embedding
#### Để huấn luyện word embedding từ đầu chạy lệnh sau:
```
python word2vec/train_w2v.py \
--corpus_path='data/wiki_corpus.txt' \
--model_dir='models/word2vec' \
--name='viki' \
--vector_size=100 \
--window=10 \
--n_epochs=10 \
--min_count=30 \
--max_final_vocab=20000
```
- corpus_path: đường dẫn tới corpus. corpus được lưu thành các file trong một thư mục.
- model_dir: đường dẫn tới mơi lưu model sau khi train.
- name: tên của file model sẽ được lưu.
- vector_size: số chiều của vector word embedding
- window: kích thước của một cửa sổ trượt
- n_epochs: số chu kỳ huấn luyện
- min_count: số lần xuất hiện nhỏ nhất của một từ để được thêm vào bộ từ điển (vocabulary)
- max_final_vocab=20000: số lượng lớn nhất có thể của bộ từ điển sau khi xây dựng.

#### Để huấn luyện thích ứng word embedding với bộ dữ liệu riêng chạy
```
python word2vec/train_w2v_adapt.py \
--corpus_path='data/sentence.txt' \
--model_path='models/viki/wiki_w2v.bin' \
--model_dir='models/adapt_w2v \
--name='viki_adapt' \
--n_epochs=20 \
```
- corpus_path: đường dẫn tới corpus. corpus được lưu thành các file trong một thư mục.
- model_path: đường dẫn tới model word embedding đã được huấn luận trước đó
- model_dir: đường dẫn tới mơi lưu model mới sau khi train.
- name: tên của file model mới sẽ được lưu.
- n_epochs: số chu kỳ huấn luyện
