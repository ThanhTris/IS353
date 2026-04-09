# CARE-GNN trên Google Colab — YelpChi Spam Detection

## Bước 1: Clone repo và cài thư viện
```python
!git clone https://github.com/ThanhTris/IS353.git
%cd IS353
!pip install torch numpy scipy scikit-learn matplotlib seaborn -q
```

## Bước 2: Giải nén YelpChi data
```python
import zipfile
with zipfile.ZipFile('data/YelpChi.zip', 'r') as z:
    z.extractall('data/')
print("Done! Files:", [f for f in __import__('os').listdir('data/')])
```

## Bước 3: Tạo adjacency lists (chạy 1 lần ~2 phút)
```python
!python data_process.py
```

## Bước 4: Train + sinh kết quả (~30-60 phút CPU / ~10 phút GPU)
```python
!python train.py
```

## Bước 5: Xem kết quả
```python
from IPython.display import Image, display

print("=== Class Distribution ===")
display(Image('results/class_distribution.png'))

print("=== Precision-Recall Curve ===")
display(Image('results/precision_recall_curve.png'))

print("=== t-SNE Embeddings ===")
display(Image('results/tsne_embeddings.png'))

print("=== Training Curve ===")
display(Image('results/training_curve.png'))
```

## Bước 6: Xem Case Study
```python
with open('results/case_study.txt', 'r', encoding='utf-8') as f:
    print(f.read())
```

## Bước 7: Tải kết quả về máy
```python
from google.colab import files

# Tải từng ảnh
files.download('results/class_distribution.png')
files.download('results/precision_recall_curve.png')
files.download('results/tsne_embeddings.png')
files.download('results/training_curve.png')
files.download('results/case_study.txt')

# Hoặc nén hết 1 lần
!zip -r results.zip results/
files.download('results.zip')
```
