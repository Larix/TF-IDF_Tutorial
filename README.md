# TF-IDF(Term Frequency - Inverse Document Frequency)
Calculate cosine-similarity between documents using TF-IDF
此專案以Python3進行開發，以新聞資料進行tf-idf結合cosine similarity實作的範例
### TF-IDF Introduction:
```
TF-IDF是一種統計方法，用以評估一字詞對於一個檔案集或一個語料庫中的其中一份檔案的重要程度。
字詞的重要性隨著它在檔案中出現的次數(TF)成正比增加，但同時會隨著它在語料庫中出現的頻率(IDF)成反比下降。
```
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/tf.JPG)
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/idf.JPG)


### Cosine Similarity Introduction:
```
餘絃相似度（cosine similarity）是資訊檢索中常用的相似度計算方式，可用來計算文件之間的相似度，
也可以計算詞彙之間的相似度，更可以計算查詢字串與文件之間的相似度。
```
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/cosine.jpg)
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/cosine_similarity.JPG)


### IDF補充:

![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/idf_high.jpg)
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/idf_low.jpg)

### 補充:
新聞資料大概只有200篇，斷詞使用jieba，有許多詞只出現在某一篇新聞文檔，考慮過濾這些詞，有可能是斷錯的詞彙。
