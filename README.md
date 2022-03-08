# 玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦

---

在此比賽中，我們的組別(把拉拉嘟嘟)在private leaderboard獲得第六名的成績(6th/317)。

## Description

- data_preprocessing.ipynb：針對原始資料進行前處理(long to wide format)，將會產生三份前處理後的資料。
- iDCG.ipynb：針對最後一個月份(dt=24)資料進行處理與計算iDCG，以進行線下驗證(NDCG score)。
- modeling_validation_mode.ipynb：驗證模式之模型訓練方式，取dt=24之資料作為驗證資料之正確標籤。
- modeling_testing_mode.ipynb：測試模式之模型訓練方式，最後預測dt=25之信用卡消費狀況。
- ensemble.ipynb：對三份資料訓練出來的模型預測結果進行blending。
- utils.py：計算NDCG分數之工具。
- sample_submission.csv：比賽官方提供的範例提交檔案格式。


## Links

- [比賽網頁](https://tbrain.trendmicro.com.tw/Competitions/Details/18)
- [簡報連結](https://drive.google.com/file/d/1otMvpmJVmZgouXvwgUf8SSaUhgD8zOde/view?usp=sharing)
- [作法說明(Youtube)](https://www.youtube.com/watch?v=RRbANwU5rzk)

