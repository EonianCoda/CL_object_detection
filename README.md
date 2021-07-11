# 訓練檔案目錄說明

root
  code
  model
    \[scenario\]/\[task\]
  dataset
    \[dataset_name\]
      annotations
      images

Ex:
root
  code
  model
    20
      task1
    15+1
      task1
      task2
    19+1
      task1
      task2
  dataset
    voc2007
      annotations
      images
    voc2012
      annotations
      images
      
# 方法敘述
1. w_o_disillation: without distillation loss
2. w_distillaion: with  distillation loss

# Task

Task從1~n

# 參數
--sample: 為每個類別sample的圖片數量, 0表示不sample