# 毕业设计
## 中文文本相似度--句向量
## 整体实验设计
### 一、文本相似度的三种常用方法
&#160; &#160; &#160; &#160;本文主要利用余弦相似度衡量句子之间的相似度，
因此如何生成优质的句向量至关重要。

   |相似度|具体做法|
   |:----:|:----:|
   |Jaccard系数|分词<br>未分词|
   |编辑距离|最小操作次数|
   |余弦相似度|Word2vec+词+加权平均<br>Word2vec+字+加权平均<br>Word2vec+词+Qucik-thoughts<br>Word2vec+字+Quick-thoughts<br>字+融合Transformer的Quick-thoughts<br>词+融合Transformer的Quick-thoughts|

### 二、实验步骤
#### 第一步：数据标注
1. 计划标注：
      * 人工标注验证集（大约4000条数据，需要20天左右，10月25日之前完成。）
      * 思路：133个类别，每个类别标注30条数据。
      * 方法：利用关键词筛选出每个类别对应的数据。

2. 实际标注：
      * 实际只标注了2000条数据
      * 1500条有对应的类别，500条没有作为反例，共涉及80个类别。

#### 第二步：训练词向量
* 方法：Word2vec
* 具体做法：
     1. 下载最新的中文维基百科数据集（1.5g)
     2. 进行一系列预处理操作：
          * [wikiextractor](https://github.com/attardi/wikiextractor)解压，提取压缩包文本信息 wiki_00
          * 繁转简：opencc-> wiki.zh.txt
          * 分句+数据清洗（只保留数字、英文和汉字）：fen_ju.py-> new_sentence.txt
          * 是否分词: jieba 
          * 是否去停词：停词表
          * 统计句子长度、词频、字频，生成字典
     3. 分词后训练word2vec模型,生成2种词向量。
          * 分词+去停词+300维
          * 未分词+去停词+300维

#### 第三步：无监督学习的句向量生成算法
1. 在文本相似度任务中比较各种句向量生成算法的效果
      1. 加权平均算法
      2. Quick-Thoughts算法
      3. 将Transformer融入Quick-Thought算法中，观察其效果
      4. 比较分词与未分词的差别
2. 具体做法
   * 实际数据共有1982条进行预测，预定义语句5166条，因为部分数据预处理后为空
   * 本文修改了f1的计算方式---因为数据中存在没有对应的预定义操作的句子
   * 对于深度学习模型，本文的实验环境有限，只选择了80万条训练样本
   
   1. 方法一：基于词向量的简单平均算法. 共耗时1771.08s.

       |阈值|P|R|F1|
       |:----:|:----:|:----:|:----:|
       |0.5|0.6846|0.7070|0.6956|
       |0.6|0.6868|0.7070|0.6968|
       |0.7|0.6931|0.7070|0.7000|
       |0.8|0.8070|0.7050|0.7526|
       |0.9|0.9525|0.6167|0.7487|

   2. 方法二：基于字向量的简单平均算法，共耗时1762.93s.

       |阈值|P|R|F1|
       |:----:|:----:|:----:|:----:|
       |0.5|0.6970|0.7492|0.7221|
       |0.6|0.6996|0.7492|0.7235|
       |0.7|0.7703|0.7492|0.7596|
       |0.8|0.8749|0.7391|0.8013|
       |0.9|0.9607|0.5719|0.7170|

   3. 方法三：基于词向量的加权平均算法. 共耗时1834.97s.

       |阈值|P|R|F1|
       |:----:|:----:|:----:|:----:|
       |0.5|0.687219|0.715719|0.701180|
       |0.6|0.687219|0.715719|0.701180|
       |0.7|0.701180|0.715719|0.708375|
       |0.8|0.838095|0.706355|0.766606|
       |0.9|0.923699|0.534448|0.677119|

   4. 方法四：基于字向量的加权平均算法. 共耗时1787.23s.

        |阈值|P|R|F1|
        |:----:|:----:|:----:|:----:|
        |0.5|0.692551|0.733779|0.712569|
        |0.6|0.692988|0.733779|0.712801|
        |0.7|0.754301|0.733110|0.743555|
        |0.8|0.879508|0.717726|0.790424|
        |0.9|0.949233|0.537793|0.686593|

   5. 方法五：基于字向量和词向量的加权平均算法. 共耗时1762.93s.
      <br>生成的句向量是600维

        |阈值|P|R|F1|
        |:----:|:----:|:----:|:----:|
        |0.5|0.697328|0.750502|0.722938|
        |0.6|0.698195|0.750502|0.723404|
        |0.7|0.763624|0.749833|0.756666|
        |0.8|0.890433|0.728428|0.801325|
        |0.9|0.950423|0.525753|0.677003|
        
   6. 方法六：基于词向量的Quick_Thoughts算法
      1. 生成训练样本Tfrecords文件---preprocess_dataset.py
      2. 训练模型---train.py
      
         |需要调节的参数|解释说明|示例|
         |:----:|:----:|:----:|
         |word2vec_path|word2vec文件的目录|../data/sent_word_n/|
         |output_dir|生成句向量的目录|../output/sent_char_n/|
         |input_file_pattern|tfrecord文件的命名格式|../output/sent_word_n/train-?????-of-00010|
         |train_dir|模型文件的保存位置|..model/train/sent_char_n|
      3. 预测：生成句向量---predict.py
      4. 训练结果
         * 生成句向量消耗的时间：14.37s（小爱数据，1982条），24.17s（预定义数据集，5166条）
         * 整个预测过程共消耗21950.48s
 
           |阈值|P|R|F1值|
           |:----:|:----:|:----:|:----:|
           |0.5|0.655587|0.620067|0.637332|
           |0.6|0.655587|0.620067|0.637332|
           |0.7|0.655587|0.620067|0.637332|
           |0.8|0.661670|0.620067|0.640193|
           |0.9|0.717054|0.618729|0.664273|
           |0.94|0.847909|0.596656|0.700432|
           
   7. 方法七：基于字向量的Quick-Thoughts算法
      * 训练结果：
          * 生成句向量消耗的时间：
               13.72s（小爱数据，1982条），23.98s（预定义数据集，5166条）
          * 共消耗21926.511472 s
         
             |阈值|P|R|F1值|
             |:----:|:----:|:----:|:----:|
             |0.5|0.673374|0.671572|0.672472|
             |0.6|0.673374|0.671572|0.672472|
             |0.7|0.673826|0.671572|0.672697|
             |0.8|0.679756|0.671572|0.675639|
             |0.9|0.839361|0.667559|0.743666|
             |0.92|0.900742|0.649498|0.754761|
       
   8. 方法八：融合Transformer的Quick-Thoughts算法（分词, 2700维）
      1. Transform的编码器得到的是一个[seq_length, dim]的向量，
         因此探索Transformer编码器生成句向量的处理方式，然后与Quick-Thoughts融合
           
         * 生成句向量消耗的时间：26.67s（小爱数据，1982条），40.14s（预定义数据集，5166条）
         * 模型参数设置（调整后得到的最优参数）：
         
           |参数|取值|
           |:----:|:---:|
           |num_head|2|
           |learing_rate|0.001|
           |dim|2700|
           |dropout_rate|0.3|
           |batch_zie|128|
           |num_ceng|2|
           
         * 预测结果对比：
         
           |处理方式|F1值（最优）|预测消耗总时间（s）|
           |:----:|:----:|:----:|
           |简单平均|0.63|26122.49|
           |直接求和|0.33|24774.29|
           |标准化后平均|0.40|24528.78|
           |标准化后求和|0.39|27671.44|
           |对简单平均后的向量进行标准化|0.45|24164.85|
           |对直接求和后的向量进行标准化|0.1|21877.37|
           
           最优的F1值为0.69，阈值为0.999588。
           
         **结论：对于Transformer模型生成的词向量进行简单平均效果最好**
         
   9. 方法九：融合Transformer的Quick-Thoughts算法（未分词，2700维）
        * 基于字向量的Transformer模型：
             1. 损失函数值：1102.33
             2. 训练时间： 48704.05
         
        **结果：最优的F1值为0.75，阈值为0.999442。**

   10. 方法十：Transformer编码器+简单平均

          |模型|阈值|最优的F1值|预测时间|日志文件名|
          |:----:|:---:|:---:|:---:|:---:|
          |基于字向量的Transformer模型|0.999991|0.41|8686.1|tr_char|
          |基于词向量的Transformer模型| 0.999999|0.30|9753.86|tr_word|
  
        **结论：仅利用Transformer编码器无法揭示句子之间的相似程度。**

#### 第四步：训练细节
  **利用sent_word_rem数据集对Transformer进行调参**
  
  1. **第一次调参时模型存在部分错误：**
  
      1. mask应该相乘而不是相加
      2. 多头注意力的输出应该添加一个线性连接层
      3. 多头注意力层和全连接层没有添加drpout
      4. 词向量矩阵应该随机正态化，没有对词向量矩阵和切分后的q进行归一化
  2. 第一调参得到的部分错误结果    
      * （1）观察预先训练的词向量对模型结果的影响
    
          |词向量表示方式|损失函数值|
          |:----:|:----:|
          |预先训练|1257.96|
          |随机初始化|1224.76|
   
      * （2）编码器和解码器层数对模型结果的影响
    
        |层数|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |2|1224.76|63801.97|
        |5| |batch_size=128|

      * （3）编码维度对模型的结果影响
    
        |维度|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |300|1224.76|63801.97|

      * （4）头数对模型结果的影响(双层，batch_size=128)
      
        |头数|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |2|1225.45|67956.94|
        |4|1224.76|63801.97|
        |6|1225.45|68619.52|

      * （5）batch_size对模型的影响(双层，num_head=4))
      
        |batch_size|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |128|1224.76|63801.97|
        |64|608.16|68647.59|

  3. **第二次调参**
      * batch_size =128, ceng_shu=2, num_head=6
      * （1）batch_size对模型的影响(双层，num_head=4))
      
        |batch_size|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |512|OOM|OOM|
        |256|2457.10|70999.85|
        |128|1242.77|71327.49|
        |64|623.93|71453.79|
    
         **问题**
      
               1. 为什么batch_size一般选择2的幂次？
                  <br>因为GPU对2的幂次的batch可以发挥更佳的性能
               2. batch_size对模型效果的影响？
                  （1）batch_size过大，训练消耗的时间会缩短，但是模型容易陷入局部最优点。
                     因为样本方差较小，可能会呆在一个局部最优点不动。
                  （2）batch_size过小，模型同样不易收敛，损失函数容易震荡
          
      * （2）编码器和解码器层数对模型结果的影响（batch_size=128,num_head=2)
    
        |层数|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |2|1143.48|71261.14|
        |3|1222.36|83205.03|
        |4|1224.50|94817.31|
        |5|1225.24|106313.79|
        |6|1229.86|117888.51|
    
      * （3）编码维度对模型的结果影响
      
             batch_size=128,num_head=8,
             num_units=[512,2048],num_ceng=2
             dim = 200时，训练结果不稳定，参考loss.png(16,17)
             dim = 300时，num_head = 2
    
        |维度|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----|
        |200|1143.52|53102.00|
        |256|1143.49|63260.60|
        |300|1143.48|71201.66|
        |400|1143.48|94607.44|
        |512|1143.47|123626.50|
    
      * （4）头数对模型结果的影响(双层，batch_size=128)
      
        |头数|损失函数值|训练模型消耗时间（s)|
        |:----:|:----:|:----:|
        |2|1143.48|71261.14|
        |4|1227.60|71728.84|
        |5|1143.78|71856.85|
        |6|1143.48|71636.31|
        |10|1143.48|73083.46|
        |20|1143.48|74627.01|
        
        **问题**  
           
             1. 为什么头数为4的时候损失函数值比较高？
      * （5）学习率对模型结果的影响（双层, num_head = 6, batch_size=128)
    
        |rate|损失函数值|F1值（最优）|阈值|训练模型消耗时间（s)|预测消耗时间（s)|
        |:----:|:----:|:----:|:----:|:----:|:----:|
        |0.00005|1235.80| | |120375.26|
        |0.000275|||
        |0.0005|1143.48| | |71636.31|
        |0.001|1143.48|0.69|0.999588|62813.25|9296.24|
        |0.003|1143.47| | |72178.98|
        |0.005|1143.47| | |71888.02|
        |0.01|1143.80|0.69|0.989927|67776.89|9477.49|
        
      * （6）探索直接输入预训练的词向量还是随机初始化效果较好
           （num_head =2, dim=300, num_units=[1200, 300], loss_19.png)
        
        * 开始主观设定阈值的取值范围是0.5：1：0.01
        
          |是否采用预训练的词向量作为输入|损失函数值|训练模型消耗时间（s)|
          |:----:|:----:|:----:|
          |是|1143.51|62143.24|
          |否|1143.48|71201.66|
        
        * 然后调整阈值取值范围为最小值：最大值：（最大值-最小值）/20
        
            |是否采用预训练的词向量作为输入|F1值（最优）|训练模型消耗时间（s)|
            |:----:|:----:|:----:|
            |是|0.63|26122.49|
            |否|0.63|21611.44|
        
        **结论**：利用预训练得到的词向量得到的输入，模型训练过程中会发生震荡，
          但是最终的结果与随机初始化相差不大，而且利用预先训练的词向量训练模型消耗的时间较少。
          
      *  (7) 探索dropout的影响（随机初始化词向量）
      
             dim =300, num_head =2, num_units = [1200, 300],
             num_ceng =2, batch_size = 128, 
             learning_rate = 0.0005
             阈值是用（最大值-最小值）/20为步长挑选出来的
       
         |rate|损失函数值|F1值（最优）|阈值|训练模型消耗时间（s)|预测消耗时间（s)|
         |:----:|:----:|:----:|:----:|:----:|:----:|
         |0.1|1143.47|0.69|0.999607|62943.26|9107.41|
         |0.3|1143.48|0.69|0.999605|62623.19|9271.86|
         |0.5|1143.58|0.69|0.999606|62186.54|9137.37|
         |0.8|1144.84|0.69|0.999590|63128.25|9588.84|
     
                 
  **调参结果**
  
       1. dim =300, num_head =2, num_units = [1200, 300],
          num_ceng =2, batch_size = 128, rate=0.3
       2. learning_rate = 0.0005, 梯度更新公式在论文的基础上乘以learning_rate效果更好
       3. 是否采用预训练的词向量对损失函数值影响不大（已有论文证明，而且本文的结果也证明是否采用预训练的词向量影响不大），但是采用预训练的词向量消耗的时间较短。
          在预测时对比两种模型在真是数据集中的效果，发现效果差别不大。
       4. 受硬件限制无法训练一个与原论文相同的6层512维的模型，大概需要三天左右才能训练结束。
   
          
#### 第五步：实验结果
  1. 实验环境:**2核8g服务器**
  2. 生成处理后的数据集需要的时间
  
     |数据集|生成字向量或者词向量(s)|获取词频文件(s)|
     |:----:|:----:|:----:|
     |sent_char_n.txt|4065.48|445|
     |sent_char_rem.txt|2663.45|334|
     |sent_word_n.txt|4594.10|443|
     |sent_word_rem.txt|3359.56|366|
     
  3. Quick-thoughts训练过程统计
     * 第一次生成训练数据的时间
  
         |数据集|词或字的总数量|词向量中字或词的数量|统计词频(s)|生成词典和词向量文件(s)|样本中句子包含<br>最少的词数量|样本中句子<br>包含最多的词数量|训练时的词数量 |
         |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
         |sent_char_n.txt|19354|12820|111.06|10.43|2|874|12820|
         |sent_char_rem.txt|18936|12403|78.99|10.28|2|767|12403|
         |sent_word_n.txt|3198221|617297|117.14|140.89|2|432|20000|
         |sent_word_rem.txt|3196599|615709|92.02|141.54|2|366|20000|
    
         |数据集|生成TF文件耗费时间（s)|训练模型消耗时间（s)|预测时间(s)|||||
         |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
         |sent_char_n.txt|86.21|29555.58|
         |sent_char_rem.txt|76.69|29743.32|
         |sent_word_n.txt|70.86|29528.91|
         |sent_word_rem.txt|80.22|29533.81|
         
     * 第二次重新分句后生成训练数据的时间
       
       |文件|字或词的数目|训练样本中长度<br>小于30的句子数量|句子的最短长度|句子的最大长度|生成record文件<br>消耗的时间(s)|
       |:----:|:----:|:----:|:----:|:----:|:----:|
       |new_sent_char_rem.txt|9591|640195|2|997|78.80|
       |new_sent_word_rem.txt|20000|938007|2|300|72.19|

  4. 第一次训练的最终结果：
     * 第一次利用Quick-Thoughts预测时，阈值是固定的（0.9)。
     * 第一次利用融合模型预测的时候，阈值是从0.5~1，步长为0.01，所以预测时间较长。
     * 第一次利用融合模型预测时，基于词向量的融合模型效果较差，因此没有预测基于字向量的模型。
     * 第一次训练时部分日志文件丢失，无法获取准确的训练时间。
     
       |模型|阈值|最优的F1值|预测时间|
       |:----:|:----:|:----:|:----:|
       |基于词向量的简单平均（去停词）|0.8|0.7526|1771.08|
       |基于字向量的简单平均（去停词）|0.8|0.8013|1762.93|
       |基于词向量的加权平均（去停词）|0.8|0.7666|1834.97|
       |基于字向量的加权平均（去停词）|0.8|0.7904|1787.23|
       |基于字向量和词向量的加权平均|0.8|0.8013|1762.93|
       |基于词向量的Quick-Thought Vectors算法|0.9|0.743666|2253.14|
       |基于字向量的Quick-Thought Vectors算法|0.9|0.664273|2250.50|
       |融合Transformer的Quick-Thought Vectors算法（分词+去停词）|0.8|0.396015|24528.78|
       |融合Transformer的Quick-Thought Vectors算法（不分词+去停词）|-|-|-|

  5. 第二次重新分句后的最终结果
     * 对于简单平均算法和加权平均算法没有重新生成词向量，进行训练。
     * 重新对数据进行清洗，删除维基百科中部分无意义的信息，
       并且按照中文习惯重新进行分句。
     * 训练模型模型时，全都随机初始化词向量。（所以Quick-thoughts效果较差）
   
       **实验结果**
   
       |模型|阈值|最优的F1值|训练时间（s)|预测时间(s)|
       |:----:|:----:|:----:|:----:|:----:|
       |基于字向量的Quick-Thoughts模型|0.922823|0.601799|31274.17(8.7h)|9354.07|
       |基于词向量的Quick-Thoughts模型|0.885814|0.647385|31443.50(8.7h)|9192.68|
       |基于字向量的Transformer模型|0.999912|0.725537|47057.17(13h)|8854.41|
       |基于词向量的Transformer模型|0.999942|0.575241|72569.82(20h)|8830.47|
       |融合Transformer的Quick-Thought模型| | |31274.17(8.7h)+47057.17(13h)||
       |融合Transformer的Quick-Thought模型|||31443.50(8.7h)+72569.82(20h)|
       
     * 结论：
          1. 两个编码器全都随机初始化词向量后，Quick-Thoughts的性能下降。
             并且基于字向量的模型比基于词向量的模型效果较差。
          2. 仅用Transformer编码器，基于字向量的模型效果较好，基于词向量的模型效果特别差。
   

### 三、其它
* 已解决的问题
  1. 词向量文件太大, 无法加载(已解决)---利用np.load加载.npy文件，直接就是numpy数组
  2. 为什么字输入比词输入消耗时间少？---原因：字的维度较小，权重矩阵较小。
  3. 换新电脑后经常出现无法打开GitHub的官网情况，原因为本地的DNS无法进行解析，
可以修改C:\Windows\System32\drivers\etc\hosts文件，具体细节参考
[连不上GitHub的解决方案](https://blog.csdn.net/believe_s/article/details/81539747)
  4. label smoothing 标签平滑
  5. batch_size固定的太死，无法预测非batch_size的东西
  6. 预测时的batch必须要与训练时的batch保持一样吗？---不一定
  7. 不足一个batch的数据在预测时是如何处理的？--- tf.shape可以获取变量的维度信息，就算是维度为None
  8. transformer的输入是等长的还是不等长的？
    * 编码器-解码器的每个batch的长度不一样，每个batch填充到这个batch内的最大长度。
  9. Quick-thoughts算法不需要对句子进行padding，transformer的每个batch需要进行padding
  10. os.remove只能删除文件，shutil.rmtree可以删除指定的目录
  11. 第一次生成的vocab.txt中含有特殊字符，需要重新生成。
    之前的vocab.txt是利用词向量文件重新生成的，所以要想生成新的必须首先生成词向量。
    第二次修改直接选择词频较高的词语作为字典，总数量不超过20000。
  12. 重新生成训练样本文件，深度学习模型的词向量嵌入均随机初始化。
  
* 未解决的问题
   1. 长度和重合度数据集未标注完成，共2000条,并且标注后的样本去停词后会变成空值
   2. 受硬件限制，无法探索词向量的维度和句向量的维度对模型效果的影响
   3. 未登录词如何处理？
        <br>目前采取的方法：随机初始化，用0进行padding
        <br>（1）加unnk,索引为1
        <br>（2）随机初始化一个向量
        <br>（3）padding 和未登录词的区别
   4. quick-thoughts 效果较差
       * 基于词向量的F1值0.7，基于字向量的F1值0.75.
   5. loss一直保持不变，是什么原因？
        * 猜测原因：
          1. 说明参数一直都没有得到更新
          2. 没有添加dropout，为啥dropout可以防止过拟合？dropout相当于集成
          3. batch_size过大，网络会收敛到局部最优点；
             batchz_size太小，类别较多时，loss可能会一直震荡
          4. 学习率过大，transformer的学习率在论文中有对应的公式
        * 真实原因：
          1. 按照论文中的学习率公式修改学习率后，损失函数缓慢下降，但是又开始震荡
             * 论文中的学习率公式如何得到的？？？
             * 为什么按照论文中的公式表示学习率就会缓慢下降？？？？
               理论上学习率如果足够小，肯定可以收敛。、
   6. beam search在预测时会遇到，本文参考tensor2tensor的代码，并没有看懂    
       *  主要函数：
            1. grow_alive 
            2. grow_finished 
            3. grow_topk 
            4. inner_loop
            5. is_not_finished
       *   不懂的点
            1. tf.bitcast()具体是怎末实现的
            2. 长度惩罚项及结束搜索的条件
            3. 主要函数的功能

   7. 机器翻译时，解码器的第一个输入是`<s>`表示开头，直到`<e>`结束
   8. 对词层面上的优化（感觉意义不大）：
        1. 对句子中所有的词向量取最大值
        2. 对句子中所有的词向量取最小值
        3. 对句子中所有的词向量取平均值
        生成句向量：
            1. 进行拼接
            2. 删除掉最大值和最小值,然后再进行平均
   9. 第二次重新分句后没有训练词向量和字向量文件，quick-thoughts算法只能全部随机初始化，效果较差。
   
###五、 参考链接
   https://blog.csdn.net/CiciliarCai/article/details/52948275
   
#目标：
    今日任务：
        预测融合后的模型效果
        统计实验时间和结果