# ASR Pipeline 量化对比评估报告 (Rust)

*生成时间：2026-03-04 21:46*
*测试集：67 条音频（corpus.json + real_manifest.json）*
*Pipeline：Qwen3-ASR (离线), Qwen3-ASR (流式)*
*运行方式：Rust C API（libqwen_asr.dylib via ctypes）*

**CER 计算方式**：保留标点符号，仅做 lower + 去空格后计算字符错误率。

**测试环境**：Apple Silicon (aarch64), 总音频时长 421.3s

> 注：离线使用 `qwen_asr_transcribe_pcm()`，流式使用 `qwen_asr_stream_push()` + 2s chunk（与 Swift FFI 同一代码路径）。模型仅加载一次。

---

## 1. 总体 CER 汇总

| Pipeline | 平均 CER | CER=0 条数 | CER≤0.10 | CER≤0.20 | CER>0.20 | 总推理时长 | RTF |
|----------|:-------:|:---------:|:-------:|:-------:|:-------:|:---------:|:---:|
| Qwen3-ASR (离线) | 0.0571 | 36/67 | 52 | 63 | 4 | 32.8s | 0.078x |
| Qwen3-ASR (流式) | 0.0562 | 34/67 | 53 | 64 | 3 | 64.9s | 0.154x |

---

## 2. 按类别 CER 汇总

| 类别 | 条数 | Qwen3-ASR (离线) | Qwen3-ASR (流式) |
|------|:----:|:------:|:------:|
| chinese_long | 1 | 0.000 | 0.000 |
| chinese_short | 1 | 0.000 | 0.000 |
| code_switching | 5 | 0.017 | 0.017 |
| developer_corpus | 8 | 0.056 | 0.056 |
| english_short | 1 | 0.000 | 0.091 |
| long_audio | 2 | 0.007 | 0.019 |
| mid_sentence_pause | 2 | 0.000 | 0.000 |
| mixed_technical | 1 | 0.120 | 0.120 |
| mixed_zh_en | 1 | 0.000 | 0.000 |
| punctuation | 3 | 0.069 | 0.069 |
| real_aishell | 8 | 0.000 | 0.000 |
| real_ascend_codeswitching | 9 | 0.138 | 0.108 |
| real_codeswitching | 8 | 0.014 | 0.017 |
| real_conversational | 3 | 0.027 | 0.027 |
| real_wenetspeech | 10 | 0.095 | 0.102 |
| speech_rate | 2 | 0.263 | 0.263 |
| speech_trailing_silence | 1 | 0.000 | 0.000 |
| technical_numbers | 1 | 0.033 | 0.033 |

---

## 3. 逐条 CER 详细

| # | ID | Qwen3-ASR (离线) | Qwen3-ASR (流式) | 期望文本 |
|---|-----|:------:|:------:|------|
| 1 | zh_short_01 | 0.000 | 0.000 | 今天天气真好。 |
| 2 | zh_long_01 | 0.000 | 0.000 | 人工智能正在深刻地改变我们的生活方式，从语音识别到自动驾驶，从医疗诊断到金融分析。 |
| 3 | mixed_01 | 0.000 | 0.000 | 我今天用Python写了一个API接口。 |
| 4 | mixed_02 | 0.120 | 0.120 | MacBook Pro M3芯片性能提升了百分之40。 |
| 5 | en_short_01 | 0.000 | 0.091 | Hello world. |
| 6 | tech_num_01 | 0.033 | 0.033 | 服务器IP地址是192.168.1.100，端口号8080。 |
| 7 | noise_01 | 0.000 | 0.000 | 你好。 |
| 8 | dev_git_01 | 0.050 | 0.050 | 执行git commit，修复登录bug。 |
| 9 | dev_swift_01 | 0.045 | 0.045 | 定义一个struct叫做UserModel。 |
| 10 | dev_rust_01 | 0.000 | 0.000 | 在Rust里面用async await处理并发。 |
| 11 | dev_k8s_01 | 0.000 | 0.000 | Kubernetes的pod状态是CrashLoopBackOff。 |
| 12 | dev_api_01 | 0.000 | 0.000 | 调用RESTful API返回JSON格式数据。 |
| 13 | dev_db_01 | 0.125 | 0.125 | 执行SQL查询SELECT FROM users WHERE id = 1。 |
| 14 | dev_url_01 | 0.077 | 0.077 | 访问github.com。 |
| 15 | dev_debug_01 | 0.150 | 0.150 | 在第42行设置一个breakpoint。 |
| 16 | cs_var_01 | 0.087 | 0.087 | 把这个variable赋值给constant。 |
| 17 | cs_build_01 | 0.000 | 0.000 | 在macOS上运行swift build。 |
| 18 | cs_error_01 | 0.000 | 0.000 | 这个error是null pointer exception。 |
| 19 | cs_deploy_01 | 0.000 | 0.000 | 把Docker image push到registry。 |
| 20 | cs_review_01 | 0.000 | 0.000 | 帮我review一下这个pull request。 |
| 21 | punct_question_01 | 0.000 | 0.000 | 你今天吃饭了吗？ |
| 22 | punct_exclaim_01 | 0.000 | 0.000 | 太好了，我成功了。 |
| 23 | punct_list_01 | 0.208 | 0.208 | 第一步打开终端，第二步输入命令，第三步确认执行。 |
| 24 | rate_fast_01 | 0.526 | 0.526 | 快速语音识别测试，1、2、3、4、5。 |
| 25 | rate_slow_01 | 0.000 | 0.000 | 慢速语音识别测试。 |
| 26 | long_30s_01 | 0.010 | 0.015 | 人工智能技术在过去10年中取得了巨大的进步。深度学习算法使得计算机能够处理和理解自然语言。语音识别技术已经广泛应用于智能... |
| 27 | long_60s_01 | 0.003 | 0.023 | 软件工程是一门研究用工程化方法构建和维护有效的实用的和高质量的软件的学科。它涉及到程序设计语言、数据库、软件开发工具、系... |
| 28 | pause_mid_01 | 0.000 | 0.000 | 打开终端。 |
| 29 | pause_long_01 | 0.000 | 0.000 | 我想要一杯咖啡。 |
| 30 | aishell_test_001 | 0.000 | 0.000 | 甚至出现交易几乎停滞的情况。 |
| 31 | aishell_test_002 | 0.000 | 0.000 | 一二线城市虽然也处于调整中。 |
| 32 | aishell_test_003 | 0.000 | 0.000 | 但因为聚集了过多公共资源。 |
| 33 | aishell_test_004 | 0.000 | 0.000 | 为了规避三四线城市明显过剩的市场风险。 |
| 34 | aishell_test_005 | 0.000 | 0.000 | 标杆房企必然调整市场战略。 |
| 35 | aishell_test_006 | 0.000 | 0.000 | 因此，土地储备至关重要。 |
| 36 | aishell_test_007 | 0.000 | 0.000 | 中原地产首席分析师张大伟说。 |
| 37 | aishell_test_008 | 0.000 | 0.000 | 一线城市土地供应量减少。 |
| 38 | conv_zh_001 | 0.080 | 0.080 | 你好，我想要了解一下我的银行账户余额有多少，谢谢。 |
| 39 | conv_zh_004 | 0.000 | 0.000 | 我想要查询我的账户余额。 |
| 40 | conv_zh_005 | 0.000 | 0.000 | 您好，我可以知道我的账户余额吗？ |
| 41 | ascend_cs_001 | 0.095 | 0.095 | No，我专业是那个ISM，Information Systems Management。 |
| 42 | ascend_cs_002 | 0.040 | 0.040 | 嗯，所以你现在还是比较focus在找工作这件事上。 |
| 43 | ascend_cs_003 | 0.324 | 0.000 | 深圳啊，或者是上海这种比较大的城市，会有更多opportunity。 |
| 44 | ascend_cs_004 | 0.357 | 0.357 | 嗯，I like hot pot。 |
| 45 | ascend_cs_005 | 0.061 | 0.082 | 所以我的我的parents，我的妈妈是chemistry老师，and我的爸爸是history老师。 |
| 46 | ascend_cs_006 | 0.123 | 0.158 | 那个玩basketball的，然后我有时候有时候会邀我的friends啊，一起打在就是after class的时候。 |
| 47 | ascend_cs_008 | 0.043 | 0.043 | 然后呃，我也喜欢play basketball。 |
| 48 | ascend_cs_009 | 0.200 | 0.200 | 然后刚忘了讲，你你是念什么major的？ |
| 49 | ascend_cs_010 | 0.000 | 0.000 | 哦，我我在UG的时候念的是electrical engineering。 |
| 50 | wenet_net_001 | 0.000 | 0.000 | 毕业歌会之后，然后我们还去吃个饭，然后就感觉。 |
| 51 | wenet_net_002 | 0.115 | 0.135 | 竖锯癌症病成那样，还打着点滴，就更不可能把女警官吊了起来。说来说去，皮特认为还有其他人在帮助竖锯这么做。 |
| 52 | wenet_net_003 | 0.192 | 0.192 | 当时心里想，我只要能跪我就能站，我在床上练着跪着走。 |
| 53 | wenet_net_004 | 0.188 | 0.188 | 下车后望着30多层的大高楼发呆。 |
| 54 | wenet_net_005 | 0.000 | 0.023 | 还有剧作模式的双线性叙事、结尾神反转等等，也成为了日后电锯惊魂系列在剧作上的结构模式。 |
| 55 | wenet_net_006 | 0.122 | 0.122 | 这位叫皮特的FBI探员一上来就一顿物理分析，认为阿曼达不可能吊起比她还重的女警官。 |
| 56 | wenet_net_007 | 0.083 | 0.083 | 她已经在商场里开起了小店铺，尽管孤身一人，但与好友见面时还是会爽朗一笑。 |
| 57 | wenet_net_008 | 0.000 | 0.000 | 把这些劳工抓起来，送到月亮岛上去。 |
| 58 | wenet_net_009 | 0.108 | 0.135 | 的的需要。嗯，如果你把他当成产品的话，你就会觉得那么消费者会需要什么样的。 |
| 59 | wenet_net_010 | 0.143 | 0.143 | 媒体也已经报了，然后呃，债主也已经围楼了。 |
| 60 | cs_edge_001 | 0.000 | 0.000 | 我们团队最近在用React和TypeScript重构前端项目。 |
| 61 | cs_edge_002 | 0.000 | 0.000 | 这个bug是因为race condition导致的memory leak。 |
| 62 | cs_edge_003 | 0.024 | 0.024 | 用Docker Compose部署了3个microservice到staging环境。 |
| 63 | cs_edge_004 | 0.023 | 0.023 | 在GitHub上提了一个issue，关于performance optimization。 |
| 64 | cs_edge_005 | 0.000 | 0.000 | 这个function的return type应该是Optional，而不是force unwrap。 |
| 65 | cs_edge_006 | 0.000 | 0.024 | 用Xcode的Instruments做了一下profiling，发现CPU占用太高。 |
| 66 | cs_edge_007 | 0.000 | 0.000 | GraphQL的schema定义比RESTful API更灵活一些。 |
| 67 | cs_edge_008 | 0.067 | 0.067 | CI pipeline跑了30分钟，还没通过unit test。 |

---

## 4. 高 CER 条目详情 (CER > 0.20)

### Qwen3-ASR (离线)

| # | ID | CER | 期望文本 | 实际输出 | 分析 |
|---|-----|:---:|---------|---------|------|
| 1 | rate_fast_01 | 0.526 | 快速语音识别测试，1、2、3、4、5。 | 快速语音识别测试：一二三四五。 | |
| 2 | ascend_cs_004 | 0.357 | 嗯，I like hot pot。 | Hmm, I like hot pot. | |
| 3 | ascend_cs_003 | 0.324 | 深圳啊，或者是上海这种比较大的城市，会有更多opportunity。 | 深圳啊，或者是上海这种比较大的城市，会有更多不听。 | |
| 4 | punct_list_01 | 0.208 | 第一步打开终端，第二步输入命令，第三步确认执行。 | 第一步，打开终端。第二步，输入命令。第三步，确认执行。 | |

### Qwen3-ASR (流式)

| # | ID | CER | 期望文本 | 实际输出 | 分析 |
|---|-----|:---:|---------|---------|------|
| 1 | rate_fast_01 | 0.526 | 快速语音识别测试，1、2、3、4、5。 | 快速语音识别测试：一二三四五。 | |
| 2 | ascend_cs_004 | 0.357 | 嗯，I like hot pot。 | Hmm, I like hot pot. | |
| 3 | punct_list_01 | 0.208 | 第一步打开终端，第二步输入命令，第三步确认执行。 | 第一步，打开终端。第二步，输入命令。第三步，确认执行。 | |

