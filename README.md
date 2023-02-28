# multimodal-recsys

### Quick Start (inference)


```
pip install -e .
```

```
hdfs dfs -copyToLocal hdfs://pct/user/nowrec/tmp/intern/multimodal/dataset
```

```
cd mmcr/cf_reg
hdfs dfs -copyToLocal hdfs://pct/user/nowrec/tmp/intern/multimodal/pretrained.tar.gz
tar -zxvf pretrained.tar.gz
```