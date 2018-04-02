#Benchmarking MXNetScala package

To run the package

```
mvn clean package
```

Tested on Resnet-152 model

Download Model files and test image as follows:

```
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params 
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json 
wget http://data.mxnet.io/models/imagenet-11k/synset.txt 
wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```

You can run the scripts as follows:

```
bash bin/old_scala.sh resnet/resnet-152 kitten.jpg images/ resnet/synset.txt 1

bash bin/new_scala.sh resnet/resnet-152 kitten.jpg images/ 1
```