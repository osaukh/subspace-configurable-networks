#! /bin/bash
curl -o ./data/modelnet.zip https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip \
&& unzip -q -d ./data ./data/modelnet.zip