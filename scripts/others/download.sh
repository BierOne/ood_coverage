DIR=$1

wget -P ${DIR} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget -P ${DIR} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget -P ${DIR} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget -P ${DIR} https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

tar -xf ${DIR}/iNaturalist.tar.gz -C ${DIR}
tar -xf ${DIR}/SUN.tar.gz -C ${DIR}
tar -xf ${DIR}/Places.tar.gz -C ${DIR}

tar -xf ${DIR}/dtd-r1.0.1.tar.gz -C ${DIR}
mv ${DIR}/dtd/images ${DIR}/Textures;
rm -r ${DIR}/dtd