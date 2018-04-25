FILE="edges2shoes"
mkdir data
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./data/$FILE.tar.gz
TARGET_DIR=./data/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./data/
rm $TAR_FILE
