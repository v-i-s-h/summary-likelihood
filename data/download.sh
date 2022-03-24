PREV_DIR=`pwd`
cd "${0%/*}"


# MNIST-C
echo "Downloading MNIST-C dataset"
wget -O mnist_c.zip https://zenodo.org/record/3239543/files/mnist_c.zip?download=1
unzip mnist_c.zip
rm mnist_c.zip

cd $PREV_DIR