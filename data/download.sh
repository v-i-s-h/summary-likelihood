PREV_DIR=`pwd`
cd "${0%/*}"


# MNIST-C
echo "Downloading MNIST-C dataset"
wget -O mnist_c.zip https://zenodo.org/record/3239543/files/mnist_c.zip?download=1
unzip mnist_c.zip
rm mnist_c.zip

# CIFAR10-C
echo "Downloading CIFAR10-C dataset"
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xf CIFAR-10-C.tar
rm CIFAR-10-C.tar

# # ChestMNIST
# ### ChestMNIST
# echo "Downloading ChestMNIST"
# curl -o chestmnist.npz https://zenodo.org/record/5208230/files/chestmnist.npz

# # BloodMNIST
# echo "Downloading BloodMNIST"
# curl -o bloodmnist.npz https://zenodo.org/record/5208230/files/bloodmnist.npz

# # PathMNIST
# echo "Downloading PathMNIST"
# curl -o pathmnist.npz https://zenodo.org/record/5208230/files/pathmnist.npz

# # TissueMNIST
# echo "Downloading TissueMNIST"
# curl -o tissuemnist.npz https://zenodo.org/record/5208230/files/tissuemnist.npz

cd $PREV_DIR
