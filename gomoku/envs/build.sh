rm -rf build
mkdir build
cd build
cmake ..
make 
cd ..
# cp tensorflow_gpu/lib/* build
# ./build/test
python src/test.py