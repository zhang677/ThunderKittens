# Compiler
NVCC?=nvcc
GPU=A100

TARGET=test_04_15_09_58
SRC=test_04_15_09_58.cu

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -I/usr/include/python3.11 -I/usr/local/lib/python3.11/dist-packages/pybind11/include -L/usr/lib/python3.11/config-3.11-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lcrypt -ldl -lm -lm -shared -fPIC -lpython3.11

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET).cpython-311-x86_64-linux-gnu.so

# Clean target
clean:
	rm -f $(TARGET)