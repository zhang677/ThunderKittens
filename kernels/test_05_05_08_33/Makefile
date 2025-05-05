# Compiler
NVCC?=nvcc
GPU?=L40S
TARGET=test_05_05_11_02
SRC=${TARGET}.cu

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -I/usr/include/python3.11 -I/usr/local/lib/python3.11/dist-packages/pybind11/include -L/usr/lib/python3.11/config-3.11-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lcrypt -ldl -lm -lm -shared -fPIC -lpython3.11

ORI_NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIC --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
ORI_NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -I/usr/include/python3.12 -I/usr/local/lib/python3.12/dist-packages/pybind11/include -L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -L/usr/local/lib -lcrypt -ldl -lm -lm -shared -fPIC

LAM_NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIC --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
LAM_NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -I/usr/include/python3.10 -I/home/ubuntu/.local/lib/python3.10/site-packages/pybind11/include -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lcrypt -ldl -lm -lm -shared -fPIC

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
ORI_NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),L40S)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
ORI_NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
ORI_NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
LAM_NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
LAM_NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
endif

# Default target
all: all_target

all_target: $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET).cpython-311-x86_64-linux-gnu.so

# Clean target
clean:
	rm -f $(TARGET)*.so

# ec2 target
ec2: ec2_target

ec2_target: $(SRC)
	$(NVCC) $(SRC) $(ORI_NVCCFLAGS) -o $(TARGET).cpython-312-x86_64-linux-gnu.so

# lam target
lam: lam_target
lam_target: $(SRC)
	$(NVCC) $(SRC) $(LAM_NVCCFLAGS) -o $(TARGET).cpython-310-x86_64-linux-gnu.so
