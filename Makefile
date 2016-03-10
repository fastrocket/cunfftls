################################################################################
#
# Build script for project
#
################################################################################
ARCH		:= 52

# Add source files here
NAME            := cunfftls
################################################################################
# Rules and targets
NVCC=nvcc
CC=g++

CUNA_DIR=../cunfft_adjoint
CUDA_VERSION=7.5
BLOCK_SIZE=256
VERSION=1.0

SRCDIR=.
HEADERDIR=.
BUILDDIR=.
LIBDIR=.
BINDIR=.

OPTIMIZE_CPU= -O3
OPTIMIZE_GPU= -Xcompiler -O3 --use_fast_math
DEFS := -DBLOCK_SIZE=$(BLOCK_SIZE) -DVERSION=\"$(VERSION)\"
NVCCFLAGS := $(DEFS) $(OPTIMIZE_GPU) -Xcompiler -fpic --gpu-architecture=compute_$(ARCH) --gpu-code=sm_$(ARCH),compute_$(ARCH) 
CFLAGS := $(DEFS) -fPIC -Wall $(OPTIMIZE_CPU)

CUDA_LIBS =`pkg-config --libs cudart-$(CUDA_VERSION)` 

CUDA_INCLUDE =`pkg-config --cflags cudart-$(CUDA_VERSION)`

LIBS := -L$(LIBDIR) -L$(CUNA_DIR)/lib $(CUDA_LIBS) -lm

###############################################################################

CPP_FILES := $(filter-out main.cpp,$(notdir $(wildcard $(SRCDIR)/*.cpp)))
CU_FILES  := $(notdir $(wildcard $(SRCDIR)/*.cu))

CPP_OBJ_FILES_SINGLE :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%f.o)
CPP_OBJ_FILES_DOUBLE :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%d.o)

CU_OBJ_FILES_SINGLE := $(CU_FILES:%.cu=$(BUILDDIR)/%f.o)
CU_OBJ_FILES_DOUBLE := $(CU_FILES:%.cu=$(BUILDDIR)/%d.o)



INCLUDE := $(CUDA_INCLUDE) -I$(HEADERDIR) -I$(CUNA_DIR)/inc

all : single double test-single test-double

single : lib$(NAME)f.so
double : lib$(NAME)d.so


%f.so : $(CU_OBJ_FILES_SINGLE) $(CPP_OBJ_FILES_SINGLE) $(BUILDDIR)/dlink-single.o
	$(CC) -shared -o $(LIBDIR)/$@ $^ $(LIBS) -lcunaf	

%d.so : $(CU_OBJ_FILES_DOUBLE) $(CPP_OBJ_FILES_DOUBLE) $(BUILDDIR)/dlink-double.o
	$(CC) -shared -o $(LIBDIR)/$@ $^ $(LIBS) -lcunad

test-single :
	$(CC) $(CFLAGS) $(INCLUDE) -o $(BINDIR)/$@ main.cpp $(LIBS) -l$(NAME)f -lcunaf

test-double : 
	$(CC) $(CFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -o $(BINDIR)/$@ main.cpp $(LIBS) -l$(NAME)d -lcunad

%-single.o : $(CU_OBJ_FILES_SINGLE)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $^ -o $@

%-double.o : $(CU_OBJ_FILES_DOUBLE)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -dlink $^ -o $@

$(CU_OBJ_FILES_SINGLE) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -c $(SRCDIR)/$(notdir $(subst f.o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CU_OBJ_FILES_DOUBLE) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -rdc=true -c $(SRCDIR)/$(notdir $(subst d.o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CPP_OBJ_FILES_SINGLE) : 
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRCDIR)/$(notdir $(subst f.o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CPP_OBJ_FILES_DOUBLE) : 
	$(CC) $(CFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -c $(SRCDIR)/$(notdir $(subst d.o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)



.PHONY : clean
RM=rm -f


clean : 
	$(RM) *.dat *.png *.so *.o 

print-%  : ; @echo $* = $($*)
