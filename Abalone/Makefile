# Build mode
# 0: development (max safety, no optimisation)
# 1: release (min safety, optimisation)
# 2: fast and furious (no safety, optimisation)
BUILD_MODE?=1

all: main
	
# Makefile definitions
MAKEFILE_INC=../../PBMake/Makefile.inc
include $(MAKEFILE_INC)

# Name of directory of the dataset
DATASET=Abalone

# Rules to make the executable
repo=nnsquidtrainer
$($(repo)_EXENAME): \
		$($(repo)_EXENAME).o \
		$($(repo)_EXE_DEP) \
		$($(repo)_DEP)
	$(COMPILER) `echo "$($(repo)_EXE_DEP) $($(repo)_EXENAME).o" | tr ' ' '\n' | sort -u` $(LINK_ARG) $($(repo)_LINK_ARG) -o $($(repo)_EXENAME) 

$($(repo)_EXENAME).o: \
		$($(repo)_DIR)/$(DATASET)/$($(repo)_EXENAME).c \
		$($(repo)_INC_H_EXE) \
		$($(repo)_EXE_DEP)
	$(COMPILER) $(BUILD_ARG) $($(repo)_BUILD_ARG) `echo "$($(repo)_INC_DIR)" | tr ' ' '\n' | sort -u` -c $($(repo)_DIR)/$(DATASET)/$($(repo)_EXENAME).c
	
convData:
	main -convData ./Prototask.data ./dataset.json

learn:
	main -squidlets ./squidlets.json -learn /home/bayashi/GitHub/NNSquidTrainer/Abalone/dataset.json /home/bayashi/GitHub/NNSquidTrainer/Abalone/WorkingDir
