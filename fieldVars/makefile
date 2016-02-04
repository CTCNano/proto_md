CC = g++
RM = rm
LINKER = ar
CFlags = -Wall -g -std=c++0x -fPIC -O3
LFlags = -shared -Wall -O3
INCL1 = -I${HOME}/.local/include
INCL2 = -I/N/soft/rhel6/python/2.7.3/include/python2.7/
INCL3 = -I/usr/lib/openmpi/include/
LIBS1 = -L${PETSC_DIR}/${PETSC_ARCH}/lib
LIBS2 = -L${HOME}/.local/lib
LIB_BR = -L/opt/cray/petsc/3.3.04/real/GNU/46/interlagos/lib
INCL_BR1 = -I/N/u/ortgrp/BigRed2/.local/include/python2.7/
INCL_BR2 = -I/opt/cray/petsc/3.3.04/real/GNU/46/include/
INCL_BR3 = -I/opt/cray/petsc/3.3.04/real/GNU/46/interlagos/include/
INCL_BR4 = -I/N/u/ortgrp/BigRed2/.local/lib/python2.7/site-packages/numpy/core/include/
MPLIB = -L/N/soft/rhel6/openmpi/intel/shared/1.6.3/lib/

all: Wrapper.o Interface.o GhostGrid.o CoarseScale.o FineScale.o
	${CC} ${LFlags} -o _FieldVars.so Interface.o Wrapper.o GhostGrid.o CoarseScale.o FineScale.o -lpetsc ${LIB1S} ${LIBS2}

Interface.o: Interface.cpp
	${CC} ${CFlags} -c Interface.cpp -o Interface.o -I/usr/include/python2.7 ${INCL1} ${INCL2} ${INCL3}

Wrapper.o: Wrapper.cpp
	${CC} ${CFlags} -c Wrapper.cpp -o Wrapper.o -I/usr/include/python2.7 ${INCL1} ${INCL2} ${INCL3}

GhostGrid.o: GhostGrid.cpp
	${CC} ${CFlags} -c GhostGrid.cpp -o GhostGrid.o -I/usr/include/python2.7 ${INCL1} ${INCL2} ${INCL3}

CoarseScale.o: CoarseScale.cpp
	${CC} ${CFlags} -c CoarseScale.cpp -o CoarseScale.o -I/usr/include/python2.7 ${INCL1} ${INCL2} ${INCL3}

FineScale.o: FineScale.cpp
	${CC} ${CFlags} -c FineScale.cpp -o FineScale.o -I/usr/include/python2.7 ${INCL1} ${INCL2} ${INCL3}

clean:
	rm *.o
