#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CC = g++-4.8
MPCC = mpicxx
OPENMP = -fopenmp #Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3
LIBS =

TARGETS = serial serial_linear openmp openmp_linear autograder mpi mpi_fast mpi_aggressive 

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
serial_linear: serial_linear.o common.o
	$(CC) -o $@ $(LIBS) serial_linear.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o
#pthreads: pthreads.o common.o
#	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o
openmp: openmp.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common.o
openmp_linear: openmp_linear.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp_linear.o common.o
mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o
mpi_fast: mpi_fast.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi_fast.o common.o
mpi_aggressive: mpi_aggressive.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi_aggressive.o common.o

autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
openmp_linear.o: openmp_linear.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp_linear.cpp
serial_linear.o: serial_linear.cpp common.h
	$(CC) -c $(CFLAGS) serial_linear.cpp
serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
#pthreads.o: pthreads.cpp common.h
#	$(CC) -c $(CFLAGS) pthreads.cpp
mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
mpi_fast.o: mpi_fast.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi_fast.cpp
mpi_aggressive.o: mpi_aggressive.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi_aggressive.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
