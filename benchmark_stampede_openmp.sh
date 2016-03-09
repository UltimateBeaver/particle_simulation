export OMP_NUM_THREADS=8
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 316 -no
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 1000 -no
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 3160 -no
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 10000 -no
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 31600 -no
srun -p development -n 1 -c 8 -t 1:00:00 ./openmp_linear -n 100000 -no
