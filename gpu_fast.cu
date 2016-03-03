#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <vector>
#include "common.h"
using std::vector;

#define NUM_THREADS 256

extern double size;

// calculate particle's bin number
int binNum(particle_t &p, int bpr)
{
    return ( floor(p.x/cutoff) + bpr*floor(p.y/cutoff) );
}

inline bool double_near(double x, double y) {
  const double eps = 1e-5;
  return (x - y < eps) && (y - x < eps);
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

__global__ void assign_particle_to_bin(int n, particle_t* d_particles,
    int bpr, int maxnum_per_bin, int* bin_count, particle_t** bin_content) {
  CUDA_KERNEL_LOOP (i, n) {
    particle_t* p = d_particles + i;
    int bin_idx = floor(p->x/cutoff) + bpr*floor(p->y/cutoff);

    // Put particle in bin_idx
    // Compute the position of particle inside the bin with AtomicAdd
    // to avoid race.
    int list_idx = atomicAdd(&bin_count[bin_idx], 1);
    bin_content[bin_idx*maxnum_per_bin + list_idx] = p;
  }
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(int n, particle_t* d_particles,
    int bpr, int maxnum_per_bin, int* bin_count, particle_t** bin_content) {
  CUDA_KERNEL_LOOP (i, n) {
    particle_t* p = &d_particles[i];
    int bin_idx = floor(p->x/cutoff) + bpr*floor(p->y/cutoff);

    int lowi = -1, highi = 1, lowj = -1, highj = 1;
    if (bin_idx < bpr)
      lowj = 0;
    if (bin_idx % bpr == 0)
      lowi = 0;
    if (bin_idx % bpr == (bpr-1))
      highi = 0;
    if (bin_idx >= bpr*(bpr-1))
      highj = 0;

    // apply nearby forces
    for (int i = lowi; i <= highi; i++)
      for (int j = lowj; j <= highj; j++)
      {
        int nbin = bin_idx + i + bpr*j;
        for (int k = 0; k < bin_count[nbin]; k++ )
          apply_force_gpu(*p, *bin_content[nbin*maxnum_per_bin+k]);
      }
  }
}

__global__ void move_gpu (particle_t * d_particles, int n, double size)
{
  CUDA_KERNEL_LOOP(i, n) {
    particle_t * p = &d_particles[i];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
  }
}

//  benchmarking program
//
int main( int argc, char **argv )
{
   // This takes a few seconds to initialize the runtime
   cudaDeviceSynchronize();

   if( find_option( argc, argv, "-h" ) >=0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    bool cpu_check = find_option( argc, argv, "-c" ) >=0;

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    // create spatial bins (of size cutoff by cutoff)
    int bpr = ceil(sqrt( density*n )/cutoff);
    int numbins = bpr*bpr;
    // the maximum possible numbers of particles inside a bin
    int maxnum_per_bin = (cutoff*2/min_r) * (cutoff*2/min_r);
    maxnum_per_bin = 200;

    // Bins for particles
    // bins will be a (bpr, bpr, maxnum_per_bin) array
    int * bin_count;
    particle_t ** bin_content;
    cudaMalloc((void **) &bin_count, numbins * sizeof(int));
    cudaMalloc((void **) &bin_content, numbins * maxnum_per_bin * sizeof(particle_t*));
    printf("size: %lu\n", numbins * maxnum_per_bin * sizeof(particle_t*));

    // CPU check buffer
    vector<particle_t*> *bins = 0;
    particle_t *check_particles = 0;
    if (cpu_check) {
      printf("turning on CPU check\n");
      bins = new vector<particle_t*>[numbins];
      particle_t *check_particles = (particle_t*) malloc( n * sizeof(particle_t) );
      memcpy(check_particles, particles, n * sizeof(particle_t));
    }

    cudaDeviceSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaDeviceSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
      // clear bins at each time step
      // Mark all bins as "no particle"
      cudaMemset(bin_count, 0, numbins*sizeof(int));


      // place particles in bins
      int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
      assign_particle_to_bin<<<blks, NUM_THREADS>>>(n, d_particles, bpr,
          maxnum_per_bin, bin_count, bin_content);

      //
      //  compute forces
      //
      compute_forces_gpu<<<blks, NUM_THREADS>>>(n, d_particles, bpr,
          maxnum_per_bin, bin_count, bin_content);

      if (cpu_check) {
        // clear bins at each time step
        for (int m = 0; m < numbins; m++)
          bins[m].clear();

        // place particles in bins
        for (int i = 0; i < n; i++)
          bins[binNum(check_particles[i],bpr)].push_back(check_particles + i);

        //
        //  compute forces
        //
        for( int p = 0; p < n; p++ )
        {
          check_particles[p].ax = check_particles[p].ay = 0;

          // find current particle's bin, handle boundaries
          int cbin = binNum( check_particles[p], bpr );
          int lowi = -1, highi = 1, lowj = -1, highj = 1;
          if (cbin < bpr)
            lowj = 0;
          if (cbin % bpr == 0)
            lowi = 0;
          if (cbin % bpr == (bpr-1))
            highi = 0;
          if (cbin >= bpr*(bpr-1))
            highj = 0;

          // apply nearby forces
          for (int i = lowi; i <= highi; i++) {
            for (int j = lowj; j <= highj; j++)
            {
              int nbin = cbin + i + bpr*j;
              for (int k = 0; k < bins[nbin].size(); k++ )
                apply_force( check_particles[p], *bins[nbin][k]);
            }
          }
        }

        // Copy the particles back to the CPU
        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);

        for( int p = 0; p < n; p++ ) {
          printf("checking particle %d\n", p);
          particle_t& a = particles[p];
          particle_t& b = check_particles[p];
          if (!double_near(a.x, b.x))
            printf("\tx failed: %f (compute) vs. %f (ref)\n", a.x, b.x);
          if (!double_near(a.y, b.y))
            printf("\ty failed: %f (compute) vs. %f (ref)\n", a.y, b.y);
          if (!double_near(a.vx, b.vx))
            printf("\tvx failed: %f (compute) vs. %f (ref)\n", a.vx, b.vx);
          if (!double_near(a.vy, b.vy))
            printf("\tvy failed: %f (compute) vs. %f (ref)\n", a.vy, b.vy);
          if (!double_near(a.ax, b.ax))
            printf("\tax failed: %f (compute) vs. %f (ref)\n", a.ax, b.ax);
          if (!double_near(a.ay, b.ay))
            printf("\tay failed: %f (compute) vs. %f (ref)\n", a.ay, b.ay);
          abort();
        }
      }

      //
      //  move particles
      //
      move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

      //
      //  save if necessary
      //
      if( fsave && (step%SAVEFREQ) == 0 ) {
        // Copy the particles back to the CPU
        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
        save( fsave, n, particles);
      }
    }
    cudaDeviceSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    cudaFree(bin_count);
    cudaFree(bin_content);
    if( fsave )
        fclose( fsave );

    return 0;
}
