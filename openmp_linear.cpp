#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "omp.h"
using namespace std;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

// calculate particle's bin number
int binNum(particle_t &p, int bpr)
{
  return ( floor(p.x/cutoff) + bpr*floor(p.y/cutoff) );
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    int navg,nabsavg=0,numthreads;
    double dmin, davg, absmin= 1.0,absavg =0.0;

    if( find_option( argc, argv, "-h" ) >=0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    // create spatial bins (of size cutoff by cutoff)
    double size = sqrt( density*n );
    int bpr = ceil(size/cutoff);
    int numbins = bpr*bpr;
    vector<particle_t*> *bins = new vector<particle_t*>[numbins];

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    // make dmin private to avoid global synchronization
    #pragma omp parallel private(dmin)
    {
        numthreads = omp_get_num_threads();
        // time dependency, can't parallelize this loop
        for( int step = 0; step < NSTEPS; step++ )
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // clear bins at each time step
            // bins are shared among threads
            // no need for synchronization, since every thread access different entry
            // TODO is there any cache line problems?
            #pragma omp master
            for (int m = 0; m < numbins; m++)
              bins[m].clear();

            // place particles in bins
            // push_back is not thread safe
            #pragma omp master
            for (int i = 0; i < n; i++)
              bins[binNum(particles[i],bpr)].push_back(particles + i);

            // this is extremely important line!
            #pragma omp barrier
            //
            //  compute forces
            //
            #pragma omp for reduction (+:navg) reduction(+:davg)
            for( int p = 0; p < n; p++ )
            {
                particles[p].ax = particles[p].ay = 0;

                // find current particle's bin, handle boundaries
                // cbin, lowi, highi, lowj and highj are private by default
                int cbin = binNum( particles[p], bpr );
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
                for (int i = lowi; i <= highi; i++)
                {
                    for (int j = lowj; j <= highj; j++)
                    {
                      int nbin = cbin + i + bpr*j;
                      for (int k = 0; k < bins[nbin].size(); k++ )
                        apply_force( particles[p], *bins[nbin][k], &dmin, &davg, &navg);
                    }
                }
            }

            //
            //  move particles
            //
            #pragma omp for
            for( int p = 0; p < n; p++ )
                move( particles[p] );

            if( find_option( argc, argv, "-no" ) == -1 )
            {
                //
                //  computing statistical data
                //
                #pragma omp master
                if (navg) {
                  absavg += davg/navg;
                  nabsavg++;
                }

                #pragma omp critical
                if (dmin < absmin) 
                    absmin = dmin;

                //
                //  save if necessary
                //
                #pragma omp master
                if( fsave && (step%SAVEFREQ) == 0 )
                    save( fsave, n, particles );
            }
        }
    }
    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
        if (nabsavg) absavg /= nabsavg;
    //
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
      fclose( fsum );
    free( particles );
    delete [] bins;
    if( fsave )
      fclose( fsave );

    return 0;
}
