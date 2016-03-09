#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "common.h"

using namespace std;

double X_BSIZE, Y_BSIZE;
int X_BNUM, Y_BNUM; 
double world_size;

#define _cutoff 0.01
#define _density 0.0005
#define sb(i, j) send_buffer[(i+1)+(j+1)*3]
#define rb(i, j) recv_buffer[(i+1)+(j+1)*3]
#define pamt(i, j) amount+(i+1)+(j+1)*3

typedef vector<particle_t> bin_t;
MPI_Datatype PARTICLE;

inline int bin_num(particle_t &p)
{
    return (floor(p.x / X_BSIZE) + X_BNUM * floor(p.y / Y_BSIZE) );
}

inline int neighbor_rank(int rank, int i, int j)
{
    return rank + i + j * X_BNUM;
}

void build_bins(vector<bin_t>& bins, particle_t* particles, int n, int n_proc)
{
    bins.resize(n_proc);
    for(int i = 0; i < n; ++i)
        bins[bin_num(particles[i])].push_back(particles[i]);
}

int which_neighbor(particle_t &p, int rank)
{
    int bn = bin_num(p);
    
}

void partition_particles(bin_t& local_bin, vector<bin_t>& send_buffer, int rank)
{
    double left_up_x = (rank % X_BNUM) * X_BSIZE;
    double left_up_y = (rank / X_BNUM) * Y_BSIZE;
    double inner_x1 = left_up_x + _cutoff;
    double inner_x2 = left_up_x + X_BSIZE - _cutoff;
    double inner_y1 = left_up_y + _cutoff;
    double inner_y2 = left_up_y + Y_BSIZE - _cutoff;

    for( int i = 0; i < local_bin.size(); ++i)
    {
        particle_t& cp = local_bin[i];
        if(cp.x < left_up_x or cp.x > left_up_x + X_BSIZE or cp.y < left_up_y or cp.y > left_up_y + Y_BSIZE)
        {
            cout << "ERROR: particle not in bin" << endl; 
        }
        if(cp.x < inner_x1)
        {
            sb(-1, 0).push_back(cp);
            if(cp.y < inner_y1)
            {
                sb(-1, -1).push_back(cp);
                sb(0, -1).push_back(cp);
            }
            else if(cp.y > inner_y2)
            {
                sb(-1, 1).push_back(cp);
                sb(0, 1).push_back(cp);
            }
            continue;
        }
        if(cp.x > inner_x2)
        {
            sb(1, 0).push_back(cp);
            if(cp.y < inner_y1)
            {
                sb(0, -1).push_back(cp);
                sb(1, -1).push_back(cp);
            }
            else if (cp.y > inner_y2)
            {
                sb(0, 1).push_back(cp);
                sb(1, 1).push_back(cp);
            }
            continue;
        }
        if (cp.y < inner_y1)
        {
            sb(0, -1).push_back(cp);
            continue;
        }
        if (cp.y > inner_y2)
        {
            sb(0, 1).push_back(cp);
        }
    }
}

void send_boundary_particles(vector<bin_t>& send_buffer, int rank)
{
    int cbin = rank;
    int lowi = -1, highi = 1, lowj = -1, highj = 1;
    if (cbin < X_BNUM)
        lowj = 0;
    if (cbin % X_BNUM == 0)
        lowi = 0;
    if (cbin % X_BNUM == (X_BNUM-1))
        highi = 0;
    if (cbin >= X_BNUM*(Y_BNUM-1))
        highj = 0;
    // declare a handle, but we won't use it
    MPI_Request req;
    for (int i = lowi; i <= highi; i++)
    {
        for (int j = lowj; j <= highj; j++)
        {
            if (i == 0 and j == 0)
                continue;
            MPI_Isend(sb(i, j).data(), sb(i, j).size(), PARTICLE, neighbor_rank(rank, i, j), 0, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // free the handle
        }
    }
}

int receive_boundary_particles(particle_t* neighbors, int rank)
{
    int cbin = rank;
    int lowi = -1, highi = 1, lowj = -1, highj = 1;
    if (cbin < X_BNUM)
        lowj = 0;
    if (cbin % X_BNUM == 0)
        lowi = 0;
    if (cbin % X_BNUM == (X_BNUM-1))
        highi = 0;
    if (cbin >= X_BNUM*(Y_BNUM-1))
        highj = 0;

    MPI_Status status;
    int nrank;
    int total_amount = 0, amount[9];
    particle_t* current;
    for (int i = lowi; i <= highi; i++)
    {
        for (int j = lowj; j <= highj; j++)
        {
            if (i == 0 and j == 0)
                continue;
            nrank = neighbor_rank(rank, i, j);
            MPI_Probe(nrank, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, pamt(i, j));
            total_amount += *pamt(i, j);
        }
    }

    neighbors = (particle_t*)malloc(total_amount * sizeof(particle_t)); 
    current = neighbors;

    for (int i = lowi; i <= highi; i++)
    {
        for (int j = lowj; j <= highj; j++)
        {
            if (i == 0 and j == 0)
                continue;
            nrank = neighbor_rank(rank, i, j);
            MPI_Recv(current, *pamt(i, j), PARTICLE, nrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            current += *pamt(i, j);
        }
    }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
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
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    const int ndims = 2;
    int dims[ndims] = {0, 0};
    MPI_Dims_create(n_proc, ndims, dims);
    X_BNUM = dims[0];
    Y_BNUM = dims[1];
    world_size = sqrt( _density * n ); // used for this source
    X_BSIZE = world_size / (double) X_BNUM;
    Y_BSIZE = world_size / (double) Y_BNUM;
    assert(X_BNUM * Y_BNUM == n_proc);
    
    //
    //  initialize and distribute the particles 
    //
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    set_size(n);
    if(rank == 0)
        init_particles(n, particles);

    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    vector<bin_t> bins;
    build_bins(bins, particles, n, n_proc);
    delete[] particles;
    particles = NULL;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        //if( find_option( argc, argv, "-no" ) == -1 )
        //  if( fsave && (step%SAVEFREQ) == 0 )
        //    save( fsave, n, particles );

        vector<bin_t> send_buffer(9); // only send to valid neighbors
        particle_t* neighbors;

        partition_particles(bins[rank], send_buffer, rank);
        send_boundary_particles(send_buffer, rank);
        int total_recv_amount = receive_boundary_particles(neighbors, rank);

        // compute local forces
        int nlocals = bins[rank].size();
        for( int i = 0; i < nlocals; ++i)
        {
            bins[rank][i].ax = bins[rank][i].ay = 0;
            for( int j = 0; j < nlocals; ++j)
            {
                apply_force(bins[rank][i], bins[rank][j], &dmin, &davg, &navg);
            }
        }

        // compute local-neighbor forces
        for (int i = 0; i < nlocals; ++i)
        {
            for (int j = 0; j < total_recv_amount; ++j)
            {
                apply_force(bins[rank][i], neighbors[j], &dmin, &davg, &navg);
            }
        }

        //
        //  move particles
        //
        vector<bin_t> remote_move(9);
        int tail = bins[rank].size(), k = 0;
        for (; k < tail; )
        {
            move(bins[rank][k])
            if(bin_num(bins[rank][k]) == rank)
            {
                ++k;
            } 
            else
            {
                remote_move[].push_back(bins[rank][k]);
                bins[rank][k] = bins[rank][--tail];
            }
        }
        bins[rank].resize(k);


     
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    //TODO need to free things here
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
