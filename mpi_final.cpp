#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include "common.h"

using namespace std;

#define _cutoff 0.01    //Value copied from common.cpp
#define _density 0.0005

typedef vector<particle_t> bin_t;

double world_size, bin_size;
int bin_count;

inline int bin_num(particle_t &p)
{
    return (floor(p.x / bin_size) + bin_count * floor(p.y / bin_size) );
}

inline void bin_particle(vector<bin_t>& bins, particle_t& p)
{
    bins[bin_num(p)].push_back(p);
}

void build_bins(vector<bin_t>& bins, particle_t* particles, int n)
{
    for (int i = 0; i < n; i++)
        bins[bin_num(particles[i])].push_back(particles[i]);
}

inline void clear_all_bins_in_row(int row, vector<bin_t>& bins)
{
    for (int j = 0; j < bin_count; ++j) {
        bin_t& bin = bins[row * bin_count + j];
        bin.clear();
    }
}

inline bin_t partial_flatten(vector<bin_t>& bins, int start, int end)
{
    bin_t result;
    for (int i = start; i < end; ++i)
        result.insert(result.end(), bins[i].begin(), bins[i].end());
    return result;
}

inline bin_t flatten(vector<bin_t>& bins)
{
    return partial_flatten(bins, 0, bins.size());
}

inline void copy_and_clear_all_bins_in_row(int row, vector<bin_t>& bins, bin_t& remote_move)
{
    // add bins in boundary row to remote_move and clear them afterwards
    for (int j = 0; j < bin_count; ++j) {
        bin_t& bin = bins[row * bin_count + j];
        remote_move.insert(remote_move.end(), bin.begin(), bin.end());
        bin.clear();
    }
}

void apply_forces_bin(vector<bin_t>& bins, int i, int j, double& dmin, double& davg, int& navg)
{
    int cbin = i * bin_count + j;
    bin_t& cvec = bins[cbin];
    int lowi = -1, highi = 1, lowj = -1, highj = 1;
    if (cbin < bin_count)
        lowj = 0;
    if (cbin % bin_count == 0)
        lowi = 0;
    if (cbin % bin_count == (bin_count - 1))
        highi = 0;
    if (cbin >= bin_count * (bin_count - 1))
        highj = 0;

    for (int k = 0; k < cvec.size(); ++k)
        cvec[k].ax = cvec[k].ay = 0;

    for (int ii = lowi; ii <= highi; ii++)
    {
        for (int jj = lowj; jj <= highj; jj++)
        {
            int nbin = (i + jj) * bin_count + j + ii;
            bin_t& nvec = bins[nbin];
            for (int k = 0; k < cvec.size(); ++k)
                for (int l = 0; l < nvec.size(); ++l)
                    apply_force(cvec[k], nvec[l], &dmin, &davg, &navg);
        }
    }
}

void move_particles(bin_t& up_remote_move, bin_t& down_remote_move, vector<bin_t>& bins, int row_start, int row_end)
{
    bin_t local_move;
    // for each row belongs to me
    for (int i = row_start; i < row_end; ++i) 
    {
        // for each column
        for (int j = 0; j < bin_count; ++j) 
        {
            bin_t& bin = bins[i * bin_count + j];
            int k = 0;
            int tail = bin.size();
            for (; k < tail; ) 
            {
                move(bin[k]);
                int col_bin = int(bin[k].x / bin_size);
                int row_bin = int(bin[k].y / bin_size);
                // if still belongs to me
                if (row_start <= row_bin and row_bin < row_end) 
                {
                    // if still belongs to original bin
                    if (col_bin == j and row_bin == i)
                        ++k;
                    else 
                    {
                        // if needs to move locally
                        local_move.push_back(bin[k]);
                        bin[k] = bin[--tail];
                    }
                } 
                else if (row_bin < row_start) 
                {
                    // else need to move to other processors
                    up_remote_move.push_back(bin[k]);
                    bin[k] = bin[--tail];
                }
                else if (row_bin >= row_end)
                {
                    down_remote_move.push_back(bin[k]);
                    bin[k] = bin[--tail];
                }
                else
                    cout << "ERROR: particle out of range" << endl;
            }
            bin.resize(k);
        }
    }
    // re-bin particles in local_move
    for (int i = 0; i < local_move.size(); ++i) 
        bin_particle(bins, local_move[i]);
}
//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    //signal(SIGSEGV, sigsegv);

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
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    particle_t *particles = new particle_t[n];
    set_size( n );
    if(rank == 0)
        init_particles( n, particles );

    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    vector<bin_t> bins;
    world_size = sqrt(n * _density);
    bin_size = _cutoff;  
    bin_count = int(world_size / bin_size) + 1; 
    bins.resize(bin_count * bin_count);
    build_bins(bins, particles, n);
    delete[] particles;
    particles = NULL;

    // each processor owns same number of rows
    int rows_per_proc = bin_count / n_proc;
    int row_start = rows_per_proc * rank;
    int row_end = rows_per_proc * (rank + 1);
    if (rank == n_proc - 1)
        row_end = bin_count;
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        for (int i = row_start; i < row_end; ++i) {
            for (int j = 0; j < bin_count; ++j) {
                apply_forces_bin(bins, i, j, dmin, davg, navg);
            }
        }

        // move particles and extract pariticles need to be moved to other processors 
        bin_t up_remote_move, down_remote_move;
        move_particles(up_remote_move, down_remote_move, bins, row_start, row_end);

        // clear outdated neighbor information
        if (rank != 0 and rank != n_proc - 1) 
        {
            clear_all_bins_in_row(row_start - 1, bins);
            clear_all_bins_in_row(row_end, bins);
        }
        // in case there's only one processor
        if (rank == 0 and rank != n_proc - 1)
            clear_all_bins_in_row(row_end, bins);
        if (rank == n_proc - 1 and rank != 0) 
           clear_all_bins_in_row(row_start - 1, bins);

        for (int i = 0; i < up_remote_move.size(); ++i)
            bin_particle(bins, up_remote_move[i]);
        for (int i = 0; i < down_remote_move.size(); ++i)
            bin_particle(bins, down_remote_move[i]);

        MPI_Request up_req, down_req;
        bin_t up_data, down_data;
        if (rank != 0)
        {
            int up_data_space = 0;
            for (int i = row_start * bin_count; i < (row_start + 1) * bin_count; ++i)
                up_data_space += bins[i].size();
            up_data_space += up_remote_move.size();
            up_data.reserve(up_data_space);
            for (int i = row_start * bin_count; i < (row_start + 1) * bin_count; ++i)
                up_data.insert(up_data.end(), bins[i].begin(), bins[i].end());
            up_data.insert(up_data.end(), up_remote_move.begin(), up_remote_move.end());
            MPI_Isend(up_data.data(), up_data_space, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, &up_req);
        }
        if (rank != n_proc - 1)
        {
            int down_data_space = 0;
            for (int i = (row_end - 1) * bin_count; i < row_end * bin_count; ++i)
                down_data_space += bins[i].size();
            down_data_space += down_remote_move.size();
            down_data.reserve(down_data_space);
            for (int i = (row_end - 1) * bin_count; i < row_end * bin_count; ++i)
                down_data.insert(down_data.end(), bins[i].begin(), bins[i].end());
            down_data.insert(down_data.end(), down_remote_move.begin(), down_remote_move.end());
            MPI_Isend(down_data.data(), down_data_space, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, &down_req);
        }

        MPI_Status status1, status2;
        int up_amount, down_amount;
        bin_t up_recv_data, down_recv_data;
        if (rank != 0)
        {
            MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status1);
            MPI_Get_count(&status1, PARTICLE, &up_amount);
            up_recv_data.resize(up_amount);
            MPI_Recv(up_recv_data.data(), up_amount, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < up_recv_data.size(); ++i)
                bin_particle(bins, up_recv_data[i]);
        }
        if (rank != n_proc - 1)
        {
            MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status2);
            MPI_Get_count(&status2, PARTICLE, &down_amount);
            down_recv_data.resize(down_amount);
            MPI_Recv(down_recv_data.data(), down_amount, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < down_recv_data.size(); ++i)
                bin_particle(bins, down_recv_data[i]);
        }

        MPI_Barrier( MPI_COMM_WORLD );

        if (find_option( argc, argv, "-no" ) == -1) {
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
          if (rank == 0){
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) 
                absmin = rdmin;
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
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}