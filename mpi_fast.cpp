#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <signal.h>
#include <unistd.h>
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

void move_particles(bin_t& remote_move, vector<bin_t>& bins, int row_start, int row_end)
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
                int x = int(bin[k].x / bin_size);
                int y = int(bin[k].y / bin_size);
                // if still belongs to me
                if (row_start <= y && y < row_end) 
                {
                    // if still belongs to original bin
                    if (x == j && y == i)
                        ++k;
                    else 
                    {
                        // if needs to move locally
                        local_move.push_back(bin[k]);
                        bin[k] = bin[--tail];
                    }
                } 
                else 
                {
                    // else need to move to other processors
                    remote_move.push_back(bin[k]);
                    bin[k] = bin[--tail];
                }
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

        // move particles and extract pariticles need to be moved to other processors 
        bin_t remote_move;
        move_particles(remote_move, bins, row_start, row_end);

        if (rank != 0 and rank != n_proc - 1) 
        {
            clear_all_bins_in_row(row_start - 1, bins);
            clear_all_bins_in_row(row_end, bins);
            copy_and_clear_all_bins_in_row(row_start, bins, remote_move);
            copy_and_clear_all_bins_in_row(row_end-1, bins, remote_move);
        }

        if (rank == 0)
        {
            clear_all_bins_in_row(row_end, bins);
            copy_and_clear_all_bins_in_row(row_end-1, bins, remote_move);
        }
        if (rank == n_proc - 1) 
        {
           clear_all_bins_in_row(row_start - 1, bins);
           copy_and_clear_all_bins_in_row(row_start, bins, remote_move); 
        }

        // root gathers send_count from all processes into recv_counts
        int send_count = remote_move.size();
        int recv_counts[n_proc];
        if (rank == 0)
            MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        else
            MPI_Gather(&send_count, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // root calculates offset for MPI_Gatherv
        int offsets[n_proc];
        int total_num = 0;
        bin_t incoming_move;
        if (rank == 0) {
            offsets[0] = 0;
            for (int i = 1; i < n_proc; ++i) {
                offsets[i] = offsets[i-1] + recv_counts[i-1];
            }
            total_num = offsets[n_proc-1] + recv_counts[n_proc-1];
            incoming_move.resize(total_num);
        }
        MPI_Gatherv(remote_move.data(), send_count, PARTICLE, 
                    incoming_move.data(), recv_counts, offsets, PARTICLE, 0, MPI_COMM_WORLD);

        vector<bin_t> scatter_particles;
        if (rank == 0) {
            // root process all particles in incoming_move and decide which processors to send to 
            scatter_particles.resize(n_proc);
            for (int i = 0; i < incoming_move.size(); ++i) 
            {
                // add this particle to the processor it belongs to 
                int y_bin = floor(incoming_move[i].y / bin_size);
                int newrank = min(y_bin / rows_per_proc, n_proc-1);
                scatter_particles[newrank].push_back(incoming_move[i]);

                // add this particle to other processors if it is on boundary
                int row = y_bin % rows_per_proc;
                if (row == 0 and newrank != 0)
                    scatter_particles[newrank - 1].push_back(incoming_move[i]);
                if (row == rows_per_proc-1 and newrank != n_proc-1)
                    scatter_particles[newrank + 1].push_back(incoming_move[i]);
            }
            for (int i = 0; i < n_proc; ++i) {
                recv_counts[i] = scatter_particles[i].size();
            }
            offsets[0] = 0;
            for (int i = 1; i < n_proc; ++i) {
                offsets[i] = offsets[i-1] + recv_counts[i-1];
            }
        }

        send_count = 0;
        MPI_Scatter(recv_counts, 1, MPI_INT, &send_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        bin_t outgoing_move;
        outgoing_move.resize(send_count);

        if (rank == 0)
        {
            bin_t scatter_flatten;
            for (int i = 0; i < scatter_particles.size(); ++i) 
                scatter_flatten.insert(scatter_flatten.end(), scatter_particles[i].begin(), scatter_particles[i].end());
            MPI_Scatterv(scatter_flatten.data(), recv_counts, offsets, PARTICLE, outgoing_move.data(), send_count, PARTICLE, 0, MPI_COMM_WORLD);
        }
        else
            MPI_Scatterv(NULL, recv_counts, offsets, PARTICLE, outgoing_move.data(), send_count, PARTICLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < send_count; ++i) {
            bin_particle(bins, outgoing_move[i]);
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