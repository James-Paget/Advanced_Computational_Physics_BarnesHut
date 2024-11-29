import os
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]     = "1"

import numpy as np;
cimport numpy as np;

cimport cython;
from mpi4py import MPI;
import time;
cimport libc.math as c_math;

cdef double G = 6.67*c_math.pow(10,-11);        #Global usage

#--------------------------------------------------------#
# General Gravity Calculation & Array Separation For MPI #
#--------------------------------------------------------#
cpdef void calculate_gravitationalForce(double separation, double p1_x, double p1_y, double p1_z, double p2_m, double p2_x, double p2_y, double p2_z, double[:] force) noexcept nogil:
    cdef double force_mag;
    with cython.boundscheck(False):
        if( (p2_x -p1_x != 0.0) and (p2_y -p1_y != 0.0) and (p2_z -p1_z != 0.0) ):
            force_mag = G*p2_m/c_math.pow(separation, 3);
            force[0] += force_mag*(p2_x -p1_x);
            force[1] += force_mag*(p2_y -p1_y);
            force[2] += force_mag*(p2_z -p1_z);
        #If 0 distance, dont sum anything to the force, likely interaction with self

def calculate_split_interaction_groups(double[:] interactions, int particle_total, int ranks, int particles_per_rank):
    """
    . Takes a set of interactions, splits them into N groups of M interactions, where N=ranks and M=particles_per_rank
    . Each particle of each interaction consists of 7 elements [m, x,y,z, u,v,w]
    . Note; this method requires you already generate the split_interactions list of fixed size beforehand, and so have to have found particles_per_rank, etc already
    """
    cdef int rank_index, start_index, end_index;
    split_interactions = np.zeros((ranks-1, 7*particles_per_rank), dtype=np.double);
    with cython.boundscheck(False):
        for rank_index in range(ranks-1):   #-1 as the MASTER does not get given a set
            #Set 0th particle as target
            split_interactions[rank_index][0:7] = interactions[0:7];
            start_index = 7*(particles_per_rank-1)*rank_index +7;   #+7 to account for not including 0th (target) particle again
            #If you are are still expecting a full set (not the last rank)
            if(rank_index == ranks-2):                                  #When looking at the last WORKER's set (Will get a full OR reduced set)
                end_index   = interactions.shape[0]-start_index +7;     #After this index, the last rank will have zeros fill its remaining interactions
                split_interactions[rank_index][7:end_index] = interactions[start_index:interactions.shape[0]];
            else:                                                       #When looking at intermediate WORKER (will get a full set)
                split_interactions[rank_index][7:7*particles_per_rank] = interactions[start_index:start_index+7*(particles_per_rank-1)];
        return split_interactions;




#----------------------------#
# Euler Method Time Stepping # -> Performed using MPI parallelism
#----------------------------#
def updateParticle_Euler_MPI(int rank, int MASTER, comm, int ranks, double timestep, double[:] interactions):
    """
    . Calculates what the position and velocity of the particle should be when stepped according to Euler's equation when interacting with a given set of particles
    . This calculation is parallelised with MPI

    . ranks        = number of threads to parallelise this process with (total step on a single particle)
    . timestep     = step time taken for euler method step
    . interactions = 1D list, set of particles that will interact with the target particle, where the target is the 0th particle (set from 0->6, [mass, x,y,z, u,v,w]), and n>0th particle is interacting with the target alone
    """
    #Can accept N threads/ranks, where N is any integer >= 2 (1 MASTER and at least 1 WORKER)
    #N=0 => Master Rank
    #N>0 => Worker Ranks

    cdef:
        double initial_time, final_time;
        int i,j, rank_index, start_index;
        int particle_total = np.floor(interactions.shape[0] / 7.0);             #Number of particles in the interaction set (each particle is consecutive list of 7 double elements, [mass, x,y,z, u,v,w])
        int particles_per_rank = 1+np.ceil( (particle_total-1) / (ranks-1));    #Ceil so excess space can just be filled with 0 values
        #double[:] updated_particle = np.zeros(7, dtype=np.double);             #Storage for updated position of target particle
        
        double pos_m = interactions[0];
        double pos_x = 0.0;
        double pos_y = 0.0;
        double pos_z = 0.0;
        double pos_u = 0.0;
        double pos_v = 0.0;
        double pos_w = 0.0;

        double p_separation = 0.0;

    #MPI works best with raw numpy arrays, not memory views
    force                = np.zeros(3, dtype=np.double);                    #Temporarily holds a force vector (length 3) objects, used for each interaction in a split_interaction
    split_interaction    = np.zeros(7*particles_per_rank, dtype=np.double); #A single interaction set, taken by WORKERS to do their part of the calculation
    split_particleForces = np.zeros((ranks, 3), dtype=np.double);           #[Fx, Fy, Fz] from all interactions, from each rank

    with cython.boundscheck(False):
        if(rank == MASTER):
            #Unhash this to get direct direct times from within the parallel section only -> Profile accuracy can sometimes skew this
            initial_time = MPI.Wtime();

            #If the MASTER rank, then send work to WORKERs
            #Split full data set into sections for each rank to compute
            split_interactions = calculate_split_interaction_groups(interactions, particle_total, ranks, particles_per_rank);

            #Broadcast sections to all WORKERS
            for rank_index in range(1, ranks):
                comm.send(split_interactions[rank_index-1], dest=rank_index, tag=rank_index);     #Tagged with index of data sent (although this is already known through its own rank)
                #comm.isend(split_interactions[rank_index-1], dest=rank, tag=rank_index);   #Non-blocking send should be fine here, however blocking is safe -> switch for speed

            #Wait for WORKERS to complete work, collect all data together
            #Set initial values for new particle parameters
            pos_x = interactions[1];
            pos_y = interactions[2];
            pos_z = interactions[3];
            pos_u = interactions[4];
            pos_v = interactions[5];
            pos_w = interactions[6];
            #Still have force = [0,0,0], contributing nothing
            #Note; particle parameters above set to original data for MASTER, so the contributions + original are summed later
        else:
            #If WORKER rank, compute section of work handed to you
            #Wait for data from MASTER (blocking => wait until you get your data)
            split_interaction = comm.recv(source=MASTER, tag=rank);   #If you are rank N, your data set is index N-1

            #Process data
            for j in range(particles_per_rank):
                #Each rank sums the forces from its block of interactions
                start_index = j*7;  #*7 doubles to describe each particle in this 1D flattened array
                p_separation = c_math.sqrt( c_math.pow(split_interaction[1]-split_interaction[start_index+1], 2) + c_math.pow(split_interaction[2]-split_interaction[start_index+2], 2) + c_math.pow(split_interaction[3]-split_interaction[start_index+3], 2) );
                calculate_gravitationalForce(
                    p_separation, 
                    split_interaction[1],
                    split_interaction[2],
                    split_interaction[3],
                    split_interaction[start_index],
                    split_interaction[start_index+1],
                    split_interaction[start_index+2],
                    split_interaction[start_index+3],
                    force
                );        #Sums by reference into the force numpy array
                #Note; Force here has already cancelled mass terms from previous calc.

        #Send data to MASTER (with other workers) with a gather
        split_particleForces = comm.gather(force, root=MASTER);
        if(rank==MASTER):
            split_particleForces = np.array(split_particleForces);          #Convert gathered value to numpy arrays again
            for i in range(split_particleForces.shape[0]):                  #
                split_particleForces[i] = np.array(split_particleForces[i]);#
            pos_u += timestep*np.sum(split_particleForces[1:,0]);  #0th contains only 0 data -> from the MASTER rank
            pos_v += timestep*np.sum(split_particleForces[1:,1]);
            pos_w += timestep*np.sum(split_particleForces[1:,2]);
            pos_x += timestep*pos_u;
            pos_y += timestep*pos_v;
            pos_z += timestep*pos_w;
        
        comm.Barrier();     #To ensure all ranks are able to leave this parallel section correctly and not left behind
        #print("     Synchronised in STEP calc: "+str(rank));

        if(rank==MASTER):
            #Unhash this to get direct direct times from within the parallel section only -> Profile accuracy can sometimes skew this
            final_time = MPI.Wtime();
            #print(str(particle_total)+","+str(ranks)+","+str(final_time-initial_time)+",");    #NOTE; This is used for the "sun_singleCalc" function to get timings data
            #print("Final time= ",final_time-initial_time);

            #Once all synchronised, return collected data
            return np.array([pos_m, pos_x, pos_y, pos_z, pos_u, pos_v, pos_w], dtype=np.double);    #Note; only the MASTER returns a value => the core python file will only recieve a single non-None returned value




#-----------------------------------#
# Runge Kutta Order 4 Time Stepping # -> Using MPI parallelism
#-----------------------------------#
"""
**********
. NOTE; This method is not implemented yet due to being outside the main topic of exploration, 
    however room is left for the program to include functions liek this quite simply
**********
"""
def updateParticle_RungeKutta_MPI(int rank, int MASTER, comm, int ranks, double timestep, double[:] interactions):
    cdef double[:] values = np.zeros(7, dtype=np.double);
    return values;




#----------------------------#
# Euler Method Time Stepping # -> Performed Sequentially (no parallelism)
#----------------------------#
def updateParticle_Euler_Seq(double timestep, double[:] interactions):
    cdef:
        double initial_time, final_time;
        int i, start_index;
        int particle_total = np.floor(interactions.shape[0] / 7.0);        #Number of particles in the interaction set (each particle is consecutive list of 7 double elements, [mass, x,y,z, u,v,w])
        
        double pos_m = interactions[0];
        double pos_x = 0.0;
        double pos_y = 0.0;
        double pos_z = 0.0;
        double pos_u = 0.0;
        double pos_v = 0.0;
        double pos_w = 0.0;

        double p_separation = 0.0;
        double[:] force = np.zeros(3, dtype=np.double);    #Temporarily holds a force vector (length 3) objects

    with cython.boundscheck(False):
        #Setup starting values
        pos_x = interactions[1];
        pos_y = interactions[2];
        pos_z = interactions[3];
        pos_u = interactions[4];
        pos_v = interactions[5];
        pos_w = interactions[6];    
        
        #Process data
        for j in range(particle_total):
            #Each rank sums the forces from its block of interactions
            start_index = j*7;  #*7 doubles to describe each particle in this 1D flattened array
            force[0] = 0.0;
            force[1] = 0.0;
            force[2] = 0.0;
            p_separation = c_math.sqrt( c_math.pow(interactions[1]-interactions[start_index+1], 2) + c_math.pow(interactions[2]-interactions[start_index+2], 2) + c_math.pow(interactions[3]-interactions[start_index+3], 2) );
            calculate_gravitationalForce(
                p_separation, 
                interactions[1],
                interactions[2],
                interactions[3],
                interactions[start_index],
                interactions[start_index+1],
                interactions[start_index+2],
                interactions[start_index+3],
                force
            );        #Sums by reference into the force numpy array
            #Note; Force here has already cancelled mass terms from previous calc.
            pos_u += timestep*np.sum(force[0]);
            pos_v += timestep*np.sum(force[1]);
            pos_w += timestep*np.sum(force[2]);
        pos_x += timestep*pos_u;
        pos_y += timestep*pos_v;
        pos_z += timestep*pos_w;
        
        #Unhash this to get direct direct times from within the parallel section only -> Profile accuracy can sometimes skew this
        final_time = MPI.Wtime();
        #print(str(particle_total)+","+str(ranks)+","+str(final_time-initial_time)+",");    #NOTE; This is used for the "sun_singleCalc" function to get timings data
        #print("Final time= ",final_time-initial_time);

        return np.array([pos_m, pos_x, pos_y, pos_z, pos_u, pos_v, pos_w], dtype=np.double);


#-----------------------------------#
# Runge Kutta Order 4 Time Stepping # -> Performed Sequentially (no parallelism)
#-----------------------------------#
"""
**********
. NOTE; This method is not implemented yet due to being outside the main topic of exploration, 
    however room is left for the program to include functions liek this quite simply
**********
"""
def updateParticle_RungeKutta_Seq(double timestep, double[:] interactions):
    cdef double[:] values = np.zeros(7, dtype=np.double);
    return values;