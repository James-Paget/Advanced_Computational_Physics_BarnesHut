import os
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]     = "1"

import numpy as np;
cimport numpy as np;

import cython;
from cython.parallel cimport parallel, prange;
cimport openmp;
cimport libc.math as c_math;

cdef double G = 6.67*c_math.pow(10,-11)


#---------------------------#
# General Force Calculation #
#---------------------------#
cdef double getCalcVal_X(double separation, double p1_x, double p1_y, double p1_z, double p2_m, double p2_x, double p2_y, double p2_z) noexcept nogil:
    cdef double diff_vec = p2_x -p1_x;
    if(diff_vec != 0.0):
        return G*p2_m*diff_vec/c_math.pow(separation, 3);
    else:
        return 0.0;

cdef double getCalcVal_Y(double separation, double p1_x, double p1_y, double p1_z, double p2_m, double p2_x, double p2_y, double p2_z) noexcept nogil:
    cdef double diff_vec = p2_y -p1_y;
    if(diff_vec != 0.0):
        return G*p2_m*diff_vec/c_math.pow(separation, 3);
    else:
        return 0.0;

cdef double getCalcVal_Z(double separation, double p1_x, double p1_y, double p1_z, double p2_m, double p2_x, double p2_y, double p2_z) noexcept nogil:
    cdef double diff_vec = p2_z -p1_z;
    if(diff_vec != 0.0):
        return G*p2_m*diff_vec/c_math.pow(separation, 3);
    else:
        return 0.0;


#----------------------------#
# Euler Method Time Stepping #
#----------------------------#
def updateParticle_Euler_openMP(double[:] interactions, double timestep, int threads):

    cdef:
        int n_particles = np.floor(interactions.shape[0] / 7.0);

        double pos_m = interactions[0];
        double pos_x = interactions[1];
        double pos_y = interactions[2];
        double pos_z = interactions[3];
        double pos_u = interactions[4];
        double pos_v = interactions[5];
        double pos_w = interactions[6];
        double[:] updated_particle = np.zeros(7, dtype=np.double);

        double[:] force_x = np.zeros(n_particles, dtype=np.double);
        double[:] force_y = np.zeros(n_particles, dtype=np.double);
        double[:] force_z = np.zeros(n_particles, dtype=np.double);

        double p_separation = 0.0;
        double initial, final;
        int i, j, start_index;

    #print("Threads set to {:2d}".format(threads))
    #initial = openmp.omp_get_wtime()
    
    with cython.boundscheck(False):
        for i in prange(i, n_particles, nogil=True, num_threads=threads, schedule='dynamic'):
            start_index = 7*i;
            p_separation = c_math.sqrt( c_math.pow(interactions[1] -interactions[start_index+1],2) + c_math.pow(interactions[2] -interactions[start_index+2],2) + c_math.pow(interactions[3] -interactions[start_index+3],2) );
            force_x[i] = getCalcVal_X(
                p_separation,
                interactions[1], interactions[2], interactions[3], 
                interactions[start_index], interactions[start_index+1], interactions[start_index+2], interactions[start_index+3]
            );
            force_y[i] = getCalcVal_Y(
                p_separation,
                interactions[1], interactions[2], interactions[3], 
                interactions[start_index], interactions[start_index+1], interactions[start_index+2], interactions[start_index+3]
            );
            force_z[i] = getCalcVal_Z(
                p_separation,
                interactions[1], interactions[2], interactions[3], 
                interactions[start_index], interactions[start_index+1], interactions[start_index+2], interactions[start_index+3]
            );

            pos_u += timestep*force_x[i];   #Seems to be slightly more consistent & same average speed as below
            pos_v += timestep*force_y[i];
            pos_w += timestep*force_z[i];
        #pos_u += timestep*np.sum(force_x);  #Similar speed when included and not in the parallel part
        #pos_v += timestep*np.sum(force_y);
        #pos_w += timestep*np.sum(force_z);
        pos_x += pos_u*timestep;
        pos_y += pos_v*timestep;
        pos_z += pos_w*timestep;

    #final = openmp.omp_get_wtime();
    #print(str(c_math.floor(interactions.shape[0]/7.0))+","+str(threads)+","+str(final-initial)+","); #NOTE; This print is used for the "run_singleInteractionSet_range()" function, unhash when wanted. ALSO, this needs the final ',' ending the list
    
    return np.array([pos_m, pos_x, pos_y, pos_z, pos_u, pos_v, pos_w], dtype=np.double);




#-----------------------------------#
# Runge Kutta Order 4 Time Stepping #
#-----------------------------------#
"""
**********
. NOTE; This method is not implemented yet as outside the scope of what is being tested, however it is clear 
    to see in the sketched approach how this would be possible.
    This would simply extend the capabilities of the simulation, and does not affect the rest of the parallel 
    approaches taken.
**********
"""
cdef double DE_x(double step, double v_coord) noexcept nogil:
    """
    v = velocity of particle

    .NOTE; Potential speedup could be gained by simply not calling this function (use the value it returns directly instead), however 
    for clarity this is left as a function instead (and speedup would only be slight)
    """
    return v_coord+step;

cdef double DE_v(double step, double x1_x, double x1_y, double x1_z, double x2_m, double x2_x, double x2_y, double x2_z, int coord) noexcept nogil:
        """
        x1 = position of particle 1
        x2 = "" "" particle 2 "" ""
        m1 = mass of particle 1
        m2 = "" ""
        coord = coordinate of position that is being stepped
        """
        #Note; Pos already stepped when input into this function
        cdef double distVec_coord;
        cdef double dist_mag;
        cdef double force_mag = 0.0;
        if(coord == 0):
            x1_x += step;
            distVec_coord = x2_x -x1_x;
        elif(coord == 1):
            x1_y += step;
            distVec_coord = x2_y -x1_y;
        else:
            x1_z += step;
            distVec_coord = x2_z -x1_z;

        dist_mag = c_math.sqrt( c_math.pow(x2_x -x1_x,2) + c_math.pow(x2_y -x1_y,2) + c_math.pow(x2_z -x1_z,2) );
        if(dist_mag != 0):
            force_mag = G*x2_m/(dist_mag**3);  #^3 NOT ^2 here as multiply by r NOT r_hat in next line -> 1 less computation
        #Left as 0 otherwise => ignored (overlapping particles, prevent infinite value)

        if(coord == 0):
            x1_x -= step;
        elif(coord == 1):
            x1_y -= step;
        else:
            x1_z -= step;
        return force_mag*distVec_coord;

def updateParticle_RungeKutta_openMP(int threads, double timestep, double[:] interactions):
    """
    . Updates the position and velocity of a single particle
    . Uses OpenMP parallelisation to calculate all interactions acting on that particle faster
    """
    cdef:
        int i,j, target_index;
        int particle_total = np.floor(interactions.shape[0]/7.0);
        double[:] updated_particle = np.zeros(7, dtype=np.double);
        double[:] k1_x_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k2_x_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k3_x_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k4_x_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k_x_shared  = np.zeros(particle_total, dtype=np.double);
        double[:] k1_v_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k2_v_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k3_v_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k4_v_shared = np.zeros(particle_total, dtype=np.double);
        double[:] k_v_shared  = np.zeros(particle_total, dtype=np.double);

    #Set initial values for updated particle
    updated_particle[0] = interactions[0];                              #Mass unchanged
    updated_particle[1] = interactions[1];        #Pos = Original + dx step
    updated_particle[2] = interactions[2];        #e.g no interactions => moves just by its unaltered t-step
    updated_particle[3] = interactions[3];        #
    updated_particle[4] = interactions[4];                              #Velocity same to begin with
    updated_particle[5] = interactions[5];                              #
    updated_particle[6] = interactions[6];                              #

    with nogil:
        """
        #Do every interaction in parallel
        for i in prange(1,particle_total):
        target_index = 7*i;     #Deal with indices between [target_index, target_index+7] --> The particle this thread is considering, handed out by scheduler
        #For each coordinate
        for j in range(3):

            k1_x_shared[i] = timestep*DE_x(
                timestep*0.0 /2.0,
                interactions[4+j]  #Vj component of main velocity
            );
            k1_v_shared[i] = timestep*DE_v(
                timestep*0.0 /2.0,
                interactions[1],
                interactions[2],
                interactions[3],
                interactions[target_index+0],
                interactions[target_index+1],
                interactions[target_index+2],
                interactions[target_index+3],
                j
            );

            k2_x_shared[i] = timestep*DE_x(
                timestep*k1_v_shared[i] /2.0,
                interactions[4+j]  #Vj component of main velocity
            );
            k2_v_shared[i] = timestep*DE_v(
                timestep*k1_x_shared[i] /2.0,
                interactions[1],
                interactions[2],
                interactions[3],
                interactions[target_index+0],
                interactions[target_index+1],
                interactions[target_index+2],
                interactions[target_index+3],
                j
            ); 
            
            k3_x_shared[i] = timestep*DE_x(
                timestep*k2_v_shared[i] /2.0,
                interactions[4+j]  #Vj component of main velocity
            );
            k3_v_shared[i] = timestep*DE_v(
                timestep*k2_x_shared[i] /2.0,
                interactions[1],
                interactions[2],
                interactions[3],
                interactions[target_index+0],
                interactions[target_index+1],
                interactions[target_index+2],
                interactions[target_index+3],
                j
            );
            
            k4_x_shared[i] = timestep*DE_x(
                timestep*k3_v_shared[i] /2.0,
                interactions[4+j]  #Vj component of main velocity
            );
            k4_v_shared[i] = timestep*DE_v(
                timestep*k3_x_shared[i] /2.0,
                interactions[1],
                interactions[2],
                interactions[3],
                interactions[target_index+0],
                interactions[target_index+1],
                interactions[target_index+2],
                interactions[target_index+3],
                j
            );

            k_x_shared[i] = timestep*(k1_x_shared[i] +2.0*k2_x_shared[i] +2.0*k3_x_shared[i] +k4_x_shared[i])/6.0;
            k_v_shared[i] = timestep*(k1_v_shared[i] +2.0*k2_v_shared[i] +2.0*k3_v_shared[i] +k4_v_shared[i])/6.0;

            updated_particle[j+1] += k_x_shared[i];   #+1 to get to positions  [1,2,3]
            updated_particle[j+4] += k_v_shared[i];   #+4 to get to velocities [4,5,6]
        """
    return updated_particle;    #Return single particle time stepped with Runge-Kutta, through all its interactions
