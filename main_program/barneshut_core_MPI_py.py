import os
os.environ["MKL_NUM_THREADS"]     = "1" #These can be overwritten if more threads wanted -> specify in the function using threads
os.environ["NUMEXPR_NUM_THREADS"] = "1" #This just sets a default value
os.environ["OMP_NUM_THREADS"]     = "1"

import sys;
import numpy as np
import random;
import math;
from mpi4py import MPI;
import time;
import cProfile;
import pstats;

from interactionCalc_MPI import updateParticle_Euler_MPI, updateParticle_RungeKutta_MPI, updateParticle_Euler_Seq, updateParticle_RungeKutta_Seq;
from interactionCalc_openMP import updateParticle_RungeKutta_openMP, updateParticle_Euler_openMP;

#-----------#
# Constants #
#-----------#
spaceLength = 0.25*pow(10, 10);         #Units of length for space that octree will sit in
timestep = 0.1;                         #In seconds
G = 6.67*pow(10,-11);


#---------#
# General #
#---------#
class vector3:
    def __init__(self, x, y, z):
        self.x = x;
        self.y = y;
        self.z = z;

def vec_mag(v):
    """
    . Magnitude of a vector
    """
    return ( (v.x)**2 + (v.y)**2 + (v.z)**2 )**(0.5);
def vec_distVec(v1, v2):
    """
    . Vector from v1 to v2
    """
    return vector3(v2.x-v1.x, v2.y-v1.y, v2.z-v1.z);
def vec_dist(v1, v2):
    """
    . Magnitude of difference vector between v1 and v2
    """
    v_diff = vec_distVec(v1,v2);
    return vec_mag(v_diff);
def vec_distUnitVec(v1, v2):
    """
    . Unit vector from v1 to v2
    """
    v_diff = vec_distVec(v1,v2);
    v_mag = vec_mag(v_diff);
    return vector3(v_diff.x/v_mag, v_diff.y/v_mag, v_diff.z/v_mag);
def generateOctID():
    """
    . Unique identifier for each Oct in the tree
    . returns 10 digit ID number to prevent repeats
    """
    return random.randint(1000000000, 9999999999);


#--------------------#
# Particle Functions #
#--------------------#
class particle:
    def __init__(self, pos, vel, mass):
        self.pos = pos;
        self.vel = vel;
        self.mass = mass;

def getDetails_particle(data, includeVel):
    """
    . Details for a FLAT particle
    . Returns formatted core information about this particle used when converting to a 
    string
    """
    dp = 1; #Round origin and dim values to 1.dp
    info = str(round(data[0], dp)) +","+str(round(data[1], dp)) +","+str(round(data[2], dp)) +","+str(round(data[3], dp));
    if(includeVel):
        info += ","+str(round(data[4], dp)) +","+str(round(data[5], dp)) +","+str(round(data[6], dp));
    return info;

def initParticles(n, totalOctLength, arrangement):
    """
    . Generates a set of particles placed in the given arrangement
    
    . n = number of particles
    . totalOctLength = width of cubic space to place particles within
    . arrangement = name of placement system

    . Returns the set it generates as a 1D flattened particle list
    """
    if(verbosity > 0):
        print("Generating ",n," Particles with '"+arrangement+"' arrangement");

    particles = np.zeros(7*n, dtype=np.double);       
    
    if(arrangement == "random"):
        velocity_comp_max = 0.005*totalOctLength;       #Maximum velocity in each direction allowed (not overall velocity max)
        for i in range(0,n):
            flat_particle = np.zeros(7, dtype=np.double);
            particle_position_occupiedWidth = 0.75;

            flat_particle[0] = 5.0*pow(10, 31);         #Sun ~10^30kg => approx a solar system per point (=> 10^31 ish)

            flat_particle[1] = particle_position_occupiedWidth*totalOctLength*random.random() +(1.0-particle_position_occupiedWidth)*totalOctLength/2.0; 
            flat_particle[2] = particle_position_occupiedWidth*totalOctLength*random.random() +(1.0-particle_position_occupiedWidth)*totalOctLength/2.0;
            flat_particle[3] = particle_position_occupiedWidth*totalOctLength*random.random() +(1.0-particle_position_occupiedWidth)*totalOctLength/2.0;

            flat_particle[4] = velocity_comp_max*(0.5-random.random());
            flat_particle[5] = velocity_comp_max*(0.5-random.random());
            flat_particle[6] = velocity_comp_max*(0.5-random.random());

            add_particle_to_flat_list(particles, i, flat_particle);
    if(arrangement == "disc"):
        disc_angular_offset = 0.0;#math.pi/4.0* 2.0*(0.5-random.random());  #Angle the disc plane makes with the y axis, in radians
        particle_position_occupationDiam = [0.48, 0.58, 0.08];              #Diameter zone of the circular disc that particles can be located on (within these two values), the last value is the error allowed for this placement
        omega_perp = 6.5*pow(10,-2);                                        #Angular velocity perpendicular to the disc (e.g. assuming not offset, as is desired behaviour)
        for i in range(0,n):
            if(i==0):
                flat_particle = np.zeros(7, dtype=np.double);
                particle_theta = 2.0*math.pi*random.random();

                flat_particle[0] = 9.5*pow(10, 33); #Approx 10^6 times larger than mass of sun (true for our galaxy)

                flat_particle[1] = totalOctLength*0.54 +(0.5-random.random())*(totalOctLength/100.0); #Near centre, with some slight wobble for more varied dynamics
                flat_particle[2] = totalOctLength*0.54 +(0.5-random.random())*(totalOctLength/100.0); #
                flat_particle[3] = totalOctLength*0.54 +(0.5-random.random())*(totalOctLength/100.0); #
                flat_particle[4] = 0.0;
                flat_particle[5] = 0.0;
                flat_particle[6] = 0.0;

                add_particle_to_flat_list(particles, i, flat_particle);
            else:
                flat_particle = np.zeros(7, dtype=np.double);
                particle_r     = (particle_position_occupationDiam[0] +random.random()*(particle_position_occupationDiam[1] -particle_position_occupationDiam[0]))*totalOctLength/2.0;
                particle_theta = 2.0*math.pi*random.random();
                flat_particle[0] = 5.0*pow(10, 30);#31        #Sun ~10^30kg => approx a solar system per point (10^31 ish)
                flat_particle[1] = particle_r*math.cos(particle_theta) +totalOctLength*0.5 +particle_position_occupationDiam[2]*totalOctLength*random.random(); 
                flat_particle[2] = particle_r*math.sin(particle_theta) +totalOctLength*0.5 +particle_position_occupationDiam[2]*totalOctLength*random.random();
                flat_particle[3] = 0.5*totalOctLength +particle_r*math.sin(particle_theta)*math.sin(disc_angular_offset) +particle_position_occupationDiam[2]*totalOctLength*random.random();

                vel_mag = omega_perp*particle_r *(1.0 -random.random()/6.0);

                flat_particle[4] = vel_mag*(-math.sin(particle_theta) );
                flat_particle[5] = vel_mag*( math.cos(particle_theta) );
                flat_particle[6] = vel_mag*( math.sin(disc_angular_offset) );

                add_particle_to_flat_list(particles, i, flat_particle);
    if(arrangement == "triangular"):
        velocity_comp_max = 0.005*totalOctLength;   #Maximum velocity in each direction allowed (not overall velocity max)
        for i in range(0,n):
            flat_particle = np.zeros(7, dtype=np.double);
            particle_position_occupiedWidth = 0.75;

            flat_particle[0] = 5.0*pow(10, 31);        #Sun ~10^30kg => approx a solar system per point (=> 10^31 ish)

            flat_particle[1] = particle_position_occupiedWidth*totalOctLength*random.random() +(1.0-particle_position_occupiedWidth)*totalOctLength/2.0; 
            flat_particle[2] = particle_position_occupiedWidth*totalOctLength*random.random() +(1.0-particle_position_occupiedWidth)*totalOctLength/2.0;
            flat_particle[3] = math.sqrt( pow(flat_particle[1] -totalOctLength/2.0,2) + pow(flat_particle[2] -totalOctLength/2.0,2) );

            flat_particle[4] = velocity_comp_max*(0.5-random.random());
            flat_particle[5] = velocity_comp_max*(0.5-random.random());
            flat_particle[6] = velocity_comp_max*(0.5-random.random());

            add_particle_to_flat_list(particles, i, flat_particle);

    return particles;

def placeParticles(particles, rootOct):
    """
    . Takes a list of particles and inputs them into a tree via a rootOct

    . particles = 1D list of particles flattened
    . rootOct   = Outermost Oct where all child octs (and all space) is stored

    . Returns the list of particles that were successfully input into the tree (failures only occur for particles placed outside the rootOct boundary 
    or placed directly on top of another particle)
    """
    existing_particles = [];
    particle_total = math.floor(len(particles)/7.0);
    if(verbosity > 1):
        print("Adding ",particle_total," Particles");
    for i in range(0,particle_total):
        start_index = i*7;
        if(particles[start_index] != 0.0):          #If a valid particle -> 0.0 mass particles can be found during updating particles step
            withinX = (0.0 < particles[start_index +1]) or (particles[start_index +1] < spaceLength);
            withinY = (0.0 < particles[start_index +2]) or (particles[start_index +2] < spaceLength);
            withinZ = (0.0 < particles[start_index +3]) or (particles[start_index +3] < spaceLength);
            if(withinX and withinY and withinZ):    #Ensure the particles are in bounds, hence out of bounds errors in tree should be cause for alarm
                for j in range(0,7):                                        #To keep track of particles involved in the system, used by linear method for quicker access to full particle set
                    existing_particles.append( particles[start_index+j] );  #
                addDataToTree(rootOct, particles[start_index:start_index+7]);
    return np.array(existing_particles, dtype=np.double);

def WithinRange(point_1, point_2, radius):
    """
    . Checks if point_1 is within range (radius) of point_2

    . point_1 / 2 = a vector position
    . radius = check radius between two points

    . Returns boolean decision on this condition
    """
    dist = vec_dist(point_1, point_2);
    return dist < radius;


#---------------#
# Oct Functions #
#---------------#
class oct:
    """
    . Oct object
    . Forms the basis of the Octree
    . Stores details about itself (e.g. origin position, dimensions, data it holds, child Octs it holds, COM and position of all child data)
    """
    def __init__(self, origin, dim):
        self.ID = generateOctID();       #Used a few case to identify octs
        self.origin = origin;
        self.dim = dim;
        self.data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.double);    #[m, x,y,z, u,v,w]
        self.octs = [None, None, None, None, None, None, None, None];
        self.tree_mass = 0.0;                   #Counts total mass of children
        self.tree_com = vector3(0.0, 0.0, 0.0); #Counts total com position of children
        #Note; These two values are INCLUSIVE of data they are holding directly too
    
    def getDetails(self, rootOct):
        """
        . Returns formatted core information about this oct used when converting to a 
        string
        . This is not used currently, as tree data is not pulled from the program, just particle data
        . Left open in case generalisation of this data saved method wanted in future
        """
        dp = 1; #Round origin and dim values to 1.dp
        info = "_ID";
        return info;

def addDataToTree(cOct, cData):
    """"
    . Attempts to add the data to the tree, splitting octs if required

    . cOct = Current oct to attempt to add data to (as a root)
    . cData = The data to try add to this section of tree
    
    . Modifies the oct without returning any types
    """
    #Update the mass and com of this oct
    #com = (p1*m1 + p2*m2) / (m1 + m2)  --> Adding a new particle to generate new COM
    cOct.tree_com.x = ((cOct.tree_com.x*cOct.tree_mass)+(cData[1]*cData[0])) / (cOct.tree_mass + cData[0]);
    cOct.tree_com.y = ((cOct.tree_com.y*cOct.tree_mass)+(cData[2]*cData[0])) / (cOct.tree_mass + cData[0]);
    cOct.tree_com.z = ((cOct.tree_com.z*cOct.tree_mass)+(cData[3]*cData[0])) / (cOct.tree_mass + cData[0]);
    cOct.tree_mass += cData[0];   #Note; This has be updated after or calc above will be wrong
    #Add to part of tree
    octCentre = vector3(
        cOct.origin.x +cOct.dim.x/2.0, 
        cOct.origin.y +cOct.dim.y/2.0, 
        cOct.origin.z +cOct.dim.z/2.0
    );
    withinX = (cOct.origin.x <= cData[1]) and (cData[1] < cOct.origin.x +cOct.dim.x); #Check within whole bounds of this oct -> should always be true due to approach from largest oct inwards
    withinY = (cOct.origin.y <= cData[2]) and (cData[2] < cOct.origin.y +cOct.dim.y); #
    withinZ = (cOct.origin.z <= cData[3]) and (cData[3] < cOct.origin.z +cOct.dim.z); #
    hasOcts = checkOctHasOcts(cOct); #If ANY octs are non-None
    offset = 0;
    if(withinX and withinY and withinZ):

        #Z determination
        #In upper layer     => offset=0
        if(cData[3] > octCentre.z):
            #In lower layer => offset=4
            offset = 4
        
        if(cData[1] < octCentre.x):
            #Left side
            if(cData[2] < octCentre.y):
                #Upper side
                #=> Top Left (0)
                if(not np.isnan(cOct.data).any()):
                    #HAS data => NO octs => final oct node (full)
                    #=> Split tree, pass to next oriented oct
                    oldData = np.array( [cOct.data[0], cOct.data[1], cOct.data[2], cOct.data[3], cOct.data[4], cOct.data[5], cOct.data[6]], dtype=np.double ); #New copy, not by ref
                    splitOct(cOct);
                    addDataToTree(cOct, oldData);
                    addDataToTree(cOct, cData);
                else:
                    if(hasOcts):
                        #NO data, HAS octs => intermediate oct
                        #=> Pass to next oriented oct
                        addDataToTree(cOct.octs[0+offset], cData);
                    else:
                        #NO data, NO octs => final oct node (empty)
                        #=> Give data to this oct
                        cOct.data = np.array( [cData[0], cData[1], cData[2], cData[3], cData[4], cData[5], cData[6]], dtype=np.double );

            else:
                #Lower side
                #=> Bottom Left (2)
                if(not np.isnan(cOct.data).any()):
                    #HAS data => NO octs => final oct node (full)
                    #=> Split tree, pass to next oriented oct
                    oldData = np.array( [cOct.data[0], cOct.data[1], cOct.data[2], cOct.data[3], cOct.data[4], cOct.data[5], cOct.data[6]], dtype=np.double );
                    splitOct(cOct);
                    addDataToTree(cOct, oldData);
                    addDataToTree(cOct, cData);
                else:
                    if(hasOcts):
                        #NO data, HAS octs => intermediate oct
                        #=> Pass to next oriented oct
                        addDataToTree(cOct.octs[2+offset], cData);
                    else:
                        #NO data, NO octs => final oct node (empty)
                        #=> Give data to this oct
                        cOct.data = np.array( [cData[0], cData[1], cData[2], cData[3], cData[4], cData[5], cData[6]], dtype=np.double );
        else:
            #Right side
            if(cData[2] < octCentre.y):
                #Upper side
                #=> Top Right (1)
                if(not np.isnan(cOct.data).any()):
                    #HAS data => NO octs => final oct node (full)
                    #=> Split tree, pass to
                    #next oriented oct
                    oldData = np.array( [cOct.data[0], cOct.data[1], cOct.data[2], cOct.data[3], cOct.data[4], cOct.data[5], cOct.data[6]], dtype=np.double );
                    splitOct(cOct);
                    addDataToTree(cOct, oldData);
                    addDataToTree(cOct, cData);
                else:
                    if(hasOcts):
                        #NO data, HAS octs => intermediate oct
                        #=> Pass to next oriented oct
                        addDataToTree(cOct.octs[1+offset], cData);
                    else:
                        #NO data, NO octs => final oct node (empty)
                        #=> Give data to this oct
                        cOct.data = np.array( [cData[0], cData[1], cData[2], cData[3], cData[4], cData[5], cData[6]], dtype=np.double );
            else:
                #Lower side
                #=> Bottom Right (3)
                if(not np.isnan(cOct.data).any()):
                    #HAS data => NO octs => final oct node (full)
                    #=> Split tree, pass to next oriented oct
                    oldData = np.array( [cOct.data[0], cOct.data[1], cOct.data[2], cOct.data[3], cOct.data[4], cOct.data[5], cOct.data[6]], dtype=np.double );
                    splitOct(cOct);
                    addDataToTree(cOct, oldData);
                    addDataToTree(cOct, cData);
                else:
                    if(hasOcts):
                        #NO data, HAS octs => intermediate oct
                        #=> Pass to next oriented oct
                        addDataToTree(cOct.octs[3+offset], cData);
                    else:
                        #NO data, NO octs => final oct node (empty)
                        #=> Give data to this oct
                        cOct.data = np.array( [cData[0], cData[1], cData[2], cData[3], cData[4], cData[5], cData[6]], dtype=np.double );
    else:
        if(verbosity > 2):
            print("-Outside Oct Boundary-");

def splitOct(cOct):
    """
    . Fills the empty oct space with real oct objects
    . Changes Oct by reference
    """
    cOct.data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.double);
    for i in range(0, len(cOct.octs)):
        #Create octs where split occurs
        #cOct.octs.remove(i);   #-> Used to be useful for prior deep copies, not needed anymore
        cOct.octs[i] = oct(getOrigin(cOct, i), getDim(cOct));

def checkOctHasOcts(cOct):
    """
    . Checks if a given Oct has any non-Null child Octs
    """
    hasOcts = False;
    for i in range(0, len(cOct.octs)):
        if(cOct.octs[i] != None):
            hasOcts = True;
            break;
    return hasOcts;

def generateRootOct(cOct):
    """
    . Sets this oct up as a root oct, which covers the entire avalable region
    . Called when a new tree is created
    . Changes made by reference
    """
    for i in range(0,8):
        cOct.octs[i] = oct(getOrigin(cOct, i), getDim(cOct));

def getOrigin(cOct, octOrientation):
    """
    . Gets the postiion of the top left corner of the specified octOrentiation

    . cOct = current Oct being considered
    . octOrientation = the position of the Oct in a 2x2x2 grid, with 0 being the top left coordinate of the upper 
        layer, traversing the layer clockwise before dropping down 1 layer at the original position

    . Returns the valid position
    """
    if(octOrientation == 0):
        return vector3(cOct.origin.x, cOct.origin.y, cOct.origin.z);
    elif(octOrientation == 1):
        return vector3(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y, cOct.origin.z);
    elif(octOrientation == 2):
        return vector3(cOct.origin.x, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z);
    elif(octOrientation == 3):
        return vector3(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z);
    elif(octOrientation == 4):
        return vector3(cOct.origin.x, cOct.origin.y, cOct.origin.z +cOct.dim.z/2.0);
    elif(octOrientation == 5):
        return vector3(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y, cOct.origin.z +cOct.dim.z/2.0);
    elif(octOrientation == 6):
        return vector3(cOct.origin.x, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z +cOct.dim.z/2.0);
    else:
        return vector3(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z +cOct.dim.z/2.0);

def getDim(cOct):
    """
    . Gets the dimensions of a child Oct for a given Oct (always half all dimensions)
    """
    return vector3(cOct.dim.x/2.0, cOct.dim.y/2.0, cOct.dim.z/2.0);

def printTreeStructure(cOct, depth):
    """
    . Prints the tree structure from this Oct onwards
    . Used for Bug-Fixing
    """
    spacing = "";
    for i in range(0, depth):
        spacing += "  ";

    print(spacing+"Depth=",depth,"/ Data; "+str(not np.isnan(cOct.data).any()),);
    print(spacing+"Origin=(",cOct.origin.x,",",cOct.origin.y,",",cOct.origin.z,") // Dim=(",cOct.dim.x,",",cOct.dim.y,",",cOct.dim.z,")");
    for i in range(0, len(cOct.octs)):
        if(cOct.octs[i] != None):
            printTreeStructure(cOct.octs[i], depth+1);
    print(spacing+"---");

def generateCleanRootOct(spaceLength):
    """
    . Creates an empty new Oct to being a tree
    """
    rootOct = oct(
        vector3(0.0, 0.0, 0.0),                             #Origin
        vector3(spaceLength, spaceLength, spaceLength)      #Dimensions
    )
    generateRootOct(rootOct);
    return rootOct;

def fetchOctsWithinRange(nearbyOcts, external_data, cOct, point, radius):
    """
    . Adds to a list (by reference) all octs inside a circular radius of a point
    . Recurrsively called
    p1  p2      p5  p6
    p3  p4      p7  p8

    external_data = [total mass, total com, number of objects summed] --> of particles located OUTSIDE the nearby particles
    """
    p1_within = vec_dist( point, vector3(cOct.origin.x             , cOct.origin.y             , cOct.origin.z) ) < radius;
    p2_within = vec_dist( point, vector3(cOct.origin.x +cOct.dim.x, cOct.origin.y             , cOct.origin.z) ) < radius;
    p3_within = vec_dist( point, vector3(cOct.origin.x             , cOct.origin.y +cOct.dim.y, cOct.origin.z) ) < radius;
    p4_within = vec_dist( point, vector3(cOct.origin.x +cOct.dim.x, cOct.origin.y +cOct.dim.y, cOct.origin.z) ) < radius;

    p5_within = vec_dist( point, vector3(cOct.origin.x             , cOct.origin.y             , cOct.origin.z +cOct.dim.z) ) < radius;
    p6_within = vec_dist( point, vector3(cOct.origin.x +cOct.dim.x, cOct.origin.y             , cOct.origin.z +cOct.dim.z) ) < radius;
    p7_within = vec_dist( point, vector3(cOct.origin.x             , cOct.origin.y +cOct.dim.y, cOct.origin.z +cOct.dim.z) ) < radius;
    p8_within = vec_dist( point, vector3(cOct.origin.x +cOct.dim.x, cOct.origin.y +cOct.dim.y, cOct.origin.z +cOct.dim.z) ) < radius;
    if(p1_within or p2_within or p3_within or p4_within or p5_within or p6_within or p7_within or p8_within):
        #If AT LEAST 1 point within range, worth still considering
        if(p1_within and p2_within and p3_within and p4_within and p5_within and p6_within and p7_within and p8_within):
            #If ALL 8 within range, all children included by default => [Ignore Children]
            hasOcts = checkOctHasOcts(cOct);
            if(hasOcts):
                #If has children, add them all => [Add Children + End of Recurrsion]
                addChildOctsToSet(nearbyOcts, cOct);
            else:
                #If has no children, this is external oct, so just add this one [End of Recurrsion]
                nearbyOcts.append(cOct);
        else:
            #Some but not all points included
            hasOcts = checkOctHasOcts(cOct);
            if(hasOcts):
                #If has children, check which of them is external and involved
                for i in range(0, len(cOct.octs)):
                    if(cOct.octs[i] != None):
                        #Note; This means None octs are also ignored in search, which is a good thing
                        fetchOctsWithinRange(nearbyOcts, external_data, cOct.octs[i], point, radius);
            else:
                #If has NO children, it is external and so should be involved => [End of Recurrsion]
                nearbyOcts.append(cOct);
    else:
        #If NO points within range, check for side overlaps
        x_overlap = abs(point.x -(cOct.origin.x +cOct.dim.x/2.0)) < (radius +cOct.dim.x/2.0);
        y_overlap = abs(point.y -(cOct.origin.y +cOct.dim.y/2.0)) < (radius +cOct.dim.y/2.0);
        z_overlap = abs(point.z -(cOct.origin.z +cOct.dim.z/2.0)) < (radius +cOct.dim.z/2.0);
        if(x_overlap and y_overlap and z_overlap):
            #If there is and X or Y side overlap, then crosses (same as >=1 point within)
            hasOcts = checkOctHasOcts(cOct);
            if(hasOcts):
                #If has children, check which of them is external and involved
                for i in range(0, len(cOct.octs)):
                    if(cOct.octs[i] != None):
                        #Note; This means None octs are also ignored in search, which is a good thing
                        fetchOctsWithinRange(nearbyOcts, external_data, cOct.octs[i], point, radius);
            else:
                #If has NO children, it is external and so should be involved => [End of Recurrsion]
                nearbyOcts.append(cOct);
        else:
            #If not an X, Y or Z overlap either, then unrelated oct => [Ignore Children]
            #Unrelated means its COM and mass should be added to total ---> DOne once over this parent oct which encompasses its children
            external_data[0] += cOct.tree_mass;
            external_data[1].x += cOct.tree_com.x;
            external_data[1].y += cOct.tree_com.y;
            external_data[1].z += cOct.tree_com.z;
            external_data[2] += 1;

def addChildOctsToSet(octSet, cOct):
    """
    . Adds all EXTERNAL child octs to the set, from the root oct given
    . Used for "fetchOctsWithinRange()"
    """
    for i in range(0,len(cOct.octs)):
        if(cOct.octs[i] != None):
            hasOcts = checkOctHasOcts(cOct.octs[i]);
            if(not hasOcts):
                octSet.append(cOct.octs[i]);
            else:
                addChildOctsToSet(octSet, cOct.octs[i]);


#-----------#
# Converter #
#-----------#
def writeOctreeDataToFile(rootOct, includeVel, overwrite=True):
    """
    . Considers a rootOct to start from, finds all the data within this rootOct and its children, then writes this to 
    one line of a "data_info.txt" file
    . This can either be configured to re-write the entire file OR append to the file
    . Each line of this file produced represents one 'frame' of data (usually time-stepped through after calculations are done)
    """
    #Generates list of data
    dataInfo_List = [];
    convertDataFromOctreeToList(rootOct, includeVel, dataInfo_List);
    #Convert the list of data to a string
    dataInfo_String = "";
    for data in dataInfo_List:
        dataInfo_String += data+"*";
    #Writes/appends data to file
    writeType = "a"
    if(overwrite):
        writeType = "w"
    file = open("data_info.txt", writeType);
    if(not overwrite):
        file.write("\n");
    file.write(dataInfo_String);
    file.close();

def writeOctreeOctsToFile(rootOct, overwrite=True):
    """
    . Considers a rootOct to start from, finds all the octs within this rootOct, then writes these octs to
    one line of a "octs_info.txt" file
    . This can either be configured to re-write the entire file OR append to the file
    . Each line of this file produced represents one 'frame' of data (usually time-stepped through after calculations are done)
    """
    #Generates list of data
    octsInfo_List = [];
    convertOctsFromOctreeToList(rootOct, octsInfo_List);

    #Convert the list of data to a string
    octsInfo_String = "";
    for oct in octsInfo_List:
        octsInfo_String += oct;

    #Writes/appends data to file
    writeType = "a"
    if(overwrite):
        writeType = "w"
    file = open("octs_info.txt", writeType);
    if(not overwrite):
        file.write("\n");
    file.write(octsInfo_String);
    file.close();

def convertOctsFromOctreeToList(cOct, octsList):
    """
    . Goes through the octree and pulls all octs found, places them in a list
    . Used to be then visualise the octs in another program
    . Currently used processing sketch "octree_experimental_visualiser" for this job
    """
    #Store this oct's data information -> Only need to know if it has data, not what that data is
    if(not np.isnan(cOct.data).any()):
        octsList.append("T");   #For 'True' => has data
    else:
        octsList.append("F");   #For 'False' => has NO data
    
    #Store children information
    for childOct_index in range(0, len(cOct.octs)):
        if(cOct.octs[childOct_index] != None):
            #If the child exists, give information on it
            octsList.append(">");                          #To indicate we are looking at the previous oct's child now
            convertOctsFromOctreeToList(cOct.octs[childOct_index], octsList);    #Write child's data, and its children's data
        else:
            #If it is empty here, mark it as empty
            octsList.append("N");                           #For 'None' => no oct here
        if(childOct_index == len(cOct.octs)-1):
            octsList.append("*");                           #To indicate the end of the search within an oct's children

def convertDataFromOctreeToList(cOct, includeVel, dataList):
    """
    . Goes through the octree and pulls out all data elements found, places them in a list
    . This can then be given to a visualiser to visualise purely the data (NOT the oct structure, 
    this is done through the "convertOctsFromOctreeToList()" method)

    . dataList = a list parsed by reference that will contain all the particle data
    """
    if(not np.isnan(cOct.data).any()):
        #If this oct has data, store it
        dataList.append( getDetails_particle(cOct.data, includeVel) );
    for childOct in cOct.octs:
        if(childOct != None):
            #If an oct exists here, have a look inside it for data
            convertDataFromOctreeToList(childOct, includeVel, dataList);



#-------------------#
# Main Calculations #
#-------------------#
def add_particle_to_flat_list(particle_list, particle_number, particle_data):
    """
    . Adds flat particle details to a flattened list
    . Particle data always 7 long [m, x,y,z, u,v,w]
    . Adds to list by reference

    . Assumes the list is already premade, with spaces assigned
    """
    start_index = 7*particle_number;
    for i in range(0,len(particle_data)):
        particle_list[start_index+i] = particle_data[i];

def parallel_setup_enviro(particles):
    """
    . Sequentially sets up the next octree and particles for calcualtion to be done on them
    . All ranks will perform this work so they all have their own copies of the octree withour it having to be sent
    """
    rootOct = generateCleanRootOct(spaceLength);        #rootOct is global
    particles = placeParticles(particles, rootOct);     #Returned particles lists number all the particles which were actually placed (e.g inside tree bounds)
    return rootOct, particles;

def sequential_write_save_data(rank, MASTER, includeVel, rootOct, iter):
    """
    . Writes relevenat data to outside text files
    . This can be;
        - Particle positions, velocities and masses for each frame
        - Octree structure each frame
    """
    if(rank==MASTER):
        #Save particles from the octree
        overwriteType = False;      #Append to clean save file
        if(iter == 0):
            overwriteType = True;   #Create clean save file
        writeOctreeDataToFile(rootOct, includeVel, overwriteType);
        #writeOctreeOctsToFile(rootOct, overwrite=True);    #Not used here, just saving particle data not tree data

def get_reducedTree_distibution(particleNumber, totalOctLength, arrangement, search_range, bin_number):
    """
    . Calculates the interactions of each particle in a random set
    . Considers the number of interactions each of these particles has
    . Stores this number of interactions into bins
    . Can use this data to plot the frequency of each set
    """
    def get_reducedTree_interactions(particle, rootOct, sourceOctID, search_range):
        interactions = [];
        for ind in range(particle.shape[0]):      #Add this as target particle
            interactions.append(particle[ind]);   #
        
        #Cannot declare interactions with fixed size here due to unknown particle number required (could however be set extremely high to accomodate any number of particles, but would waste memory)            
        nearbyOcts = [];
        external_data = [0.0, vector3(0.0, 0.0, 0.0), 0];

        #Get nearby octs + external particle data
        #Convert particle to interface with non-flattened version of position
        vector_pos = vector3(particle[1], particle[2], particle[3]);
        fetchOctsWithinRange(nearbyOcts, external_data, rootOct, vector_pos, search_range);

        #Search within the near octs for valid
        for oct in nearbyOcts:
            if(not np.isnan(oct.data).any()):
                if(oct.ID != sourceOctID):                                          #Ensure not adding the source particle twice
                    vector_pos_target = vector3(oct.data[1], oct.data[2], oct.data[3]);
                    if(WithinRange(vector_pos, vector_pos_target, search_range)):    #Have checked that the oct is range, BUT NOT if the particle in the oct is in range
                        #If there is a particle AND it is within the range
                        for i in range(oct.data.shape[0]):          #Add this interaction
                            interactions.append(oct.data[i]);       #
        
        #Collect external octs into a single particle -> core of the approximation
        if(external_data[2] > 0):   #If more than 1 external set added, then include it as another interaction
            ext_particle = np.array([external_data[0] / external_data[2], external_data[1].x / external_data[2], external_data[1].y / external_data[2], external_data[1].z / external_data[2], 0.0, 0.0, 0.0], dtype=np.double); #Make external into a single particle
            for i in range(ext_particle.shape[0]):   #Add this interaction
                interactions.append(ext_particle[i]);  #
        #The forceInteractions_particle set is parsed back through the reference (not required here, but done to follow same format of other recurrsive structure)

        return np.array(interactions, dtype=np.double);

    particles = initParticles(particleNumber, totalOctLength, arrangement);
    rootOct = generateCleanRootOct(totalOctLength);
    particles = placeParticles(particles, rootOct);

    frequency_bin = np.zeros(bin_number);
    bin_size = particleNumber/bin_number;
    for i in range(particleNumber):
        start_index = int(i*7);
        interactions = get_reducedTree_interactions( particles[start_index:start_index+7], rootOct, 0, search_range );
        interaction_number = (interactions.shape[0]/7.0) -1;    #-1 as was allowed to include itself, now removed
        bin_index = min(math.floor(interaction_number/bin_size), bin_number-1); #To prevent bins being overrun if errors occur
        frequency_bin[bin_index] += 1;
    return frequency_bin;


def updateParticles_search_parallel_linearList(rank, MASTER, comm, ranks, step_type, particles):
    """
    . Searches through list and calculates interactions, returns set of updated particles
    . Does this WITHOUT the use of an octree
    . Faster for non-reduced tree methods (methods that do not use the tree)
    . Slower than reduced-tree when used
    . Used to compare to Barnes Hut / Reduced Tree
    """
    updated_particles = [];
    interactions = np.zeros(particles.shape[0], dtype=np.double);
    for i in range( math.floor(particles.shape[0]/7.0) ):
        start_index = 7*i;
        interactions[0:7] = particles[start_index:start_index+7];   #Set target
        if(i > 0):
            interactions[7:7+start_index]  = particles[0:start_index];               #Set previous particles
        interactions[7+start_index:]  = particles[7+start_index:];              #set next particles
        if(step_type == "Euler"):
            updated_particle = updateParticle_Euler_MPI(rank, MASTER, comm, ranks, timestep, interactions);
        elif(step_type == "RungeKutta"):
            updated_particle = updateParticle_RungeKutta_MPI(rank, MASTER, comm, ranks, timestep, interactions);
        if(rank==MASTER):
            for ind in range(len(updated_particle)):
                updated_particles.append(updated_particle[ind]);
    if(rank==MASTER):
        return np.array(updated_particles, dtype=np.double);

def updateParticles_search_parallel_mixed(rank, MASTER, comm, ranks, threads, interaction_type, step_type, rootOct, particle_total, partialPercent=1.0):
    """
    . Searches the tree in parallel (using MPI) and performs force calculations also in parallel (OpenMP)

    . interaction_type = name of interaction grouping wanted, e.g. ReducedTree searching or LinearTree searching
    . step_type = method used to calc force and step particles (currently just 'Euler', but process left generalised in case 'RungeKutta' or any others added in future)
    . particle_total = total number of particles in system
    . partialPercent = percentage of calcualtions to run (per thread) before cancelling (in case simualtions are too long and partial times want ot be recorded instead)

    . Returns set of updated particles based on this calculation
    """
    def recursive_find_particles(cOct, interactions, sourceOctID):
        """
        . Finds particles through searching the tree
        . Will continue to find all particles in tree, linearly NOT using Barnes-Hut method
        """
        #NOTE; interactions parsed by reference, so will implicitly have given all particles
        for childOct in cOct.octs:
            if(childOct!=None):                                         #If this child oct exists
                if(not np.isnan(childOct.data).any()):                  #And if it has data
                    if(childOct.ID != sourceOctID):
                        for i in range(childOct.data.shape[0]):             #Add its components to the interaction
                            interactions.append(childOct.data[i]);          #
                recursive_find_particles(childOct, interactions, sourceOctID);       #Continue searching tree for more particles


    def get_interactions(interaction_type, particle, rootOct, sourceOctID):
        """
        . interaction_type = Which method to generate interactions through
        . particle = [m,x,y,z,u,v,w] particle target, to be interacted with
        . sourceOctID = ID of oct that target particle is located within, used to prevent duplicates in interactions

        . Returns the set of particles required to interact (gravitationally) with this particle
        """
        interactions = [];
        for ind in range(particle.shape[0]):      #Add this as target particle
            interactions.append(particle[ind]);   #
        
        if(interaction_type == "Linear"):
            recursive_find_particles(rootOct, interactions, sourceOctID);
        
        elif(interaction_type == "ReducedTree"):
            """
            . Adds particles to interaction set if within a specified distance fo the given particle
            . This is locally referred to as the 'Reduced-Tree' method, but is the true implementation of the Barnes-Hut method
            . It is named as such due to to the N-Nearest-Neighbours version also being implemented, a spin-off version of Barnes-Hut/Reduced-Tree
            """
            #Cannot declare interactions with fixed size here due to unknown particle number required (could however be set extremely high to accomodate any number of particles, but would waste memory)            
            search_range = 1.0*spaceLength;   #Search radius
            nearbyOcts = [];
            external_data = [0.0, vector3(0.0, 0.0, 0.0), 0];

            #Get nearby octs + external particle data
            #Convert particle to interface with non-flattened version of position
            vector_pos = vector3(particle[1], particle[2], particle[3]);
            fetchOctsWithinRange(nearbyOcts, external_data, rootOct, vector_pos, search_range);

            #Search within the near octs for valid
            for oct in nearbyOcts:
                if(not np.isnan(oct.data).any()):
                    if(oct.ID != sourceOctID):                                          #Ensure not adding the source particle twice
                        vector_pos_target = vector3(oct.data[1], oct.data[2], oct.data[3]);
                        if(WithinRange(vector_pos, vector_pos_target, search_range)):    #Have checked that the oct is range, BUT NOT if the particle in the oct is in range
                            #If there is a particle AND it is within the range
                            for i in range(oct.data.shape[0]):       #Add this interaction
                                interactions.append(oct.data[i]);       #
            
            #Collect external octs into a single particle -> core of the approximation
            if(external_data[2] > 0):   #If more than 1 external set added, then include it as another interaction
                ext_particle = np.array([external_data[0] / external_data[2], external_data[1].x / external_data[2], external_data[1].y / external_data[2], external_data[1].z / external_data[2], 0.0, 0.0, 0.0], dtype=np.double); #Make external into a single particle
                for i in range(ext_particle.shape[0]):   #Add this interaction
                    interactions.append(ext_particle[i]);  #
            #The forceInteractions_particle set is parsed back through the reference (not required here, but done to follow same format of other recurrsive structure)
        return np.array(interactions, dtype=np.double);

    def recursive_update_data(rank, MASTER, comm, ranks, threads, cOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent):
        #NOTE; Here for partialPercent, each WORKER will do up to that percent => 4 WORKERS with partialPercent=0.1 => 40% of particles calculated
        if(particle_counter[0] <= particle_total*partialPercent):  #Continue computing values for particles until reached calculation percentage; This will usually be the entire set (partialPercent=1.0), but for long sets only a percentage of calculations can be performed in order to get partial performance times
            for childOct in cOct.octs:              #For every oct in the tree
                if(childOct != None):               #If the child oct exists
                    if(not np.isnan(childOct.data).any()):      #And if it is holding data (external node)
                        if((particle_counter[0] % (ranks-1))+1 == rank):   #Give this problem to 1 WORKER rank to perform, other move on and try perform the next
                            #Getting particle interactions to act on this one particle
                            interactions = get_interactions(interaction_type, childOct.data, rootOct, childOct.ID);
                            updated_particle = np.zeros(7, dtype=np.double);

                            if(step_type == "Euler"):
                                updated_particle = updateParticle_Euler_openMP(interactions, timestep, threads);
                            elif(step_type == "RungeKutta"):
                                updated_particle = updateParticle_RungeKutta_openMP(threads, timestep, interactions);
                            else:
                                print("-- Invalid Step Type: "+step_type+" --");

                            #All ranks update their own updated_particle sets
                            for ind in range(len(updated_particle)):
                                updated_particles.append(updated_particle[ind]);
                        particle_counter[0] += 1;   #All ranks keep a running count of how many particles encountered in tree -> indexing elements in tree essentially
                    #Then search all its children
                    recursive_update_data(rank, MASTER, comm, ranks, threads, childOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);

    updated_particles = [];
    particle_counter = [0];     #Stored in a list to ensure python parses by reference, ideally would have an int pointer
    
    if(rank!=MASTER):
        #All WORKERS gather their own sets of updated particles
        recursive_update_data(rank, MASTER, comm, ranks, threads, rootOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);
    updated_particles_set =  comm.gather(updated_particles, root=MASTER);   #Gather all updated particles into the MASTER
    if(rank==MASTER):
        #MASTER combines into 1 nicely formatted list, returns this
        compiled_updated_particles = np.array([], dtype=np.double);
        for data_set_index in range(1,len(updated_particles_set)):   #Skip 1 as is the empty contribution given by MASTER
            data_set = np.array(updated_particles_set[data_set_index], dtype=np.double);
            compiled_updated_particles = np.concatenate((compiled_updated_particles, data_set));
        return compiled_updated_particles;



def updateParticles_search_parallel_modified(rank, MASTER, comm, ranks, interaction_type, step_type, rootOct, particle_total, partialPercent=1.0):
    """
    . Searches the tree in parallel (using MPI) and performs force calculations sequentially

    . interaction_type = name of interaction grouping wanted, e.g. ReducedTree searching or LinearTree searching
    . step_type = method used to calc force and step particles (currently just 'Euler', but process left generalised in case 'RungeKutta' or any others added in future)
    . particle_total = total number of particles in system
    . partialPercent = percentage of calcualtions to run (per thread) before cancelling (in case simualtions are too long and partial times want ot be recorded instead)

    . Returns set of updated particles based on this calculation
    """
    def recursive_find_particles(cOct, interactions, sourceOctID):
        """
        . Finds particles through searching the tree
        . Will continue to find all particles in tree, linearly NOT using Barnes-Hut method
        """
        #NOTE; interactions parsed by reference, so will implicitly have given all particles
        for childOct in cOct.octs:
            if(childOct!=None):                                         #If this child oct exists
                if(not np.isnan(childOct.data).any()):                  #And if it has data
                    if(childOct.ID != sourceOctID):
                        for i in range(childOct.data.shape[0]):             #Add its components to the interaction
                            interactions.append(childOct.data[i]);          #
                recursive_find_particles(childOct, interactions, sourceOctID);       #Continue searching tree for more particles


    def get_interactions(interaction_type, particle, rootOct, sourceOctID):
        """
        . interaction_type = Which method to generate interactions through
        . particle = [m,x,y,z,u,v,w] particle target, to be interacted with
        . sourceOctID = ID of oct that target particle is located within, used to prevent duplicates in interactions

        . Returns the set of particles required to interact (gravitationally) with this particle
        """
        interactions = [];
        for ind in range(particle.shape[0]):      #Add this as target particle
            interactions.append(particle[ind]);   #
        
        if(interaction_type == "Linear"):
            recursive_find_particles(rootOct, interactions, sourceOctID);
        
        elif(interaction_type == "ReducedTree"):
            """
            . Adds particles to interaction set if within a specified distance fo the given particle
            . This is locally referred to as the 'Reduced-Tree' method, but is the true implementation of the Barnes-Hut method
            . It is named as such due to to the N-Nearest-Neighbours version also being implemented, a spin-off version of Barnes-Hut/Reduced-Tree
            """
            #Cannot declare interactions with fixed size here due to unknown particle number required (could however be set extremely high to accomodate any number of particles, but would waste memory)            
            search_range = 0.3*spaceLength;   #Search radius
            nearbyOcts = [];
            external_data = [0.0, vector3(0.0, 0.0, 0.0), 0];

            #Get nearby octs + external particle data
            #Convert particle to interface with non-flattened version of position
            vector_pos = vector3(particle[1], particle[2], particle[3]);
            fetchOctsWithinRange(nearbyOcts, external_data, rootOct, vector_pos, search_range);

            #Search within the near octs for valid
            for oct in nearbyOcts:
                if(not np.isnan(oct.data).any()):
                    if(oct.ID != sourceOctID):                                          #Ensure not adding the source particle twice
                        vector_pos_target = vector3(oct.data[1], oct.data[2], oct.data[3]);
                        if(WithinRange(vector_pos, vector_pos_target, search_range)):    #Have checked that the oct is range, BUT NOT if the particle in the oct is in range
                            #If there is a particle AND it is within the range
                            for i in range(oct.data.shape[0]):       #Add this interaction
                                interactions.append(oct.data[i]);       #
            
            #Collect external octs into a single particle -> core of the approximation
            if(external_data[2] > 0):   #If more than 1 external set added, then include it as another interaction
                ext_particle = np.array([external_data[0] / external_data[2], external_data[1].x / external_data[2], external_data[1].y / external_data[2], external_data[1].z / external_data[2], 0.0, 0.0, 0.0], dtype=np.double); #Make external into a single particle
                for i in range(ext_particle.shape[0]):   #Add this interaction
                    interactions.append(ext_particle[i]);  #
            #The forceInteractions_particle set is parsed back through the reference (not required here, but done to follow same format of other recurrsive structure)
        return np.array(interactions, dtype=np.double);

    def recursive_update_data(rank, MASTER, comm, ranks, cOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent):
        #NOTE; Here for partialPercent, each WORKER will do up to that percent => 4 WORKERS with partialPercent=0.1 => 40% of particles calculated
        if(particle_counter[0] <= particle_total*partialPercent):  #Continue computing values for particles until reached calculation percentage; This will usually be the entire set (partialPercent=1.0), but for long sets only a percentage of calculations can be performed in order to get partial performance times
            for childOct in cOct.octs:              #For every oct in the tree
                if(childOct != None):               #If the child oct exists
                    if(not np.isnan(childOct.data).any()):      #And if it is holding data (external node)
                        if((particle_counter[0] % (ranks-1))+1 == rank):   #Give this problem to 1 WORKER rank to perform, other move on and try perform the next
                            #Getting particle interactions to act on this one particle
                            interactions = get_interactions(interaction_type, childOct.data, rootOct, childOct.ID);
                            #print("interactions= ",interactions.shape[0]);
                            updated_particle = np.zeros(7, dtype=np.double);

                            if(step_type == "Euler"):
                                updated_particle = updateParticle_Euler_Seq(timestep, interactions);
                            elif(step_type == "RungeKutta"):
                                updated_particle = updateParticle_RungeKutta_Seq(timestep, interactions);
                            else:
                                print("-- Invalid Step Type: "+step_type+" --");

                            #All ranks update their own updated_particle sets
                            for ind in range(len(updated_particle)):
                                updated_particles.append(updated_particle[ind]);
                        particle_counter[0] += 1;   #All ranks keep a running count of how many particles encountered in tree -> indexing elements in tree essentially
                    #Then search all its children
                    recursive_update_data(rank, MASTER, comm, ranks, childOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);

    updated_particles = [];
    particle_counter = [0];     #Stored in a list to ensure python parses by reference, ideally would have an int pointer
    
    if(rank!=MASTER):
        #All WORKERS gather their own sets of updated particles
        recursive_update_data(rank, MASTER, comm, ranks, rootOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);
    updated_particles_set =  comm.gather(updated_particles, root=MASTER);   #Gather all updated particles into the MASTER
    if(rank==MASTER):
        #MASTER combines into 1 nicely formatted list, returns this
        compiled_updated_particles = np.array([], dtype=np.double);
        for data_set_index in range(1,len(updated_particles_set)):   #Skip 1 as is the empty contribution given by MASTER
            data_set = np.array(updated_particles_set[data_set_index], dtype=np.double);
            compiled_updated_particles = np.concatenate((compiled_updated_particles, data_set));
        return compiled_updated_particles;

def updateParticles_search_parallel(rank, MASTER, comm, ranks, interaction_type, step_type, rootOct, particle_total, partialPercent=1.0):
    """
    . Searches through the entire octree in python to find each particle
    . For each particle, it then finds the force with every other particle in some chosen set (according to method picked, e.g. 'Linear', 'Reduced-Tree')
    . It will then used the parsed in "step_format" to find the new position and velocity of the particle from each resulting force interaction
    . The final new particle position/velocity is then found
    . The list of all these new particle parameters is returned

    . step_type = Name of the method used to step particles in time
    . rootOct   = Oct object containing all other octs, root of the full tree, searched through for interaction pairings
    . particles_copy = copy of original particles, flattened, which are used in each linear calculation => time save to reuse copied set

    . Returns a set of all particles that have been updated (time-stepped) aaccording to the step-type and interaction type provided
    """
    def recursive_find_particles(cOct, interactions, sourceOctID):
        """
        . Finds particles through searching the tree
        . Will continue to find all particles in tree, linearly NOT using Barnes-Hut method
        """
        #NOTE; interactions parsed by reference, so will implicitly have given all particles
        for childOct in cOct.octs:
            if(childOct!=None):                                         #If this child oct exists
                if(not np.isnan(childOct.data).any()):                  #And if it has data
                    if(childOct.ID != sourceOctID):
                        for i in range(childOct.data.shape[0]):             #Add its components to the interaction
                            interactions.append(childOct.data[i]);          #
                recursive_find_particles(childOct, interactions, sourceOctID);       #Continue searching tree for more particles


    def get_interactions(interaction_type, particle, rootOct, sourceOctID):
        """
        . interaction_type = Which method to generate interactions through
        . particle = [m,x,y,z,u,v,w] particle target, to be interacted with
        . sourceOctID = ID of oct that target particle is located within, used to prevent duplicates in interactions

        . Returns the set of particles required to interact (gravitationally) with this particle
        """
        interactions = [];
        for ind in range(particle.shape[0]):      #Add this as target particle
            interactions.append(particle[ind]);   #
        
        if(interaction_type == "Linear"):
            recursive_find_particles(rootOct, interactions, sourceOctID);
        
        elif(interaction_type == "ReducedTree"):
            """
            . Adds particles to interaction set if within a specified distance fo the given particle
            . This is locally referred to as the 'Reduced-Tree' method, but is the true implementation of the Barnes-Hut method
            . It is named as such due to to the N-Nearest-Neighbours version also being implemented, a spin-off version of Barnes-Hut/Reduced-Tree
            """
            #Cannot declare interactions with fixed size here due to unknown particle number required (could however be set extremely high to accomodate any number of particles, but would waste memory)            
            search_range = 0.3*spaceLength;   #Search radius
            nearbyOcts = [];
            external_data = [0.0, vector3(0.0, 0.0, 0.0), 0];

            #Get nearby octs + external particle data
            #Convert particle to interface with non-flattened version of position
            vector_pos = vector3(particle[1], particle[2], particle[3]);
            fetchOctsWithinRange(nearbyOcts, external_data, rootOct, vector_pos, search_range);

            #Search within the near octs for valid
            for oct in nearbyOcts:
                if(not np.isnan(oct.data).any()):
                    if(oct.ID != sourceOctID):                                          #Ensure not adding the source particle twice
                        vector_pos_target = vector3(oct.data[1], oct.data[2], oct.data[3]);
                        if(WithinRange(vector_pos, vector_pos_target, search_range)):    #Have checked that the oct is range, BUT NOT if the particle in the oct is in range
                            #If there is a particle AND it is within the range
                            for i in range(oct.data.shape[0]):       #Add this interaction
                                interactions.append(oct.data[i]);       #
            
            #Collect external octs into a single particle -> core of the approximation
            if(external_data[2] > 0):   #If more than 1 external set added, then include it as another interaction
                ext_particle = np.array([external_data[0] / external_data[2], external_data[1].x / external_data[2], external_data[1].y / external_data[2], external_data[1].z / external_data[2], 0.0, 0.0, 0.0], dtype=np.double); #Make external into a single particle
                for i in range(ext_particle.shape[0]):   #Add this interaction
                    interactions.append(ext_particle[i]);  #
            #The forceInteractions_particle set is parsed back through the reference (not required here, but done to follow same format of other recurrsive structure)
        return np.array(interactions, dtype=np.double);

    def recursive_update_data(rank, MASTER, comm, ranks, cOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent):
        """
        . Implementation of standard approach to gravity simualtion, with some overhead loss through particles requiring location 
            within the octree (rather than their own linear list initially)
        . Recursively travels through the entire tree, updates all particles found
        . Start cycle with some oct (usually rootOct), searches all children of that oct (through cOct)
        . 0(N) travelling
        """
        if(particle_counter[0] <= particle_total*partialPercent):  #Continue computing values for particles until reached calculation percentage; This will usually be the entire set (partialPercent=1.0), but for long sets only a percentage of calculations can be performed in order to get partial performance times
            for childOct in cOct.octs:              #For every oct in the tree
                if(childOct != None):               #If the child oct exists
                    if(not np.isnan(childOct.data).any()):      #And if it is holding data (external node)
                        if(verbosity > 3):
                            print("particle_counter= ",particle_counter);
                        #Getting particle interactions to act on this one particle
                        interactions = get_interactions(interaction_type, childOct.data, rootOct, childOct.ID);
                        updated_particle = np.zeros(7, dtype=np.double);

                        if(step_type == "Euler"):
                            updated_particle = updateParticle_Euler_MPI(rank, MASTER, comm, ranks, timestep, interactions);
                        elif(step_type == "RungeKutta"):
                            updated_particle = updateParticle_RungeKutta_MPI(rank, MASTER, comm, ranks, timestep, interactions);
                        else:
                            print("-- Invalid Step Type: "+step_type+" --");

                        if(rank==MASTER):
                            #add_particle_to_flat_list(updated_particles, particle_counter[0], updated_particle);
                            for ind in range(len(updated_particle)):
                                updated_particles.append(updated_particle[ind]);
                        particle_counter[0] += 1;
                    #Then search all its children
                    recursive_update_data(rank, MASTER, comm, ranks, childOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);

    updated_particles = [];
    particle_counter = [0];     #Stored in a list to ensure python parses by reference, ideally would have an int pointer
    recursive_update_data(rank, MASTER, comm, ranks, rootOct, updated_particles, particle_counter, step_type, interaction_type, particle_total, partialPercent=partialPercent);
    if(rank==MASTER):
        return np.array(updated_particles,dtype=np.double);

def parallel_initialise_enviro(rank, MASTER, comm, particle_number):
    """
    . Intialises the environment the same for all ranks (as particles are placed randomly)
    . This is done by the MASTER creating the setup then sending that data to all ranks to implement
    . After this the ranks will all have the same data and so continue to match (each frame they are updated with the same particle data)
    """
    particles = np.zeros(7*particle_number, dtype=np.double);   #All ranks generate a blank particle set

    if(rank==MASTER):
        #MASTER seeds the random placement of particles, and sends data to other ranks
        particles = initParticles(particle_number, spaceLength, "disc");     #MASTER rank will mainly deal with them however, then hand data to WORKERS
        comm.bcast(particles, root=MASTER);
    else:
        particles = comm.bcast(particles, root=MASTER);
    #Later, all particles will then add these macthing particles to their own octrees
    
    return particles;

def filter_particle_zeros(particles):
    """
    . Considers a list of flattened particles and checks for any purely 0.0 outputs (7 consecutive zeros)
    . These can occur when a particle leaves the octree boundary on lands close to boundary edges, and can cause division by 0 errors
    . This is avoided where possible due to slow speed
    """
    zeroSet_indices = [];       #Will be an ordered list
    particle_total = math.floor(particles.shape[0]/7.0);
    #Locate all particles that have been set to zeros
    for i in range(particle_total):
        isZeroSet = True;
        for j in range(0,7):
            if(particles[i*7 +j]!=0.0):
                isZeroSet=False;
                break;
        if(isZeroSet):
            zeroSet_indices.append(i);
    #Remove all particles set as such
    filter_particles = np.zeros(7*(particle_total-len(zeroSet_indices)), dtype=np.double);
    filter_index = 0;
    for particle_index in range(particle_total):
        if(len(zeroSet_indices) > 0):
            if(particle_index==zeroSet_indices[0]):
                #Is a zero => don't add, and remove this marker
                zeroSet_indices.pop(0);
            else:
                #Is NOT a zero => add to new set
                for i in range(0,7):
                    filter_particles[filter_index*7 +i] = particles[particle_index*7 +i];
                filter_index+=1;
        else:
            #Is NOT a zero => add to new set
            for i in range(0,7):
                filter_particles[filter_index*7 +i] = particles[particle_index*7 +i];
            filter_index+=1;
    return filter_particles;

def displayParticleList(title, particles):
    """
    . Visually displays particle [m, x,y,z, u,v,w] in the console for each particle in the list supplied
    """
    print(title);
    for i in range(particles.shape[0]):
        spaces = "";
        space_number = math.floor(i/7.0) % 2;
        for j in range(space_number):
            spaces+="    ";
        print(spaces+str(particles[i]));
    print("---");



#---------------#
# Main Programs #
#---------------#

"""
Verbosity Order;
Each phase is IN ADDITION to the previous;
=0: Program started/ended
>0: Particles generated + iteration number + timestep used
>1: Particles placed each frame
>2: Particles found out-of-bounds + MPI synchonised prints
>3: Force interactions completed
...
"""
verbosity = 0;
def main_mixed(particle_number, iterations, threads, step_type, interaction_type, partialPercent=1.0):
    """
    . Performs simualtion calculation for N frames (iterations) using the mixed method (MPI and OpenMP)

    . particle_number = total number of particles wanted in system
    . iterations = number of frames to calculate for
    . threads = number of threads to use for OpenMP calcualtion (ranks already fixed on program run through -np ...)
    . interaction type = name of interacting grouping method, e.g ReducedTree, LinearTree, etc
    . partialPercent = percentage of particles to have forces calculated for (in order to record partial times for longer simulations)
    """
    #print("Program Start");

    comm  = MPI.COMM_WORLD;
    ranks = MPI.COMM_WORLD.Get_size();   #Set when you run the program with "mpirun -np 2 python <filename>.py" --> where this file imports this cython
    rank  = MPI.COMM_WORLD.Get_rank();
    MASTER= 0;

    #Initial setup
    particles = parallel_initialise_enviro(rank, MASTER, comm, particle_number);

    #Loop for each frame
    for iter in range(iterations):
        rootOct, particles = parallel_setup_enviro(particles);  #All ranks are now have their own copy of rootOct and particles that should match
        sequential_write_save_data(rank, MASTER, True, rootOct, iter);  #True/False to include/disclude velocity in save
        particles = updateParticles_search_parallel_mixed(rank, MASTER, comm, ranks, threads, interaction_type, step_type, rootOct, particles.shape[0]/7.0, partialPercent=partialPercent);
        particles = comm.bcast(particles, root=MASTER);     #Then send this data to other particles and wait for them, so all ranks are on the same page
        #comm.Barrier(); #Sync up ranks ready for next iteration --> Should already be synced here however

def main_modified(particle_number, iterations, step_type, interaction_type, partialPercent=1.0):
    """
    . Performs simualtion calculation for N frames (iterations) using the modified method (MPI parallel over tree searching)

    . particle_number = total number of particles wanted in system
    . iterations = number of frames to calculate for
    . interaction type = name of interacting grouping method, e.g ReducedTree, LinearTree, etc
    . partialPercent = percentage of particles to have forces calculated for (in order to record partial times for longer simulations)
    """
    #print("Program Start");

    comm  = MPI.COMM_WORLD;
    ranks = MPI.COMM_WORLD.Get_size();   #Set when you run the program with "mpirun -np 2 python <filename>.py" --> where this file imports this cython
    rank  = MPI.COMM_WORLD.Get_rank();
    MASTER= 0;

    #Initial setup
    particles = parallel_initialise_enviro(rank, MASTER, comm, particle_number);

    #Loop for each frame
    for iter in range(iterations):
        rootOct, particles = parallel_setup_enviro(particles);  #All ranks are now have their own copy of rootOct and particles that should match
        sequential_write_save_data(rank, MASTER, True, rootOct, iter);  #True/False to include/disclude velocity in save
        particles = updateParticles_search_parallel_modified(rank, MASTER, comm, ranks, interaction_type, step_type, rootOct, particles.shape[0]/7.0, partialPercent=partialPercent);
        particles = comm.bcast(particles, root=MASTER);     #Then send this data to other particles and wait for them, so all ranks are on the same page
        #comm.Barrier(); #Sync up ranks ready for next iteration --> does not seem needed here


def main(particle_number, iterations, step_type, interaction_type, useGeneralMethod=True, partialPercent=1.0):
    """
    . Performs simualtion calculation for N frames (iterations) using the original main method (MPI parallel over force calculation)

    . particle_number = total number of particles wanted in system
    . iterations = number of frames to calculate for
    . interaction type = name of interacting grouping method, e.g ReducedTree, LinearTree, etc
    . useGeneralMethod = whether to use tree searchig methods (True) or the purely linear calcualtion (False)
    . partialPercent = percentage of particles to have forces calculated for (in order to record partial times for longer simulations)
    """
    #print("Program Start");

    comm  = MPI.COMM_WORLD;
    ranks = MPI.COMM_WORLD.Get_size();   #Set when you run the program with "mpirun -np 2 python <filename>.py" --> where this file imports this cython
    rank  = MPI.COMM_WORLD.Get_rank();
    MASTER= 0;

    if(verbosity > 0):
        if(rank==MASTER):
            print("timestep= "+str(timestep));

    #Initial setup
    particles = parallel_initialise_enviro(rank, MASTER, comm, particle_number);

    #Loop for each frame
    for iter in range(iterations):
        if(verbosity > 0):
            if(rank==MASTER):
                if(iter % 5 == 0):
                    print("iteration= "+str(iter)+"/"+str(iterations));
        rootOct, particles = parallel_setup_enviro(particles);  #All ranks are now have their own copy of rootOct and particles that should match


        sequential_write_save_data(rank, MASTER, True, rootOct, iter);  #True/False to include/disclude velocity in save


        #updated_particles = parallel_update_particles(rank, MASTER, comm, ranks, rootOct, particles.shape[0], step_type, [particles]);
        if(useGeneralMethod):   #May want to see purely linear version in testing, but in general avoid it
            #General Method (tree searches)
            particles = updateParticles_search_parallel(rank, MASTER, comm, ranks, interaction_type, step_type, rootOct, particles.shape[0]/7.0, partialPercent=partialPercent);
        else:
            #Fast Linear Method (No tree searching FOR linear)
            particles = updateParticles_search_parallel_linearList(rank, MASTER, comm, ranks, step_type, particles);


        if(rank==MASTER):
            if(verbosity > 2):
                displayParticleList("Master Returned List= ", particles);
        
        #Then send this data to other particles and wait for them, so all ranks are on the same page
        particles = comm.bcast(particles, root=MASTER);
        #comm.Barrier(); #Sync up ranks ready for next iteration --> does not seem needed here
        if(verbosity > 2):
            print("Synchonised in MAIN: "+str(rank));

    #print("Program Ended");

def run_singleInteractionSet_calc(particle_number):
    """
    . Generates a set of 'particle_number' random particles
    . Performs an interaction calculation on the set
    . This gives an average gauage of the time taken to perform set of interactions
    . In the real program calculation, this will performed for every particle in the set, and hence an estimate to the lower bound of 
    the time for the simulation can be found
    """
    comm  = MPI.COMM_WORLD;
    ranks = MPI.COMM_WORLD.Get_size();   #Set when you run the program with "mpirun -np 2 python <filename>.py" --> where this file imports this cython
    rank  = MPI.COMM_WORLD.Get_rank();
    MASTER= 0;
    #Generate set & tree
    particles = parallel_initialise_enviro(rank, MASTER, comm, particle_number);
    #Run test set
    updated_particle = updateParticle_Euler_MPI(rank, MASTER, comm, ranks, timestep, particles);

def search_profiler_time(cProfiler, cFunc):
    """
    . Finds the cumulative time spent in the given function, from the profiler provided

    . Returns the total time in the given function, or 0.0 if not found
    """
    totalTime = 0.0;
    for stat_key, stat_value in pstats.Stats(cProfiler).stats.items():  #Look through dictionary of tuples
            if(stat_key[2] == cFunc):                                   #If the function_name parameter in the tuple key is for the function in question
                totalTime = stat_value[3];                              #Pull the cumualtive time (3rd arg) from this value
    return totalTime;


#-------------------#
# Run Program Modes #
#-------------------#
"""
. Program is always run with 3 arguments;
    <mode_number> <particles> <iterations>
. Note that ranks is given when running the program with "mpirun -np X"
. Certain modes may ignore some of these arguments, in which case any placeholder value can be used
"""
#All ranks enter through here
ranks  = MPI.COMM_WORLD.Get_size();
rank   = MPI.COMM_WORLD.Get_rank();
MASTER = 0;
if(int(len(sys.argv)) == 1+3):      #Filename +3 parameters
    #Allow program to try run correct mode
    mode_number      = int(sys.argv[1]);
    particle_number  = int(sys.argv[2]);
    iteration_number = int(sys.argv[3]);


    if(mode_number == 0):   #Run main program

        if(rank==MASTER):
            profile_manager = cProfile.Profile();
            profile_manager.enable();
        
        main(particle_number, iteration_number, "Euler", "Linear", useGeneralMethod=False, partialPercent=1.0);  #"ReducedTree"
    
        #Output for timings
        if(rank==MASTER):
            profile_manager.disable();
            main_totalTime = search_profiler_time(profile_manager, "main");
            print("main_totalTime= ",main_totalTime);
    
            #Output timings (with other simualtion details) to an output log
            recordPythonLogs=False;     #Whether to store results in a file
            if(recordPythonLogs):
                outputName = "output_MPI_main_"+str(particle_number)+"_"+str(ranks)+".txt";
                outfile = open(outputName, "w");
                sys.stdout = outfile;
            
            #NOTE; Can alternatively just consider the time spent updating particles here
            print(str(int(particle_number))+","+str(int(ranks))+","+str(main_totalTime)+",");

            #Reset output, prevent possible changes on future runs
            if(recordPythonLogs):
                sys.stdout = sys.__stdout__;
                outfile.close();
    

    elif(mode_number == 1): #Run single interaction calculations
        if(rank==MASTER):
            profile_manager = cProfile.Profile();
            profile_manager.enable();

        run_singleInteractionSet_calc(particle_number);
    
        #Output for timings
        if(rank==MASTER):
            profile_manager.disable();
            singleInter_totalTime = search_profiler_time(profile_manager, "run_singleInteractionSet_calc");
            paraInit_totalTime = search_profiler_time(profile_manager, "parallel_initialise_enviro");
            update_totalTime = search_profiler_time(profile_manager, "updateParticle_Euler_MPI");
            print("singleInter_totalTime= ",singleInter_totalTime);
            print("paraInit_totalTime= ",paraInit_totalTime);
            print("update_totalTime= ",update_totalTime);

    elif(mode_number == 2): #Running main calculation once (1 step), testing various particle sizes for a given number of ranks, repeated N times

        iteration_number = 1;   #For repeated test like this, only interested in single iteration (each identical assuming random distribution)
        repeat_number = 3;
        p_min  = 1000;
        p_max  = 1000+1;
        p_jump = 500;

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=True;
        if(recordPythonLogs):
            outputName = "output_MPI_mainOne_"+str(ranks)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;
        
        for i in range(p_min, p_max, p_jump):
            particle_number = i;
            for j in range(repeat_number):
                if(rank==MASTER):
                    profile_manager = cProfile.Profile();
                    profile_manager.enable();
                
                main(particle_number, iteration_number, "Euler", "Linear", useGeneralMethod=False, partialPercent=1.0);
            
                #Output for timings
                if(rank==MASTER):
                    profile_manager.disable();
                    main_totalTime = search_profiler_time(profile_manager, "main");
                    
                    #NOTE; Can alternatively just consider the time spent updating particles here too (or any other function found through profiler)
                    print(str(int(particle_number))+","+str(int(ranks))+","+str(main_totalTime)+",");

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();
    

    elif(mode_number == 3): #Running a test of just parallel calc interaction, testing various particle sizes for a given number of ranks, repeated N times

        iteration_number = 1;   #For repeated test like this, only interested in single iteration (each identical assuming random distribution)
        repeat_number = 3;
        p_min  = 100;
        p_max  = 100+1;
        p_jump = 5000;

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=True;
        if(recordPythonLogs):
            outputName = "output_MPI_singleCalc_"+str(ranks)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;
        
        for i in range(p_min, p_max, p_jump):
            particle_number = i;
            for j in range(repeat_number):
                
                #NOTE; This should use timings from directly within the interactionCalc function, hence requires the file to be built again with this print unhashed
                #   -> The profiler does not have the accuracy to pick up on the speeds for this single calc => better to print from within and log all prints here
                run_singleInteractionSet_calc(particle_number);

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();
    

    elif(mode_number == 4): #By using MPI for each particle, linear interactions, using Tree searching methods
        
        if(rank==MASTER):
            print("Modified MPI Approach");
            profile_manager = cProfile.Profile();
            profile_manager.enable();

        main_modified(particle_number, iteration_number, "Euler", "ReducedTree", partialPercent=1.0);

        main_totalTime = 0.0;
        if(rank==MASTER):
            profile_manager.disable();
            main_totalTime = search_profiler_time(profile_manager, "main_modified");
        
    
        #Output for timings
        if(rank==MASTER):
            recordPythonLogs=False;
            if(recordPythonLogs):
                outputName = "output_MPI_MainModified_"+str(ranks)+".txt";
                outfile = open(outputName, "w");
                sys.stdout = outfile;
        
            #NOTE; Can alternatively just consider the time spent updating particles here too (or any other function found through profiler)
            print(str(int(particle_number))+","+str(int(ranks))+","+str(main_totalTime)+",");

            #Reset output, prevent possible changes on future runs
            if(recordPythonLogs):
                sys.stdout = sys.__stdout__;
                outfile.close();
    

    elif(mode_number == 5): #By using MPI for each particle, OpenMP prange for the set of interactions, using Tree searching methods

        #Output for timings
        recordPythonLogs=True;
        outputName = "output_MPI_MainMixed_"+str(ranks)+".txt";
        outfile = open(outputName, "w");
        if(rank==MASTER):
            print("Mixed MPI-OpenMP Approach");
            if(recordPythonLogs):
                sys.stdout = outfile;

        thread_set = [1, 4];
        repeat_number = 3;
        for thread_number in thread_set:
            for repeat in range(repeat_number):
                if(rank==MASTER):
                    profile_manager = cProfile.Profile();
                    profile_manager.enable();
                
                main_mixed(particle_number, iteration_number, thread_number, "Euler", "ReducedTree", partialPercent=1.0);

                if(rank==MASTER):
                    profile_manager.disable();
                    main_totalTime = search_profiler_time(profile_manager, "main_mixed");
                    #NOTE; Can alternatively just consider the time spent updating particles here too (or any other function found through profiler)
                    print(str(int(particle_number))+","+str(int(ranks))+","+str(main_totalTime)+","+str(thread_number)+",");

        #Reset output, prevent possible changes on future runs
        if(rank==MASTER):
            if(recordPythonLogs):
                sys.stdout = sys.__stdout__;
                outfile.close();
    

    elif(mode_number == 6): #Run main_modified for various particle numbers

        iteration_number = 1;   #For repeated test like this, only interested in single iteration (each identical assuming random distribution)
        repeat_number = 3;

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=True;
        if(recordPythonLogs):
            outputName = "output_MPI_mainOne_modified_"+str(ranks)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;
        
        particle_num_set = [500];
        for i in particle_num_set:
            particle_number = i;
            for j in range(repeat_number):
                if(rank==MASTER):
                    profile_manager = cProfile.Profile();
                    profile_manager.enable();
                
                main_modified(particle_number, iteration_number, "Euler", "ReducedTree", partialPercent=1.0);
            
                #Output for timings
                if(rank==MASTER):
                    profile_manager.disable();
                    main_totalTime = search_profiler_time(profile_manager, "main_modified");
                    
                    #NOTE; Can alternatively just consider the time spent updating particles here too (or any other function found through profiler)
                    print(str(int(particle_number))+","+str(int(ranks))+","+str(main_totalTime)+",");

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();
    

    elif(mode_number == 7): #Get reduced tree particle number distribution

        if(rank==MASTER):
            factor_set = [0.2, 0.4, 0.5];
            for factor in factor_set:
                search_range = factor*spaceLength;
                frequency_bin = get_reducedTree_distibution(particle_number, spaceLength, "disc", search_range, 21);
                print("search_range= ",search_range,",  particle= ",particle_number);
                print("frequency_bin= ",frequency_bin);
                #Manually copy data over to plotter --> Only ever interested in ~20 bins


    else:
        print("Not a valid mode number, check program for available modes (0 to 7)");


else:
    print("Invalid number of arguments; require 1+3, found "+str(int(len(sys.argv))));
    print("<mode_number> <particles> <iterations>");
    print("args= "+str( sys.argv ));