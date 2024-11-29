import os
os.environ["MKL_NUM_THREADS"]     = "1" #These can be overwritten if more threads wanted -> specify in the function using threads
os.environ["NUMEXPR_NUM_THREADS"] = "1" #This just sets a default value
os.environ["OMP_NUM_THREADS"]     = "1"

import sys;
import numpy as np
import random;
import math;
import cProfile;
import pstats;

from interactionCalc_openMP import updateParticle_RungeKutta_openMP, updateParticle_Euler_openMP;
#updateParticle_Euler_openMP, 

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
    . ID used to locate particles
    . Chosen between large enough range such that matching IDs are exceedingly unlikely
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
    . Details for a FLAT particle (flat meaning stored as [m,x,y,z,u,v,w], as oppose to particle object)
    . Returns formatted core information about this particle used when converting to a string
    """
    dp = 1; #Round origin and dim values to 1.dp
    info = str(round(data[0], dp)) +","+str(round(data[1], dp)) +","+str(round(data[2], dp)) +","+str(round(data[3], dp));
    if(includeVel):
        info += ","+str(round(data[4], dp)) +","+str(round(data[5], dp)) +","+str(round(data[6], dp));
    return info;

def initParticles(n, totalOctLength, arrangement):
    """
    . Initially generates the positions and velocities of particles in the system
    . Structures can be added here and tested in the simulation

    . n = Integer number of particles to be added to system
    . totalOctLength = The width of each side of the cubic rootOct that the particles will be placed into
    . arrangement = Name of structure to determine placment of particles

    . Returns the set of generated particles in a 1D list
    """
    if(verbosity > 0):
        print("Generating ",n," Particles with '"+arrangement+"' arrangement");

    particles = np.zeros(7*n, dtype=np.double);
    
    if(arrangement == "random"):
        velocity_comp_max = 0.005*totalOctLength;   #Maximum velocity in each direction allowed (not overall velocity max)
        for i in range(0,n):
            flat_particle = np.zeros(7, dtype=np.double);
            particle_position_occupiedWidth = 0.75;

            flat_particle[0] = 5.0*pow(10, 31);        #Sun ~10^30kg => approx a solar system per point (10^31 ish)

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

            flat_particle[0] = 5.0*pow(10, 31);        #Sun ~10^30kg => approx a solar system per point (10^31 ish)

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
    . Takes a set of particles and sequentially adds them to the octree
    . Particles may not be successfully added to the tree if they are outside the bounds of the rootOct, resulting in them being removed from the system entirely
    . This is irrelevent given the rootOct is made large enough to contain all particles

    . particles = set of particles to add to tree, formatted as [m,x,y,z,u,v,w]
    . rootOct = original oct in octree to add them to. They are parsed down and generate more octs within the tree as required

    . Returns a list of all particles that were successfully added to the tree
    """
    existing_particles = [];
    particle_total = math.floor(len(particles)/7.0);
    if(verbosity > 1):
        print("Adding ",particle_total," Particles");
    for i in range(0,particle_total):
        start_index = i*7;
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
    """
    dist = vec_dist(point_1, point_2);
    return dist < radius;

#---------------#
# Oct Functions #
#---------------#
class oct:
    def __init__(self, origin, dim):
        self.ID = generateOctID();                  #Used a few case to identify octs
        self.origin = origin;
        self.dim = dim;
        self.data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.double);    #[m, x,y,z, u,v,w]
        self.octs = [None, None, None, None, None, None, None, None];
        self.tree_mass = 0.0;                       #Counts total mass of children
        self.tree_com = vector3(0.0, 0.0, 0.0);     #Counts total com position of children
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
    offset = 0
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
    """
    cOct.data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.double);
    for i in range(0, len(cOct.octs)):
        #Create octs where split occurs
        #cOct.octs.remove(i);   #May need for a deep copy???
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
    . Prints the tree structure from this oct onwards
    . Used for bug fixing
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
    . Each line of this file produced represents one 'frame' of data (usually timestepped through after calculations are done)
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
    . Each line of this file produced represents one 'frame' of data (usually timestepped through after calculations are done)
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

    . The origin and dim of an oct can all be inferred from its position in the list, given the position and 
    dim of the rootOct is given => octs only need to say (1) what data they hold (which particle OR no particles) 
    and (2) what their children have (what data they hold)
    . If an oct does not exist, it can be marked as 'None' in this set
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
def updateParticles_search_parallel_linearList(step_type, threads, particles):
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
            updated_particle = updateParticle_Euler_openMP(interactions, timestep, threads);#updateParticle_Euler_openMP(threads, timestep, interactions);
        elif(step_type == "RungeKutta"):
            updated_particle = updateParticle_RungeKutta_openMP(threads, timestep, interactions);
        for ind in range(len(updated_particle)):
            updated_particles.append(updated_particle[ind]);
    return np.array(updated_particles, dtype=np.double);



def updateParticles_search_parallel(interaction_type, step_type, threads, rootOct, particle_total, partialPercent=1.0):
    """
    . Searches through the entire octree in python to find each particle
    . For each particle, it then finds the force with every other particle in some chosen set (according to method picked, e.g. 'Linear', 'Reduced-Tree')
    . It will then used the parsed in "step_format" to find the new position and velocity of the particle from each resulting force interaction
    . The final new particle position/velocity is then found
    . The list of all these new particle parameters is returned

    . step_type = Name of the method used to step particles in time
    . threads   = Integer number of threads to perform multi-threaded calculation with
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
            search_range = 0.20*spaceLength;   #Search radius
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
        elif("NN_ReducedTree"):
            pass;
        return np.array(interactions, dtype=np.double);

    def recursive_update_data(interaction_type, cOct, threads, updated_particles, particle_counter, particle_total, partialPercent=1.0):
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
                            updated_particle = updateParticle_Euler_openMP(interactions, timestep, threads);#updateParticle_Euler_openMP(threads, timestep, interactions);
                        elif(step_type == "RungeKutta"):
                            updated_particle = updateParticle_RungeKutta_openMP(threads, timestep, interactions);
                        else:
                            print("-- Invalid Step Type: "+step_type+" --");

                        #add_particle_to_flat_list(updated_particles, particle_counter[0], updated_particle);
                        for ind in range(len(updated_particle)):
                            updated_particles.append(updated_particle[ind]);
                        particle_counter[0] += 1;
                    #Then search all its children
                    recursive_update_data(interaction_type, childOct, threads, updated_particles, particle_counter, particle_total, partialPercent=partialPercent);
                #Note; A partial run will cause only the percentage calculated to have dynamics, and hence is not valid to simulate dynamics in, however is valid to get timings for a fraction of a timestep -> as part of this the remianing unchecked particles will not be added to the list, and so returned particles will consist only of those checked, which can have their distribution analysed if needed

    particle_counter  = [0];    #Defined as a list so python parses by reference. Counts the particle being considered, so can efficiently be placed into updated_particle set rtaher than appending each time
    #updated_particles = np.zeros(particles_copy.shape[0], dtype=np.double);         #1D flattened matrix -> [particle, ...], where particle = [m, x,y,z, u,v,w]
    updated_particles = [];

    recursive_update_data(interaction_type, rootOct, threads, updated_particles, particle_counter, particle_total, partialPercent); #Search whole tree

    return np.array(updated_particles, dtype=np.double);

def add_particle_to_flat_list(particle_list, particle_number, particle_data):
    """
    . Adds flat particle details to a flattened list
    . Particle data always 7 long [m, x,y,z, u,v,w]
    . Adds to list by reference
    . Assumes the list is already premade, with spaces assigned

    . particle_list = List to have 7 doubles added to
    . particle_number = Integer index to start adding particle to in this list
    . particle_data = List of 7 doubles describing the particle; [m,x,y,z,u,v,w]

    . Nothing returned (parsed back by reference)
    """
    start_index = 7*particle_number;
    for i in range(0,len(particle_data)):
        particle_list[start_index+i] = particle_data[i];

def sequential_write_save_data(includeVel, rootOct, iter):
    """
    . Writes relevenat data to outside text files
    . This can be;
        - Particle positions, velocities and masses for each frame
        - Octree structure each frame

    . includeVel = Boolean value, 'True' implies particle velocity should be saved to the text file as well
    . rootOct = Oct object, initial oct from which all children will be search for particles to be saved
    . iter = Integer iteration number currently being performed (decides whether to overwrite ro append to saved data)

    . Nothing returned
    """
    #Save particles from the octree
    overwriteType = False;      #Append to clean save file
    if(iter == 0):
        overwriteType = True;   #Create clean save file
    writeOctreeDataToFile(rootOct, includeVel, overwriteType);
    #writeOctreeOctsToFile(rootOct, overwrite=True);



#--------------#
# Main Program #
#--------------#
"""
Verbosity Order;
Each phase is IN ADDITION to the previous;
=0: Program started/ended
>0: Particles generated + iteration number + timestep used
>1: Particles placed each frame
>2: Particles found out-of-bounds
>3: Force interactions completed
...
"""
verbosity = 0;
def main(n_particles, iterations, threads, step_type, interaction_type, useGeneralMethod=True, partialPercent=1.0):
    """
    . Generates a set of particles, formatting in a 1D list as [mass, xPos, yPos, zPos, xVel, yVel, zVel] = [m,x,y,z,u,v,w]
    . Place these particles into a blank oct-object, forming an octree (in which each particle occupies its own oct)
    . Search this tree particles to have their position and velocity updated (according to gravitational interactions)
    . Find set of gravitational interactions to make this occur (using octree to find nearby particles, approximate others as a single particle)
    . Step each of these particles according to these interactions and timestep
    . Repeat the process (place new particles into a fresh octree)

    . n_particles = Integer number of particles to be placed into system initially
    . iterations  = Number of times to repeat the time-stepping calculation on the particle set
    . threads   = Number of threads to compute this calcualtion with
    . step_type = Method used to step particles (e.g. 'Euler', 'RungeKutta', etc)

    . Nothing is returned from this function
    """
    #print("Program Started...");

    #Generate some initial set of particles
    particles = initParticles(n_particles, spaceLength, "disc");    #Change arrangement of particles through the name parameter here

    if(verbosity > 0):
        print("timestep= "+str(timestep));
        print("threads= "+str(threads));

    for iter in range(iterations):
        #State iteration number; Easier to follow program timings for bug fixing
        if(verbosity > 0):
            if(iter % 5 == 0):
                print("iteration= "+str(iter)+"/"+str(iterations));
        
        #Place particles and form tree
        rootOct = generateCleanRootOct(spaceLength);
        particles = placeParticles(particles, rootOct);
        #Save particles from the octree
        sequential_write_save_data(True, rootOct, iter);

        if(useGeneralMethod):   #May want to see purely linear version in testing, but in general avoid it
            #General Method (tree searches)
            particles = updateParticles_search_parallel(interaction_type, step_type, threads, rootOct, particles.shape[0]/7.0, partialPercent=partialPercent);
        else:
            #Fast Linear Method
            particles = updateParticles_search_parallel_linearList(step_type, threads, particles);

    #print("Program Ended");

def run_singleInteractionSet_calc(particle_number, threads):
    """
    . Generates a set of 'particle_number' random particles
    . Performs an interaction calculation on the set
    . This gives an average gauage of the time taken to perform set of interactions
    . In the real program calculation, this will performed for every particle in the set, and hence an estimate to the lower bound of 
    the time for the simulation can be found
    """
    timestep = 0.1;
    particles = initParticles(particle_number, spaceLength, "random");

    updateParticle_Euler_openMP(particles, timestep, threads);

def run_singleInteractionSet_range(vary_value, vary_set, repeats, particle_number, thread_number):
    """
    . Runs the "run_singleInteractionSet_calc()" over a range of values
    . Outputs these results to the given filename

    . vary_range = [initial_value, final_value, jump]

    NOTE; This function requires the print() statement inside the calcualtion to be activate, otherwise 
        there will be no python output, and the output from within the file is more accurate than the 
        cProfile value inside this function.
    """
    particles = particle_number;    #Will be changed if it is the vary_value, will be left alone if not
    threads   = thread_number;      #
    for iter in vary_set:
        if(vary_value == "particles"):
            particles = iter;
        elif(vary_value == "threads"):
            threads   = iter;
        for repeat in range(repeats):
            run_singleInteractionSet_calc(particles, threads);

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
. Program is always run with 4 arguments;
    <mode_number> <threads> <particles> <iterations>
. Certain modes may ignore some of these arguments, in which case any placeholder value can be used
"""
if(int(len(sys.argv)) == 1+4):      #Filename +4 parameters
    #Allow program to try run correct mode
    mode_number      = int(sys.argv[1]);
    thread_number    = int(sys.argv[2]);
    particle_number  = int(sys.argv[3]);
    iteration_number = int(sys.argv[4]);


    if(mode_number == 0):   #Run main simulation

        profile_manager = cProfile.Profile();
        profile_manager.enable();

        main(particle_number, iteration_number, thread_number, "Euler", "ReducedTree", useGeneralMethod=True, partialPercent=1.0);  #"ReducedTree"
    
        profile_manager.disable();
        profile_manager.print_stats(sort='cumulative');
        main_totTime = search_profiler_time(profile_manager, "main");
        print("main_totTime= ",main_totTime);
        update_totTime = search_profiler_time(profile_manager, "updateParticles_search_parallel_linearList");
        print("update_totTime= ",update_totTime);

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=False;
        if(recordPythonLogs):
            outputName = "output_OpenMP_main_"+str(particle_number)+"_"+str(thread_number)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;
        
        profile_manager.print_stats(sort='cumulative');
        #NOTE; Can alternatively just consider the time spent updating particles here
        print(str(int(particle_number))+","+str(int(thread_number))+","+str(main_totTime)+",");

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();


    elif(mode_number == 1): #Run single interaction set as a benchmark

        #If you want to save python terminal output to file, used for pulling data times from OpenMP and MPI calculations
        recordPythonLogs=True;
        if(recordPythonLogs):
            outfile = open("output_openMP.txt", "w");
            sys.stdout = outfile;

        #Running main program
        average_repeats = 5;
        run_singleInteractionSet_range("threads", [2, 4, 8, 16, 24], average_repeats, particle_number, thread_number);

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();
    

    elif(mode_number == 2): #Record single interaction values over a range of threads

        particle_set = [110000];
        for given_particles in particle_set:
            particle_number = given_particles;
            #If you want to save python terminal output to file, used for pulling data times from OpenMP and MPI calculations
            recordPythonLogs=True;
            if(recordPythonLogs):
                outputName = "output_openMP_singleCalc_"+str(int(particle_number))+".txt";
                outfile = open(outputName, "w");
                sys.stdout = outfile;

            #Running main program
            average_repeats = 5;
            run_singleInteractionSet_range("threads", [2, 4, 8, 16], average_repeats, particle_number, thread_number);

            #Reset output, prevent possible changes on future runs
            if(recordPythonLogs):
                sys.stdout = sys.__stdout__;
                outfile.close();
    

    elif(mode_number == 3): #Run main simulation for a range of threads, with repeats

        iteration_number = 1;   #Only interested in the time for a given step ~ same for all steps given random distribution
        repeat_number = 3;

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=True;
        if(recordPythonLogs):
            outputName = "output_OpenMP_main_"+str(particle_number)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;

        for i in range(1,8,1):
            thread_number = i;
            for j in range(repeat_number):
                profile_manager = cProfile.Profile();
                profile_manager.enable();

                main(particle_number, iteration_number, thread_number, "Euler", "Linear",useGeneralMethod=False);
            
                profile_manager.disable();
                main_totTime = search_profiler_time(profile_manager, "main");
                update_totTime = search_profiler_time(profile_manager, "updateParticles_search_parallel_linearList");
                
                #NOTE; Can alternatively just consider the time spent updating particles here
                print(str(int(particle_number))+","+str(int(thread_number))+","+str(main_totTime)+",");

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();
    

    elif(mode_number == 4): #Run just single interactions, no other operations to interfere

        run_singleInteractionSet_calc(particle_number, thread_number);
    

    elif(mode_number == 5): #For getting [Main times] for [varying threads] -Used on BC4 and HomePc

        #Output timings (with other simualtion details) to an output log
        recordPythonLogs=True;
        if(recordPythonLogs):
            outputName = "output_OpenMP_mainThreadSet_"+str(particle_number)+".txt";
            outfile = open(outputName, "w");
            sys.stdout = outfile;
        
        thread_set = [2,4];
        iteration_number = 1;
        repeat_number = 3;
        particle_original_number = particle_number;
        for threads_chosen in thread_set:
            for repeat in range(repeat_number):
                profile_manager = cProfile.Profile();
                profile_manager.enable();

                main(particle_number, iteration_number, threads_chosen, "Euler", "ReducedTree", useGeneralMethod=True, partialPercent=1.0);  #"ReducedTree"
            
                profile_manager.disable();
                main_totTime = search_profiler_time(profile_manager, "main");
                
                #NOTE; Can alternatively just consider the time spent updating particles here
                print(str(int(particle_original_number))+","+str(int(threads_chosen))+","+str(main_totTime)+",");

        #Reset output, prevent possible changes on future runs
        if(recordPythonLogs):
            sys.stdout = sys.__stdout__;
            outfile.close();

    else:
        print("Not a valid mode number, check program for available modes (0 to 5)");
else:
    print("Invalid number of arguments; require 1+4, found "+str(int(len(sys.argv))));
    print("<mode_number> <threads> <particles> <iterations>");
    print("args= "+str( sys.argv ));
