class particle {
    vector3 pos;
    vector3 vel;
    float m;

    float particle_size = 10.0;
    vector3 particle_color = new vector3(255.0, 255.0, 255.0);//fetchParticleColor(pos.z);

    particle(float m, vector3 pos, vector3 vel) {
        this.m = m;
        this.pos = pos;
        this.vel = vel;
        float factor = ( log(m)/log(10) )/30.0; //Sun ~ 10^30kg
        particle_size = 5.0*factor;
        if(m > pow(10,32)) {    //FOr VERY large masses, scale non-linearly to reach desired size faster (mass is far too big for linear scaling without breaking old sizes)
            particle_size = 0.5*(log(m)-31);
        }
        if(factor > 1.1) {
            particle_color = new vector3(255.0, 150.0, 150.0);
        }
    }
}

void displayParticlesInTree(oct cOct) {
    if(cOct.data != null) {
        particle cParticle = cOct.data;
        //Display the particle -> marks end of the line for this oct
        displayParticle(cParticle, false);
    }
    for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            displayParticlesInTree(cOct.octs.get(i));
        }
    }
}

void displayParticle(particle cParticle, boolean useVelColor) {
    pushMatrix();
    translate(cParticle.pos.x*realToPixel_conversion, cParticle.pos.y*realToPixel_conversion, cParticle.pos.z*realToPixel_conversion);
    pushStyle();
    if(useVelColor) {
        PVector vCol = fetchVelColor(cParticle, false);
        fill(vCol.x, vCol.y, vCol.z);
    } else {
        fill(cParticle.particle_color.x, cParticle.particle_color.y, cParticle.particle_color.z);
    }
    noStroke();
    sphere(cParticle.particle_size);
    popStyle();
    popMatrix();
}

PVector fetchVelColor(particle cParticle, boolean cameraShift) {
    /*
    . Colours the particle based on its velocity
    . cameraShift = true  => do colors as a red/blue shift from camera
                  = false => do color based on magnitude alone
    */
    if(cParticle.m < pow(10,32)) {
        if(cameraShift) {
            return new PVector(255.0, 255.0, 255.0);
        } else {
            float vel_mag = sqrt( pow(cParticle.vel.x, 2)+ pow(cParticle.vel.y, 2)+ pow(cParticle.vel.z, 2) );
            float factor = 50.0*vel_mag/totalOctLength_Real;    //Factor designed to gives values between 0.0->1.0 for all particles based on speed
            
            //Based on magnitude of velocity
            PVector vcol_mag   = new PVector(factor*255.0, 0.0, (1-factor)*255.0);

            //Based on velocity towards the camera, brightness for magnitude
            float phi   = 2.0*PI*(float(mouseX)/width);   //0->2*PI from 0->width
            float theta = PI*(float(mouseY)/height); //0->PI from 0->height
            PVector camera_UnitPos = new PVector(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
            float vel_alignment = ((camera_UnitPos.x*cParticle.vel.x) +(camera_UnitPos.y*cParticle.vel.y) +(camera_UnitPos.z*cParticle.vel.z))/vel_mag; //-1.0 -> 1.0;  Dot product between vel and -eye pos (eye to origin, NOT origin to eye)
            float vel_align_norm = (1.0+vel_alignment)/2.0;     //Now between 0.0 and 1.0
            PVector vcol_shift = new PVector(255.0*vel_align_norm, 100.0*pow(vel_align_norm,2) ,255.0 -vel_align_norm*255.0);
            
            return vcol_shift;
        }
    } else {
        //Follow this separate rule for VERY large masses (e.g immovable blackhole at galaxy centre)
        return new PVector(100, 60, 60);
    }
}

vector3 fetchParticleColor(float zCoord) {
    /*
    . Returns color for a particle, used to represent its Z-coordinate so it is easier to 
    visualise its placement in 3D space
    */
    float factor = zCoord/totalOctLength_Pixels;
    return new vector3(255*(1.0-factor), 100.0 ,255*(factor));
}