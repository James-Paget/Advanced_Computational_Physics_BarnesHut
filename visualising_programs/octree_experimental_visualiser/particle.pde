class particle {
    vector3 pos;
    float m;

    float particle_size = 10.0;
    vector3 particle_color = new vector3(255.0, 255.0, 255.0);//fetchParticleColor(pos.z);

    particle(vector3 pos, float m) {
        this.pos = pos;
        this.m = m;
    }

    //pass
}

void initParticles(int n) {
    for(int i=0; i<n; i++) {
        particle newParticle = new particle(
            new vector3(
                0.8*totalOctLength*(random(100000))/100000.0, 
                0.8*totalOctLength*(random(100000))/100000.0,
                0.8*totalOctLength*(random(100000))/100000.0
            ),
            1.0
        );
        rootOct.addParticleToTree(newParticle);
    }
}
void displayParticlesInTree(oct cOct) {
    if(cOct.data != null) {
        particle cParticle = cOct.data;
        //Display the particle -> marks end of the line for this oct
        displayParticle(cParticle);
    }
    for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            displayParticlesInTree(cOct.octs.get(i));
        }
    }
}

void displayParticle(particle cParticle) {
    pushMatrix();
    translate(cParticle.pos.x, cParticle.pos.y, cParticle.pos.z);
    pushStyle();
    fill(cParticle.particle_color.x, cParticle.particle_color.y, cParticle.particle_color.z);
    noStroke();
    sphere(cParticle.particle_size);
    popStyle();
    popMatrix();
}

vector3 fetchParticleColor(float zCoord) {
    /*
    . Returns color for a particle, used to represent its Z-coordinate so it is easier to 
    visualise its placement in 3D space
    */
    float factor = zCoord/totalOctLength;
    return new vector3(255*(1.0-factor), 100.0 ,255*(factor));
}