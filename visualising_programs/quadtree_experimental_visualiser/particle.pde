class particle {
    vector3 pos;
    float m;

    float particle_size = 10.0;
    PVector particle_color = new PVector(250,250,250);

    particle(vector3 pos, float m) {
        this.pos = pos;
        this.m = m;
    }

    //pass
}

void initParticles(int n) {
    for(int i=0; i<n; i++) {
        particle newParticle = new particle(
            new vector3( 0.8*width*(random(100000))/100000.0, 0.8*height*(random(100000))/100000.0, 0.0 ),
            1.0
        );
        rootQuad.addParticleToTree(newParticle);
    }
}
void displayParticlesInTree(quad cQuad) {
    if(cQuad.data != null) {
        //Display the particle -> marks end of the line for this quad
        particle cParticle = cQuad.data;
        fill(cParticle.particle_color.x, cParticle.particle_color.y, cParticle.particle_color.z);
        noStroke();
        ellipse(cParticle.pos.x, cParticle.pos.y, cParticle.particle_size, cParticle.particle_size);
    }
    for(int i=0; i<cQuad.quads.size(); i++) {
        if(cQuad.quads.get(i) != null) {
            displayParticlesInTree(cQuad.quads.get(i));
        }
    }
}