float totalOctLength = 600.0;
oct rootOct;  //The initial node for the oct tree, always starts with 4 non-null components

boolean trackCamera = true;     //JUst used for visualisation
float mouseZ = 0.0;
float mouseZ_change = 0.0;

void setup() {
    rootOct = new oct(new vector3(0.0, 0.0, 0.0), new vector3(totalOctLength, totalOctLength, totalOctLength), 0);
    rootOct.generateRootOct();
    
    initParticles(100);

    printTreeStructure(rootOct, 0);  //*Bug-Fixing Tool

    size(600, 600, P3D);
}

void draw() {
    if(trackCamera) {
        float phi = 2.0*PI*(float(mouseX)/width);   //0->2*PI from 0->width
        float theta = PI*(float(mouseY)/height); //0->PI from 0->height
        //println("phi= ",phi,",   theta= ",theta);
        camera(
            totalOctLength/2.0 +1.75*totalOctLength*cos(phi)*sin(theta), totalOctLength/2.0 +1.75*totalOctLength*sin(phi)*sin(theta), totalOctLength/2.0 +1.75*totalOctLength*cos(theta),    //Eye pos
            totalOctLength/2.0, totalOctLength/2.0, totalOctLength/2.0,     //Centre pos
            0.0, 0.0, 1.0   //Up direction
        );
    }

    background(0);
    
    displayAxes();
    displayParticlesInTree(rootOct);
    displayOctTree(rootOct);

    //Draw a particle for the mouse --> Used to test "fetchOctAtPos()"
    mouseZ += mouseZ_change;
    particle hoverParticle = new particle(
        new vector3(mouseX, mouseY, mouseZ),
        1.0
    );
    hoverParticle.particle_color = new vector3(200, 200, 0);
    displayParticle(hoverParticle);
    oct hoveredOct = fetchOctAtPos(rootOct, new vector3(mouseX, mouseY, mouseZ));
    if(hoveredOct != null) {
        displayOct(hoveredOct, true);
    }
}

void displayAxes() {
    pushStyle();
    noFill();
    strokeWeight(10.0);
    //X axis
    stroke(255,0,0);
    line(
        0.0, 0.0, 0.0,
        1.2*totalOctLength, 0.0, 0.0
    );
    //Y axis
    stroke(0,255,0);
    line(
        0.0, 0.0, 0.0,
        0.0, 1.2*totalOctLength, 0.0
    );
    //Z axis
    stroke(0,0,255);
    line(
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.2*totalOctLength
    );
    popStyle();
}

void keyPressed() {
    if(key == '1') {
        printTreeStructure(rootOct, 0);}
    if(key == '2') {
        //Random particle
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
    if(key == '3') {
        //Particle at mouse
        particle newParticle = new particle(
            new vector3( mouseX, mouseY, 0.0 ),
            1.0
        );
        rootOct.addParticleToTree(newParticle);
    }
    if(key == '4') {
        //Rotate camera with mouse
        trackCamera = !trackCamera;
    }
    if(key == '6') {
        println("--Frame Saved--");
        save("framesave.png");
    }

    if(key == 'w') {
        mouseZ_change = 2.0;}
    if(key == 's') {
        mouseZ_change = -2.0;}
}
void keyReleased() {
    if( (key == 'w') || key == 's' ) {
        mouseZ_change = 0.0;
    }
}
