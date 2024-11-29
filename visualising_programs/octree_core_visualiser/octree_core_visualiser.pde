float totalOctLength_Pixels  = 600.0;
float totalOctLength_Real    = 0.25*pow(10, 10);
float realToPixel_conversion = totalOctLength_Pixels / totalOctLength_Real;
oct rootOct;  //The initial node for the oct tree, always starts with 4 non-null components

boolean animateFrames = false;
int animateSpeed = 24;  //Enforced FPS

boolean showAxis = true;
boolean trackCamera = true;     //Just used for visualisation
float mouseZ = 0.0;
float mouseZ_change = 0.0;
float zoomFactor = 1.5;

int currentFrame = 0;

ArrayList<ArrayList<particle>> particles = new ArrayList<ArrayList<particle>>();

void setup() {
    particles = readDataFromFile("data_info.txt", true);    //Include vel or not

    size(600, 600, P3D);
}

void draw() {
    if(animateFrames) {
        frameRate(animateSpeed);
        currentFrame = (currentFrame+1)%particles.size();
    } else {
        frameRate(60);
    }

    calculateCameraTrack();

    background(0);
    
    if(showAxis) {
        displayAxes();}
    displayParticles(particles.get(currentFrame), true);    //Boolean for whether vel is included or not
    if(showAxis) {
        //displayMouseParticle();
        //displayOcts_only(rootOct);
        displayOverlay();}

    /*
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
    */
}

void calculateCameraTrack() {
    if(trackCamera) {
        float phi = 2.0*PI*(float(mouseX)/width);   //0->2*PI from 0->width
        float theta = PI*(float(mouseY)/height); //0->PI from 0->height
        //println("phi= ",phi,",   theta= ",theta);
        camera(
            totalOctLength_Pixels/2.0 +zoomFactor*totalOctLength_Pixels*cos(phi)*sin(theta), totalOctLength_Pixels/2.0 +zoomFactor*totalOctLength_Pixels*sin(phi)*sin(theta), totalOctLength_Pixels/2.0 +zoomFactor*totalOctLength_Pixels*cos(theta),    //Eye pos
            totalOctLength_Pixels/2.0, totalOctLength_Pixels/2.0, totalOctLength_Pixels/2.0,     //Centre pos
            0.0, 0.0, 1.0   //Up direction
        );
    }
}

void displayOverlay() {
    pushStyle();
    textSize(40.0);
    fill(255,255,255);
    text("Current Frame= "+str(currentFrame), 50.0, 50.0);
    popStyle();
}
void displayAxes() {
    pushStyle();
    noFill();
    strokeWeight(10.0);
    //X axis
    stroke(255,0,0);
    line(
        0.0, 0.0, 0.0,
        1.2*totalOctLength_Pixels, 0.0, 0.0
    );
    //Y axis
    stroke(0,255,0);
    line(
        0.0, 0.0, 0.0,
        0.0, 1.2*totalOctLength_Pixels, 0.0
    );
    //Z axis
    stroke(0,0,255);
    line(
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.2*totalOctLength_Pixels
    );
    popStyle();
}
void displayParticles(ArrayList<particle> current_particles, boolean includeVel) {
    for(int i=0; i<current_particles.size(); i++) {
        displayParticle( current_particles.get(i), includeVel );
    }
}
void displayMouseParticle() {
    particle newParticle = new particle(
        1.0,
        new vector3(
            mouseX/realToPixel_conversion, 
            mouseY/realToPixel_conversion, 
            totalOctLength_Pixels/(2.0*realToPixel_conversion)
        ),
        new vector3(
            0.0,
            0.0,
            0.0
        )
    );
    newParticle.particle_size = (0.2*totalOctLength_Real)*realToPixel_conversion;
    displayParticle(newParticle, false);
}
void displayOcts_only(oct cOct) {
    /*
    . Displays oct and all its children but NOT any data within them, just the occupied space of the octs
    */
    if(cOct.data != null) {
        displayOct(cOct, false);                    //Display this oct if exists and is non-empty
    }
    for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            displayOcts_only(cOct.octs.get(i));     //Try do the same for all its children
        }
    }
}

void keyPressed() {
    if(key == '1') {
        currentFrame--;
        if(currentFrame < 0) {
            currentFrame = particles.size()-1;
        }
    }
    if(key == '2') {
        currentFrame++;
        if(currentFrame > particles.size()-1) {
            currentFrame = 0;
        }
    }
    if(key == '3') {
        //Rotate camera with mouse
        trackCamera = !trackCamera;
    }
    if(key == '4') {
        //Causes animation to progress
        animateFrames = !animateFrames;
    }
    if(key == '5') {
        //Reset anim
        currentFrame = 0;
    }
    if(key == '6') {
        println("--Frame Saved--");
        save("barneshut_framesave.png");
    }
    if(key == '7') {
        //Toggle axis display
        showAxis = !showAxis;
    }

    if(key == 'w') {
        zoomFactor += 0.1;
    }
    if(key == 's') {
        zoomFactor -= 0.1;
    }
}