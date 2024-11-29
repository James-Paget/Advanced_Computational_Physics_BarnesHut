quad rootQuad;  //The initial node for the quad tree, always starts with 4 non-null components

ArrayList<quad> nearbyQuads_set = new ArrayList<quad>();
float searchRadius = 50.0;

void setup() {
    rootQuad = new quad(new vector3(0.0, 0.0, 0.0), new vector3(width, height, 0.0), 0);
    rootQuad.generateRootQuad();
    
    initParticles(100);

    //printTreeStructure(rootQuad, 0);  //*Bug-Fixing Tool

    size(600, 600);
}

void draw() {
    background(0);

    displayParticlesInTree(rootQuad);
    displayQuadTree(rootQuad);
    displaySearchArea();
    displayNearbyQuads(nearbyQuads_set);

    quad hoveredQuad = fetchQuadAtPos(rootQuad, new vector3(mouseX, mouseY, 0.0));
    if(hoveredQuad != null) {
        displayQuad(hoveredQuad, true);
    }
}

void displaySearchArea() {
    pushStyle();
    noFill();
    stroke(255);
    strokeWeight(4);
    ellipse(mouseX, mouseY, 2.0*searchRadius, 2.0*searchRadius);
    popStyle();
}
void displayNearbyQuads(ArrayList<quad> quadSet) {
    for(int i=0; i<quadSet.size(); i++) {
        displayQuad(quadSet.get(i), true);
    }
}

void keyPressed() {
    if(key == '1') {
        printTreeStructure(rootQuad, 0);}
    if(key == '2') {
        //Random particle
        particle newParticle = new particle(
            new vector3( 0.8*width*(random(100000))/100000.0, 0.8*height*(random(100000))/100000.0, 0.0 ),
            1.0
        );
        rootQuad.addParticleToTree(newParticle);
    }
    if(key == '3') {
        //Particle at mouse
        particle newParticle = new particle(
            new vector3( mouseX, mouseY, 0.0 ),
            1.0
        );
        rootQuad.addParticleToTree(newParticle);
    }
    if(key == '4') {
        println("Looking for near quads...");
        nearbyQuads_set = new ArrayList<quad>();
        fetchQuadsWithinRange(nearbyQuads_set, rootQuad, new vector3(mouseX, mouseY, 0), searchRadius);
    }
}