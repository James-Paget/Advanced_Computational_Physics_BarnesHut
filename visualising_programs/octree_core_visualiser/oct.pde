class oct {
    /*
    Upper Layer     Lower Layer
    0 1             4 5
    2 3             6 7
    */
    vector3 origin;    //Top left corner
    vector3 dim;       //Full width/height
    particle data;     //Stores particles as its data
    int depth;
    ArrayList<oct> octs = new ArrayList<oct>();

    oct(vector3 origin, vector3 dim, int depth) {
        this.origin = origin;
        this.dim = dim;
        this.depth = depth;         //Purely used for visualisation -> can remove for efficiency
        for(int i=0; i<8; i++) {    //Init octs list
            octs.add(null);         //
        }                           //
    }
}
void displayOct(oct cOct, boolean isSolid) {
    pushMatrix();
    //** Note; offset to draw from corner
    translate(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z +cOct.dim.z/2.0);
    pushStyle();
    vector3 octColor = fetchOctColor(cOct.depth);
    noFill();
    if(isSolid) {
        fill(octColor.x, octColor.y, octColor.z);}
    stroke(octColor.x, octColor.y, octColor.z);
    strokeWeight(5);
    rectMode(CORNER); 
    box(cOct.dim.x, cOct.dim.y, cOct.dim.z);
    popStyle();
    popMatrix();
}

vector3 fetchOctColor(int depth) {
    /*
    . Returns color for a oct depth layer
    */
    float factor = (depth/5.0)%1.0;
    return new vector3(255*(1.0-factor),100.0,255*(factor));
}