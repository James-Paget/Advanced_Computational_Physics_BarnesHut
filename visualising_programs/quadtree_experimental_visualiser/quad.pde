class quad {
    /*
    0 1
    2 3
    */
    vector3 origin;    //Top left corner
    vector3 dim;       //Full width/height
    particle data;      //Stores particles as its data
    int depth;
    ArrayList<quad> quads = new ArrayList<quad>();

    quad(vector3 origin, vector3 dim, int depth) {
        this.origin = origin;
        this.dim = dim;
        this.depth = depth;             //Purely used for visualisation -> can remove for efficiency
        for(int i=0; i<4; i++) {    //Init quads list
            quads.add(null);        //
        }                           //
    }

    void addParticleToTree(particle cParticle) {
        /*
        . Should create the tree as particles are added
        */
        //Determine region the particle is in
        vector3 quadCentre = new vector3(origin.x +dim.x/2.0, origin.y +dim.y/2.0, origin.z +dim.z/2.0);
        boolean withinX = (origin.x < cParticle.pos.x) && (cParticle.pos.x < origin.x +dim.x); //Check within whole bounds of this quad -> should always be true due to approach from largest quad inwards
        boolean withinY = (origin.y < cParticle.pos.y) && (cParticle.pos.y < origin.y +dim.y); //
        boolean hasQuads = checkQuadHasQuads(this); //If ANY quads are non-null (should imply ALL are non-null in current version of code
        if(withinX && withinY) {
            if(cParticle.pos.x < quadCentre.x) {
                //Left side
                if(cParticle.pos.y < quadCentre.y) {
                    //Upper side
                    //=> Top Left (0)
                    if(data != null) {
                        //HAS data => NO quads => final quad node (full)
                        //=> Split tree, pass to next oriented quad
                        particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                        splitQuad();
                        addParticleToTree(oldData);
                        addParticleToTree(cParticle);
                    } else {
                        if(hasQuads) {
                            //NO data, HAS quads => intermediate quad
                            //=> Pass to next oriented quad
                            quads.get(0).addParticleToTree(cParticle);
                        } else {
                            //NO data, NO quads => final quad node (empty)
                            //=> Give data to this quad
                            data = cParticle;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Left (2)
                    if(data != null) {
                        //HAS data => NO quads => final quad node (full)
                        //=> Split tree, pass to next oriented quad
                        particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                        splitQuad();
                        addParticleToTree(oldData);
                        addParticleToTree(cParticle);
                    } else {
                        if(hasQuads) {
                            //NO data, HAS quads => intermediate quad
                            //=> Pass to next oriented quad
                            quads.get(2).addParticleToTree(cParticle);
                        } else {
                            //NO data, NO quads => final quad node (empty)
                            //=> Give data to this quad
                            data = cParticle;
                        }
                    }
                }
            } else {
                //Right side
                if(cParticle.pos.y < quadCentre.y) {
                    //Upper side
                    //=> Top Right (1)
                    if(data != null) {
                        //HAS data => NO quads => final quad node (full)
                        //=> Split tree, pass to next oriented quad
                        particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                        splitQuad();
                        addParticleToTree(oldData);
                        addParticleToTree(cParticle);
                    } else {
                        if(hasQuads) {
                            //NO data, HAS quads => intermediate quad
                            //=> Pass to next oriented quad
                            quads.get(1).addParticleToTree(cParticle);
                        } else {
                            //NO data, NO quads => final quad node (empty)
                            //=> Give data to this quad
                            data = cParticle;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Right (3)
                    if(data != null) {
                        //HAS data => NO quads => final quad node (full)
                        //=> Split tree, pass to next oriented quad
                        particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                        splitQuad();
                        addParticleToTree(oldData);
                        addParticleToTree(cParticle);
                    } else {
                        if(hasQuads) {
                            //NO data, HAS quads => intermediate quad
                            //=> Pass to next oriented quad
                            quads.get(3).addParticleToTree(cParticle);
                        } else {
                            //NO data, NO quads => final quad node (empty)
                            //=> Give data to this quad
                            data = cParticle;
                        }
                    }
                }
            }
        } else {
            println("-Outside Quad Boundary-");
        }
    }
    void splitQuad() {
        /*
        . Splits this quad into 4 sub-quads AND clears the data from this quad
        */
        //println("--Quad Split at ",quadOrientation,"--");
        data = null;
        for(int i=0; i<quads.size(); i++) {
            //Create quads where split occurs
            quads.remove(i);
            quads.add(i, new quad(getOrigin(i), getDim(), depth+1));
        }
    }
    void generateRootQuad() {
        /*
        . Sets this quad up as a root quad, which covers the entire avalable region
        . Called when a new tree is created
        */
        for(int i=0; i<4; i++) {
            quads.remove(i);
            quads.add(i, new quad(getOrigin(i), getDim(), depth+1));
        }
    }
    vector3 getOrigin(int quadOrientation) {
        if(quadOrientation == 0) {
            return new vector3(origin.x, origin.y, dim.z);
        } else if(quadOrientation == 1) {
            return new vector3(origin.x +dim.x/2.0, origin.y, dim.z);
        } else if(quadOrientation == 2) {
            return new vector3(origin.x, origin.y +dim.y/2.0, dim.z);
        } else {
            return new vector3(origin.x +dim.x/2.0, origin.y +dim.y/2.0, dim.z);
        }
    }
    vector3 getDim() {
        return new vector3(dim.x/2.0, dim.y/2.0, dim.z/2.0);
    }
}

void fetchQuadsWithinRange(ArrayList<quad> nearbyQuads, quad cQuad, vector3 point, float radius) {
    /*
    . Adds to a list (by reference) all quads inside a circular radius of a point
    . Recurrsively called
    p1  p2
    p3  p4
    */
    boolean p1_within = vec_dist( point, new vector3(cQuad.origin.x             , cQuad.origin.y             , cQuad.origin.z) ) < radius;
    boolean p2_within = vec_dist( point, new vector3(cQuad.origin.x +cQuad.dim.x, cQuad.origin.y             , cQuad.origin.z) ) < radius;
    boolean p3_within = vec_dist( point, new vector3(cQuad.origin.x             , cQuad.origin.y +cQuad.dim.y, cQuad.origin.z) ) < radius;
    boolean p4_within = vec_dist( point, new vector3(cQuad.origin.x +cQuad.dim.x, cQuad.origin.y +cQuad.dim.y, cQuad.origin.z) ) < radius;
    if(p1_within || p2_within || p3_within || p4_within) {
        //If AT LEAST 1 point within range, worth still considering
        if(p1_within && p2_within && p3_within && p4_within) {
            //If ALL 4 within range, all children included by default => [Ignore Children]
            boolean hasQuads = checkQuadHasQuads(cQuad);
            if(hasQuads) {
                //If has children, add them all => [Add Children + End of Recurrsion]
                addChildQuadsToSet(nearbyQuads, cQuad);
            } else {
                //If has no children, this is external quad, so just add this one [End of Recurrsion]
                nearbyQuads.add(cQuad);
            }
        } else {
            //Some but not all points included
            boolean hasQuads = checkQuadHasQuads(cQuad);
            if(hasQuads) {
                //If has children, check which of them is external and involved
                for(int i=0; i<cQuad.quads.size(); i++) {
                    if(cQuad.quads.get(i) != null) {
                        //Note; This means null quads are also ignored in search, which is a good thing
                        fetchQuadsWithinRange(nearbyQuads, cQuad.quads.get(i), point, radius);
                    }
                }
            } else {
                //If has NO children, it is external and so should be involved => [End of Recurrsion]
                nearbyQuads.add(cQuad);
            }
        }
    } else {
        //If NO points within range, check for side overlaps
        boolean x_overlap = abs(point.x -(cQuad.origin.x +cQuad.dim.x/2.0)) < (radius +cQuad.dim.x/2.0);
        boolean y_overlap = abs(point.y -(cQuad.origin.y +cQuad.dim.y/2.0)) < (radius +cQuad.dim.y/2.0);
        if(x_overlap && y_overlap) {
            //If there is and X or Y side overlap, then crosses (same as >=1 point within)
            boolean hasQuads = checkQuadHasQuads(cQuad);
            if(hasQuads) {
                //If has children, check which of them is external and involved
                for(int i=0; i<cQuad.quads.size(); i++) {
                    if(cQuad.quads.get(i) != null) {
                        //Note; This means null quads are also ignored in search, which is a good thing
                        fetchQuadsWithinRange(nearbyQuads, cQuad.quads.get(i), point, radius);
                    }
                }
            } else {
                //If has NO children, it is external and so should be involved => [End of Recurrsion]
                nearbyQuads.add(cQuad);
            }
        } else {
            //If not an X or Y overlap either, then unrelated quad => [Ignore Children]
            //Do Nothing ...
        }
    }
}

void addChildQuadsToSet(ArrayList<quad> quadSet, quad cQuad) {
    /*
    . Adds all EXTERNAL child quads to the set, from the root quad given
    . Used for "fetchQuadsWithinRange()"
    */
    for(int i=0; i<cQuad.quads.size(); i++) {
        if(cQuad.quads.get(i) != null) {
            boolean hasQuads = checkQuadHasQuads(cQuad.quads.get(i));
            if(!hasQuads) {
                quadSet.add(cQuad.quads.get(i));
            } else {
                addChildQuadsToSet(quadSet, cQuad.quads.get(i));
            }
        }
    }
}


void displayQuadTree(quad cQuad) {
    /*
    . Displays subsequent quads in the tree
    */
    //If this quad is non-empty, draw it and its contributions from child quads
    if(!checkQuadIsEmpty(cQuad)) {
        displayQuad(cQuad, false);

        for(int i=0; i<cQuad.quads.size(); i++) {
        if(cQuad.quads.get(i) != null) {
            displayQuadTree(cQuad.quads.get(i));
        }
    }
    }
}
void printTreeStructure(quad cQuad, int depth) {
    /*
    . Prints the tree structure from this quad onwards
    */
    String spacing = "";
    for(int i=0; i<depth; i++) {
        spacing += "  ";
    }

    println(spacing+"Depth=",depth,"/ Data; ",cQuad.data);
    println(spacing+"Origin=(",cQuad.origin.x,",",cQuad.origin.y,") // Dim=(",cQuad.dim.x,",",cQuad.dim.y,")");
    for(int i=0; i<cQuad.quads.size(); i++) {
        if(cQuad.quads.get(i) != null) {
            printTreeStructure(cQuad.quads.get(i), depth+1);
        }
    }
    println(spacing+"---");
}
boolean checkQuadIsEmpty(quad cQuad) {
    /*
    . Checks if a quad contains NO quads AND NO data
    */
    boolean isEmpty = true;
    if(cQuad.data != null) {
        isEmpty = false;
    } else {
        for(int i=0; i<cQuad.quads.size(); i++) {
            if(cQuad.quads.get(i) != null) {
                isEmpty = false;
                break;
            }
        }
    }
    return isEmpty;
}
boolean checkQuadHasQuads(quad cQuad) {
    boolean hasQuads = false;
    for(int i=0; i<cQuad.quads.size(); i++) {
        if(cQuad.quads.get(i) != null) {
            hasQuads = true;
            break;
        }
    }
    return hasQuads;
}
quad fetchQuadAtPos(quad cQuad, vector3 pos) {
    /*
    . Returns the quad that the position is located in
    . Searches for the pos from within cQuad => start from rootQuad if 
    you want to search the entire tree, or from a child to search a portion 
    of the tree
    */
    //Determine region the particle is in
    vector3 quadCentre = new vector3(cQuad.origin.x +cQuad.dim.x/2.0, cQuad.origin.y +cQuad.dim.y/2.0, cQuad.origin.z +cQuad.dim.z/2.0);
    boolean withinX = (cQuad.origin.x < pos.x) && (pos.x < cQuad.origin.x +cQuad.dim.x); //Check within whole bounds of this quad -> should always be true due to approach from largest quad inwards
    boolean withinY = (cQuad.origin.y < pos.y) && (pos.y < cQuad.origin.y +cQuad.dim.y); //
    boolean hasQuads = checkQuadHasQuads(cQuad); //If ANY quads are non-null (should imply ALL are non-null in current version of code
    if(withinX && withinY) {
        if(pos.x < quadCentre.x) {
            //Left side
            if(pos.y < quadCentre.y) {
                //Upper side
                //=> Top Left (0)
                if(cQuad.data != null) {
                    //HAS data => NO quads => final quad node (full)
                    //=> Split tree, pass to next oriented quad
                    return cQuad;
                } else {
                    if(hasQuads) {
                        //NO data, HAS quads => intermediate quad
                        //=> Pass to next oriented quad
                        return fetchQuadAtPos(cQuad.quads.get(0), pos);
                    } else {
                        //NO data, NO quads => final quad node (empty)
                        //=> Give data to this quad
                        return cQuad;
                    }
                }
            } else {
                //Lower side
                //=> Bottom Left (2)
                if(cQuad.data != null) {
                    //HAS data => NO quads => final quad node (full)
                    //=> Split tree, pass to next oriented quad
                    return cQuad;
                } else {
                    if(hasQuads) {
                        //NO data, HAS quads => intermediate quad
                        //=> Pass to next oriented quad
                        return fetchQuadAtPos(cQuad.quads.get(2), pos);
                    } else {
                        //NO data, NO quads => final quad node (empty)
                        //=> Give data to this quad
                        return cQuad;
                    }
                }
            }
        } else {
            //Right side
            if(pos.y < quadCentre.y) {
                //Upper side
                //=> Top Right (1)
                if(cQuad.data != null) {
                    //HAS data => NO quads => final quad node (full)
                    //=> Split tree, pass to next oriented quad
                    return cQuad;
                } else {
                    if(hasQuads) {
                        //NO data, HAS quads => intermediate quad
                        //=> Pass to next oriented quad
                        return fetchQuadAtPos(cQuad.quads.get(1), pos);
                    } else {
                        //NO data, NO quads => final quad node (empty)
                        //=> Give data to this quad
                        return cQuad;
                    }
                }
            } else {
                //Lower side
                //=> Bottom Right (3)
                if(cQuad.data != null) {
                    //HAS data => NO quads => final quad node (full)
                    //=> Split tree, pass to next oriented quad
                    return cQuad;
                } else {
                    if(hasQuads) {
                        //NO data, HAS quads => intermediate quad
                        //=> Pass to next oriented quad
                        return fetchQuadAtPos(cQuad.quads.get(3), pos);
                    } else {
                        //NO data, NO quads => final quad node (empty)
                        //=> Give data to this quad
                        return cQuad;
                    }
                }
            }
        }
    } else {
        //When outside boundary for quad
        return null;
    }
}
void displayQuad(quad cQuad, boolean isSolid) {
    pushStyle();
    vector3 quadColor = fetchQuadColor(cQuad.depth);
    noFill();
    if(isSolid) {
        fill(quadColor.x, quadColor.y, quadColor.z);}
    stroke(quadColor.x, quadColor.y, quadColor.z);
    strokeWeight(5);
    rectMode(CORNER);
    rect(cQuad.origin.x, cQuad.origin.y, cQuad.dim.x, cQuad.dim.y);
    popStyle();
}

vector3 fetchQuadColor(int depth) {
    /*
    . Returns color for a quad depth layer
    */
    float factor = (depth/5.0)%1.0;
    return new vector3(255*(1.0-factor),100.0,255*(factor));
}