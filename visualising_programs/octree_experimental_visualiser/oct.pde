class oct {
    /*
    Upper Layer     Lower Layer
    0 1             4 5
    2 3             6 7
    */
    vector3 origin;    //Top left corner
    vector3 dim;       //Full width/height
    particle data;      //Stores particles as its data
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

    void addParticleToTree(particle cParticle) {
        /*
        . Should create the tree as particles are added
        */
        //Determine region the particle is in
        vector3 octCentre = new vector3(origin.x +dim.x/2.0, origin.y +dim.y/2.0, origin.z +dim.z/2.0);
        boolean withinX = (origin.x < cParticle.pos.x) && (cParticle.pos.x < origin.x +dim.x); //Check within whole bounds of this oct -> should always be true due to approach from largest oct inwards
        boolean withinY = (origin.y < cParticle.pos.y) && (cParticle.pos.y < origin.y +dim.y); //
        boolean withinZ = (origin.z < cParticle.pos.z) && (cParticle.pos.z < origin.z +dim.z); //
        boolean hasOcts = checkOctHasOcts(this); //If ANY octs are non-null (should imply ALL are non-null in current version of code
        if(withinX && withinY && withinZ) {
            if(cParticle.pos.z < octCentre.z) {
                //In upper layer
                if(cParticle.pos.x < octCentre.x) {
                    //Left side
                    if(cParticle.pos.y < octCentre.y) {
                        //Upper side
                        //=> Top Left (0)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(0).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    } else {
                        //Lower side
                        //=> Bottom Left (2)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(2).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    }
                } else {
                    //Right side
                    if(cParticle.pos.y < octCentre.y) {
                        //Upper side
                        //=> Top Right (1)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(1).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    } else {
                        //Lower side
                        //=> Bottom Right (3)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(3).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    }
                }
            } else {
                //In lower layer
                if(cParticle.pos.x < octCentre.x) {
                    //Left side
                    if(cParticle.pos.y < octCentre.y) {
                        //Upper side
                        //=> Top Left (4)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(4).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    } else {
                        //Lower side
                        //=> Bottom Left (6)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(6).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    }
                } else {
                    //Right side
                    if(cParticle.pos.y < octCentre.y) {
                        //Upper side
                        //=> Top Right (5)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(5).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    } else {
                        //Lower side
                        //=> Bottom Right (7)
                        if(data != null) {
                            //HAS data => NO octs => final oct node (full)
                            //=> Split tree, pass to next oriented oct
                            particle oldData = new particle(new vector3(data.pos.x, data.pos.y, data.pos.z), 1.0);//data;
                            splitOct();
                            addParticleToTree(oldData);
                            addParticleToTree(cParticle);
                        } else {
                            if(hasOcts) {
                                //NO data, HAS octs => intermediate oct
                                //=> Pass to next oriented oct
                                octs.get(7).addParticleToTree(cParticle);
                            } else {
                                //NO data, NO octs => final oct node (empty)
                                //=> Give data to this oct
                                data = cParticle;
                            }
                        }
                    }
                }
            }
        } else {
            println("-Outside Oct Boundary-");
        }
    }
    void splitOct() {
        /*
        . Splits this oct into 8 sub-octs AND clears the data from this oct
        */
        //println("--Oct Split at ",octOrientation,"--");
        data = null;
        for(int i=0; i<octs.size(); i++) {
            //Create octs where split occurs
            octs.remove(i);
            octs.add(i, new oct(getOrigin(i), getDim(), depth+1));
        }
    }
    void generateRootOct() {
        /*
        . Sets this oct up as a root oct, which covers the entire avalable region
        . Called when a new tree is created
        */
        for(int i=0; i<8; i++) {
            octs.remove(i);
            octs.add(i, new oct(getOrigin(i), getDim(), depth+1));
        }
    }
    vector3 getOrigin(int octOrientation) {
        if(octOrientation == 0) {
            return new vector3(origin.x, origin.y, origin.z);
        } else if(octOrientation == 1) {
            return new vector3(origin.x +dim.x/2.0, origin.y, origin.z);
        } else if(octOrientation == 2) {
            return new vector3(origin.x, origin.y +dim.y/2.0, origin.z);
        } else if(octOrientation == 3) {
            return new vector3(origin.x +dim.x/2.0, origin.y +dim.y/2.0, origin.z);
        } else if(octOrientation == 4) {
            return new vector3(origin.x, origin.y, origin.z +dim.z/2.0);
        } else if(octOrientation == 5) {
            return new vector3(origin.x +dim.x/2.0, origin.y, origin.z +dim.z/2.0);
        } else if(octOrientation == 6) {
            return new vector3(origin.x, origin.y +dim.y/2.0, origin.z +dim.z/2.0);
        } else {
            return new vector3(origin.x +dim.x/2.0, origin.y +dim.y/2.0, origin.z +dim.z/2.0);
        }
    }
    vector3 getDim() {
        return new vector3(dim.x/2.0, dim.y/2.0, dim.z/2.0);
    }
}

void displayOctTree(oct cOct) {
    /*
    . Displays subsequent octs in the tree
    */
    //If this oct is non-empty, draw it and its contributions from child octs
    if(!checkOctIsEmpty(cOct)) {
        displayOct(cOct, false);

        for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            displayOctTree(cOct.octs.get(i));
        }
    }
    }
}
void printTreeStructure(oct cOct, int depth) {
    /*
    . Prints the tree structure from this oct onwards
    */
    String spacing = "";
    for(int i=0; i<depth; i++) {
        spacing += "  ";
    }

    println(spacing+"Depth=",depth,"/ Data; ",cOct.data);
    println(spacing+"Origin=(",cOct.origin.x,",",cOct.origin.y,",",cOct.origin.z,") // Dim=(",cOct.dim.x,",",cOct.dim.y,",",cOct.dim.z,")");
    for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            printTreeStructure(cOct.octs.get(i), depth+1);
        }
    }
    println(spacing+"---");
}
boolean checkOctIsEmpty(oct cOct) {
    /*
    . Checks if a oct contains NO octs AND NO data
    */
    boolean isEmpty = true;
    if(cOct.data != null) {
        isEmpty = false;
    } else {
        for(int i=0; i<cOct.octs.size(); i++) {
            if(cOct.octs.get(i) != null) {
                isEmpty = false;
                break;
            }
        }
    }
    return isEmpty;
}
boolean checkOctHasOcts(oct cOct) {
    boolean hasOcts = false;
    for(int i=0; i<cOct.octs.size(); i++) {
        if(cOct.octs.get(i) != null) {
            hasOcts = true;
            break;
        }
    }
    return hasOcts;
}
oct fetchOctAtPos(oct cOct, vector3 pos) {
    /*
    . Returns the oct that the position is located in
    . Searches for the pos from within cOct => start from rootOct if 
    you want to search the entire tree, or from a child to search a portion 
    of the tree
    */
    //Determine region the particle is in
    vector3 octCentre = new vector3(cOct.origin.x +cOct.dim.x/2.0, cOct.origin.y +cOct.dim.y/2.0, cOct.origin.z +cOct.dim.z/2.0);
    boolean withinX = (cOct.origin.x < pos.x) && (pos.x < cOct.origin.x +cOct.dim.x); //Check within whole bounds of this oct -> should always be true due to approach from largest oct inwards
    boolean withinY = (cOct.origin.y < pos.y) && (pos.y < cOct.origin.y +cOct.dim.y); //
    boolean withinZ = (cOct.origin.z < pos.z) && (pos.z < cOct.origin.z +cOct.dim.z); //
    boolean hasOcts = checkOctHasOcts(cOct); //If ANY octs are non-null (should imply ALL are non-null in current version of code
    if(withinX && withinY && withinZ) {
        if(pos.z < octCentre.z) {
            //Upper layer
            if(pos.x < octCentre.x) {
                //Left side
                if(pos.y < octCentre.y) {
                    //Upper side
                    //=> Top Left (0)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(0), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Left (2)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(2), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                }
            } else {
                //Right side
                if(pos.y < octCentre.y) {
                    //Upper side
                    //=> Top Right (1)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(1), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Right (3)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(3), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                }
            }
        } else {
            //Lower layer
            if(pos.x < octCentre.x) {
                //Left side
                if(pos.y < octCentre.y) {
                    //Upper side
                    //=> Top Left (4)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(4), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Left (6)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(6), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                }
            } else {
                //Right side
                if(pos.y < octCentre.y) {
                    //Upper side
                    //=> Top Right (5)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(5), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                } else {
                    //Lower side
                    //=> Bottom Right (7)
                    if(cOct.data != null) {
                        //HAS data => NO octs => final oct node (full)
                        //=> Split tree, pass to next oriented oct
                        return cOct;
                    } else {
                        if(hasOcts) {
                            //NO data, HAS octs => intermediate oct
                            //=> Pass to next oriented oct
                            return fetchOctAtPos(cOct.octs.get(7), pos);
                        } else {
                            //NO data, NO octs => final oct node (empty)
                            //=> Give data to this oct
                            return cOct;
                        }
                    }
                }
            }
        }
    } else {
        //When outside boundary for oct
        return null;
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