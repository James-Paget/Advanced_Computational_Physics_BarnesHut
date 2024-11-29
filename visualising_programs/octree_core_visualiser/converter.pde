String octID = "";
String octreeText = "";

ArrayList<ArrayList<particle>> readDataFromFile(String filename, boolean includeVel) {
    /*
    . Looks at the "data_info.txt" file to read the data for each frame of calculation
    . In this case, this si the position and masses of each particle
    */
    String[] lines = loadStrings(filename);
    ArrayList<ArrayList<particle>> particles = new ArrayList<ArrayList<particle>>();
    for(int line_index=0; line_index<lines.length; line_index++) {
        String line = lines[line_index];
        int start_index = 0;
        ArrayList<Float> particle_info = new ArrayList<Float>();
        ArrayList<particle> line_particles = new ArrayList<particle>();
        for(int char_index=0; char_index<line.length(); char_index++) {
            String character = line.substring(char_index, char_index+1);
            if(character.equals(",")) {
                particle_info.add(float(line.substring(start_index, char_index)));
                start_index = char_index +1;
            } else if(character.equals("*")) {
                particle_info.add(float(line.substring(start_index, char_index)));  //Make sure to trigger this final data pull
                if(includeVel) {
                    //Denser file, but nicer visualisation (with vel data)
                    particle newParticle = new particle(
                        particle_info.get(0), 
                        new vector3(particle_info.get(1), particle_info.get(2), particle_info.get(3)),  
                        new vector3(particle_info.get(4), particle_info.get(5), particle_info.get(6))
                    ); 
                    line_particles.add(newParticle);
                }
                else {
                    //More compact file, but worse visualisation (no vel data)
                    particle newParticle = new particle(
                        particle_info.get(0), 
                        new vector3(particle_info.get(1), particle_info.get(2), particle_info.get(3)),  
                        new vector3(0.0, 0.0, 0.0)
                    );
                    line_particles.add(newParticle);
                }
                particle_info = new ArrayList<Float>();
                start_index = char_index +1;
            }
        }
        particles.add(line_particles);
    }
    return particles;
}
void readOctsFromFile() {
    /*
    . Looks at the "octs_info.txt" file to read the oct structure used for each frame of calculation
    */
}
