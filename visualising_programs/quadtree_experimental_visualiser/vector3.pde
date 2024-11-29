class vector3 {
    float x, y, z;

    vector3(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    //pass
}

float vec_mag(vector3 v) {
    return sqrt( pow(v.x,2) + pow(v.y,2) + pow(v.z,2) );
}
float vec_dist(vector3 v1, vector3 v2) {
    return vec_mag( new vector3(v2.x-v1.x, v2.y-v1.y, v2.z-v1.z) );
}