std_msgs/Header header

# External robot-vs-human body/capsule pairs.
int32[] robot_body_index
int32[] human_capsule_index
float32[] clearance

# Internal robot sphere-vs-sphere pairs, used in sphere collision mode.
int32[] internal_body_a_index
int32[] internal_sphere_a_index
int32[] internal_body_b_index
int32[] internal_sphere_b_index
float32[] internal_clearance
