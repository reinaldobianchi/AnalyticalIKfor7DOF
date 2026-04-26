import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar

##################################################################################################
#                           Direct Kinematics 
##################################################################################################

# DH homogeneous transformation matrix function
def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH Parameters for the DLR 7 DOF arm (theta, d, a, alpha)
dh_params = [
    (0,    0.0,  0.0, -np.pi/2),
    (0,    0.0,  0.0,  np.pi/2),
    (0,    0.4,  0.0,  np.pi/2),
    (0,    0.0,  0.0, -np.pi/2),
    (0,    0.4,  0.0, -np.pi/2),
    (0,    0.0,  0.0,  np.pi/2),
    (0,    0.126,  0.0,  0.0)
]

# Forward Kinematics
def forward_kinematics(dh_params, thetas):
    Tw = np.eye(4)
    T = np.eye(4)
    positions = [T[:3, 3]]                                      # Initial position (base)
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        Ti = dh_matrix(thetas[i], d, a, alpha)
        T = T @ Ti
        positions.append(T[:3, 3])
        # print("T")
        if i == 3:
            Tw = T
    return np.array(positions), Tw



# INVERSE KINEMATICS: Q1, Q2, Q3 and Q4 (using Pw)
def inverse_kinematics_from_points(Ps, Pe, Pw, t5, t6, t7):    
    q = np.zeros(7)

    ########       Calculation of angles q1 and q2              #######
    # print("\n     2.1 - Calculation of q1 and q2")
    if Pe[0]==0 and Pe[1] == 0:
        # print("            Special condition Pex = Pey = 0")
        q1 = np.arctan2(Pe[1],Pe[0])
        # print("            q1 = %f rd" % q1)
        junta1 = (q1*180)/np.pi
        # print("            q1 = %f gr" % junta1)
        q2 = 0
        # print("\n            q2 = %f rd" % q2)
        junta2 = (q2*180)/np.pi
        # print("            q2 = %f gr" % junta2)
    else:
        # print("\n            Normal condition Pex and Pey != 0")
        q1 = np.arctan2(Pe[1],Pe[0])
        # print("            q1 = %f rd" % q1)
        junta1 = (q1*180)/np.pi
        # print("            q1 = %f gr" % junta1)

        q2 = np.arctan2(np.sqrt(np.power(Pe[1],2) + np.power(Pe[0],2)), Pe[2])
        # print("\n            q2 = %f rd" % q2)
        junta2 = (q2*180)/np.pi
        # print("            q2 = %f gr" % junta2)




    ########             # Calculation of q4 (using Pw and Ps)        ########
    # print("\n    2.2 - Calculation of q4 - Law of Cosines - Triangle geometry")
    v = Pw - Ps
    d = np.linalg.norm(v)
    a = (d**2 -L1**2 - L2**2) / (-2 * L1 * L2)                                              # Law of Cosines
    arco_q4 = np.arccos(a)

    q4 = np.pi - arco_q4
    # print("            q4 = %f rd" % q4)
    junta4 = (q4*180)/np.pi
    # print("            q4 = %f gr" % junta4)



    ########                     Calculation of q3                #######
    # print("\n    2.3 - Calculation of q3 (setting q3 = 0 to find comparison vector)")
    # Calculation of Pei (imaginary)
    # Direction vector (difference between points)
    v = Pe - Ps
    norma = np.linalg.norm(v)
    v_unitario = v / norma

    h =  L2 * np.cos(q4)                                                                    # triangle height
    reta_total = L1 + h                                                                     # total arm length + triangle height

    tempo = 0
    tamanho_reta = 0
    while tamanho_reta < reta_total:                                                        # calculating the position of Pei
        Pei_x = Ps[0] + v_unitario[0] * tempo
        Pei_y = Ps[1] + v_unitario[1] * tempo
        Pei_z = Ps[2] + v_unitario[2] * tempo
        tamanho_reta = np.sqrt( ((Ps[0]-Pei_x)**2) + ((Ps[1]-Pei_y)**2)  +  ((Ps[2]-Pei_z)**2)   )
        tempo = tempo + 0.001
    Pei = np.array([Pei_x,Pei_y,Pei_z])                                                     # position Pei

    # --- Circumference Data ---
    C = Pei                                                                                 # center
    # r = np.sqrt( ((Pw[0]-Pei[0])**2) + ((Pw[1]-Pei[1])**2)  +  ((Pw[2]-Pei[2])**2) )        # radius
    # n = np.array([0.0, 0.0, 1.0])                                                           # normal vector of the plane (here, XY plane)

    # Points on the circumference

    # Simulating with theta3 = 0
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = np.radians([junta1, junta2, 0, junta4, 0, 0, 0])
    theta_test = [theta1, theta2, theta3, -theta4, theta5, theta6, theta7]

    points_Pw_zero, _ = forward_kinematics(dh_params, theta_test)
    
    Pe = np.array(points_Pw_zero[4])
    Pw_zero = np.array(points_Pw_zero[5])
    # print("PW_zero =", Pw_zero)

    # plot_3d(points_Pw_zero,'Setting q3 to zero to calculate angle')

    #################################
    P1 = Pw_zero
    P2 = Pw
    # --- Radial vectors ---
    v1 = P1 - C
    v2 = P2 - C
    # --- Dot and Cross product ---
    dot = np.dot(v1, v2)
    # cross = np.cross(v1, v2)

    # --- Unsigned angle ---
    q3 = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # --- Convert to degrees ---
    junta3 = np.round(np.degrees(q3),2)
    # print("            q3 = %f rad" % q3)
    # print("            q3 = %f gr" % junta3)


    # print("\n    2.4 - Calculation of Q5, Q6 and Q7 (Euler)")

    # print("\n          2.4.1 - Using Tw4 (q1...q4)")
    q[0] = q1
    q[1] = q2
    q[2] = q3
    q[3] = q4

    q[4] = t5
    q[5] = t6
    q[6] = t7

    # print('q =', q)

    # -------------------------
    # Calculate T04
    # -------------------------
    T = np.eye(4)
    for i in range(4):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)

    T04 = T.copy()                                                       # T0_4

    # -------------------------
    # Calculate T47
    # -------------------------
    T = np.eye(4)
    for i in range(4, 7):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)

    T47 = T.copy()                                                      # T4_7

    # -------------------------
    # Multiplication
    # -------------------------
    T07 = T04 @ T47                                                     # T0_7

    # -------------------------
    # Extracting R47 via correct definition
    # -------------------------
    R04 = T04[:3,:3]                                                    # get orientation from 0 to 4
    R07 = T07[:3,:3]                                                    # get orientation from 4 to 7

    R47 = R04.T @ R07                                                   # calculate the target

    q5_calc, q6_calc, q7_calc = wrist_ik_general(R47)

    q[0] = q1
    q[1] = q2
    q[2] = q3
    q[3] = q4
    q[4] = q5_calc
    q[5] = q6_calc
    q[6] = q7_calc
    
    return q


# -------------------------------------------------------------------------
# Option A – General Procedure:
# Calculates q5, q6, q7 from residual orientation (after q1–q4)
# -------------------------------------------------------------------------
def wrist_ik_general(R):                                            # General Procedure

    s6 = np.sqrt(R[2,0]**2 + R[2,1]**2)
    q6 = np.arctan2(s6, R[2,2])

    if np.isclose(s6, 0):
        print("Singularity detected")
        q5 = 0
        q7 = np.arctan2(-R[0,1], R[0,0])
        return q5, q6, q7

    q7 = np.arctan2(R[2,1], -R[2,0])
    q5 = np.arctan2(R[1,2], R[0,2])

    return q5, q6, q7

def calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi):
    v = Pw - Ps
    d = np.linalg.norm(v)
    n = v / d
 
    m = Pwa - Ps
    da = np.linalg.norm(m)
    a = m / da    
    vetor_a_transposto = np.transpose(a)                           # using the np.transpose() method for transpose matrix to calculate the dot product
    alpha = np.dot(n, vetor_a_transposto) 
    beta = alpha * n
    gama = a - beta
    omega = np.linalg.norm(gama)
    u = gama / omega
    v = np.cross(n, u) 
    # triangle geometry
    a = (L1**2 - L2**2 + d**2) / (2 * d)
    R = np.sqrt(max(L1**2 - a**2, 0))                               # distance between Pe and hypotenuse, which will be the radius of the circumference        
    psi_graus = (psi*180)/np.pi   
    cos_alpha = (L2**2 - L1**2 - d**2) / (-2 * L1 * d)              # law of cosines
    Pc = Ps + L1 * cos_alpha * n    
    Pe = Pc + R * (np.cos(psi) * u + np.sin(psi) * v)    
    return Pe


# (ELLIPSOID) Special Jacobian based on swivel angle (ψ) in a humanoid robotic arm (7 DOF, shoulder–elbow–wrist).
def compute_jacobiano(psi):
    p_elbow = calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi)         # calculate Pe
    z1 = np.cross(p_elbow - p_shoulder, [1, 0, 0])                  # axis associated with the shoulder. Creates the first Jacobian vector associated with shoulder rotation.
    z2 = np.cross(p_wrist - p_elbow, [0, 1, 0])                     # axis associated with the elbow. Represents the elbow rotation.
    z3 = np.cross(p_wrist - p_elbow, [0, 0, 1])                     # axis associated with the wrist. Represents the effect of wrist rotation.
    return np.vstack([z1, z2, z3]).T                                # Stacks the vectors z1, z2, z3 as columns. Result: a 3×3 matrix

def compute_jacobian(q):    
    delta = 1e-5
    J = np.zeros((3, 7))
    T_base = forward_kinematics1(q)                                  # End-effector position with joints q
    p0 = T_base[:3, 3]                                               # Extract only the position (x,y,z) of the end-effector
    for i in range(7):
        dq = np.copy(q)                                              # Create a copy of q → dq (preserving the original q)
        dq[i] += delta                                               # Increment only joint i by δ
        T_i = forward_kinematics1(dq)                                # Recalculate end-effector position only with change in joint i
        pi = T_i[:3, 3]                                              # Extract linear position (x,y,z)
        J[:, i] = (pi - p0) / delta                                  # Calculate Jacobian and store in matrix J at joint i position
    return J

def forward_kinematics1(q):
    q1, q2, q3, q4, q5, q6, q7 = q
    R_shoulder = R.from_euler('zyx', [q1, q2, q3]).as_matrix()
    p_elbow = R_shoulder @ np.array([0, 0, L1])
    R_elbow = R.from_rotvec([0, q4, 0]).as_matrix()
    R_wrist = R.from_euler('zyx', [q5, q6, q7]).as_matrix()
    R_hand = R_shoulder @ R_elbow @ R_wrist
    p_wrist = p_elbow + (R_shoulder @ R_elbow @ np.array([0, 0, L2]))

    T = np.eye(4)
    T[:3, :3] = R_hand
    T[:3, 3] = p_wrist
    return T

def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))

def optimize_swivel(p_shoulder, p_wrist, L1, L2):
    Pwa =np.array([p_wrist[0], p_wrist[1], 0])
    values = []
    def cost(psi):
        try:
            p_elbow = calc_elbow_position(p_shoulder, p_wrist, Pwa, L1, L2, psi)                                # SWIVEL ANGLE
            q = inverse_kinematics_from_points(p_shoulder, p_elbow, p_wrist, Tw, theta5, theta6, theta7)        # IK
            J = compute_jacobian(q)                                                                             # MANIPULABILITY - JACOBIAN
            w = manipulability(J)                                                                               # MANIPULABILITY - ELLIPSOID
            values.append((psi, w))
            return -w
        except:
            return np.inf

    result = minimize_scalar(cost, bounds=(0, np.pi), method='bounded')
    return result.x, -result.fun, values


# Plotting the 3D Graph
def plot_3d(positions, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', linewidth=3, markersize=8, color='blue')

    # Adjust graph limits
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                          positions[:, 1].max()-positions[:, 1].min(),
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # ax.set_title('DLR 7 DOF Robotic Arm - Forward Kinematics')
    ax.set_title(title)
    ax.grid(True)
    # ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
    plt.show()

#########################################################################################################################
###########################                         START                        ########################################
#########################################################################################################################

############################ TEST: TYPE 1 - Simple reach movement for a target     ###############################

def psi_humano(pw):
    # Coefficients from the article/your regression
    a, b, c = 0.5, 0.9, 0.0

    # Type 1: Cup in front
    # pose_type1 = np.array([0.50, 0.00, 0.20])
    pose_type1 = pw

    # 1. Extrinsics
    R = np.linalg.norm(pose_type1)
    chi = np.arctan2(pose_type1[1], pose_type1[0])  # ~0 rad (front)
    psi = np.arccos(pose_type1[2] / R)             # ~1.0 rad

    # 2. Anthropomorphic swivel
    phi_s = a * chi + b * psi + c  # ~0.9 rad (neutral elbow)

    print(f"Type 1 - Expected ϕ_s: {np.degrees(phi_s):.0f}°")
    # Output: ~52° (typical human frontal reach)
    return phi_s

def swivel_manip(Pw):
    Ps = np.array([0., 0., 0.])
    L1 = dh_params[2][1]
    L2 = dh_params[4][1] 
    best_psi, max_w, _ = optimize_swivel(Ps, Pw, L1, L2)            
    return best_psi, max_w

traj = np.linspace(
    np.array([0.4,0.1,0.5]),
    np.array([0.7,0.3,0.6]),
    20
)

psi_h_list = []
psi_m_list = []
delta_psi = []

for pw in traj:
    
    # Human ψ
    psi_h = psi_humano(pw)
    
    # Manipulability ψ
    psi_m, w_m = swivel_manip(pw)
    
    psi_h_list.append(psi_h)
    psi_m_list.append(psi_m)
    delta_psi.append(psi_m - psi_h)
    print("difference =", ((psi_m-psi_h)/psi_h)*100 )

# Single plot (without subplots or specific colors)
plt.figure()
# plt.plot(psi_h_list)
# plt.plot(psi_m_list)

plt.plot(psi_h_list, label='ψ_h (human)', linewidth=3.2)
plt.plot(psi_m_list, label='ψ_m (manipulability)', linewidth=3.2)

plt.xlabel("Point in the trajectory (Type 1)")
plt.ylabel("angle ψ (rad)")
plt.title("Comparison: ψ human vs ψ manipulability")
plt.legend()
plt.show()

# Single plot (without specific colors or subplots)
plt.figure()
plt.plot(delta_psi, linewidth=3.2)
plt.xlabel("Point in the trajectory (Type 1)")
plt.ylabel("Δψ (rad)")
plt.title("Error Δψ throughout the journey")
plt.show()


#########################################################################################################################
#####################################       Trajectory           ##################################################
#########################################################################################################################
Ps = np.array([0., 0., 0.])
L1 = dh_params[2][1]
L2 = dh_params[4][1]

Pw_a = traj[0]
psi_m, _ = swivel_manip(Pw_a)
Pwa_a = np.array([Pw_a[0], Pw_a[1], 0])    
Pe_a = calc_elbow_position(Ps, Pw_a, Pwa_a, L1, L2, -psi_m)
qa = inverse_kinematics_from_points(Ps, Pe_a, Pw_a, 0, 0, 0)
theta_values_a = [qa[0], qa[1], qa[2], -qa[3], qa[4], qa[5], qa[6]]


Pw_b = traj[19]
psi_m, _ = swivel_manip(Pw_b)
Pwa_b =np.array([Pw_b[0], Pw_b[1], 0])    
Pe_b = calc_elbow_position(Ps, Pw_b, Pwa_b, L1, L2, -psi_m)
qb = inverse_kinematics_from_points(Ps, Pe_b, Pwa_b, 0, 0, 0)
theta_values_b = [qb[0], qb[1], qb[2], -qb[3], qb[4], qb[5], qb[6]]

#########################################################################################################
#                                       POSITION TESTS
#########################################################################################################

steps = 20
trajectory = np.linspace(theta_values_a, theta_values_b, steps)
# print("len =", len(trajectory))

CD_theta_group = []
IK_theta_group = []

CD_eta_group = []
IK_eta_group = []

CD_alpha_group = []
IK_alpha_group = []

CD_beta_group = []
IK_beta_group = []

for i in trajectory:
    
    # print ("i =", i)
    # print ("i[3] =", i[3])

    CD_theta_group.append(np.degrees(i[3]))                                               # store theta in CD list
    CD_eta_group.append(np.degrees(i[2]))                                                 # store eta in CD list
    CD_alpha_group.append(np.degrees(i[0]))                                               # store alpha in CD list
    CD_beta_group.append(np.degrees(i[1]))                                                # store beta in CD list
   
    # Forward Kinematics
    positions1, _ = forward_kinematics(dh_params, i)                                      # get angles for each trajectory

    Ps = np.array([0, 0, 0])
    Pe = np.array(positions1[4])
    Pw = np.array(positions1[5])
    Ph = np.array(positions1[7])

    # Inverse Kinematics
    qb = inverse_kinematics_from_points(Ps, Pe, Pw, 0, 0, 0)                               # getting new angles with IK from final effector positions

    IK_theta_group.append(np.degrees(-qb[3]))                                              # store theta in IK list
    IK_eta_group.append(np.degrees(qb[2]))                                                 # store eta in IK list
    IK_alpha_group.append(np.degrees(qb[0]))                                               # store alpha in IK list
    IK_beta_group.append(np.degrees(qb[1]))                                                # store beta in IK list


######################################################################################################
#                                  Plotting curves FORWARD vs INVERSE KINEMATICS
######################################################################################################

plt.style.use('_mpl-gallery')
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Graph 1 - theta
# axs[0, 0].plot(CD_theta_group, 'red', linewidth=1.0, label='Forward Kinematics')
axs[0, 0].plot(
    CD_theta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)
# axs[0, 0].plot(IK_theta_group, 'blue', linewidth=1.0, label='Inverse Kinematics')
axs[0, 0].plot(
    IK_theta_group,
    color='blue',
    linestyle='--',
    linewidth=1.2,
    marker='s',
    markersize=2,
    label='IK'
)
axs[0, 0].legend(fontsize=3)
axs[0, 0].set_title("Spherical coordinates of position: theta", fontsize=8)
axs[0, 0].set_xlabel("time", fontsize=8)
axs[0, 0].set_ylabel("Upper arm elevation (deg)", fontsize=8)
axs[0, 0].grid(False)
axs[0, 0].legend()

# Graph 2 - eta
# axs[0, 1].plot(CD_eta_group, 'red', linewidth=1.0, label='Forward Kinematics')
axs[0, 1].plot(
    CD_eta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)
# axs[0, 1].plot(IK_eta_group, 'blue', linewidth=1.0, label='Inverse Kinematics')
axs[0, 1].plot(
    IK_eta_group,
    color='blue',
    linestyle='--',
    linewidth=1.2,
    marker='s',
    markersize=2,
    label='IK'
)
axs[0, 1].legend(fontsize=3)
axs[0, 1].set_title("Spherical coordinates of position: eta", fontsize=8)
axs[0, 1].set_xlabel("time", fontsize=8)
axs[0, 1].set_ylabel("Upper arm yaw (deg)", fontsize=8)
axs[0, 1].grid(False)
axs[0, 1].legend()

# Graph 3 - alpha
# axs[1, 0].plot(CD_alpha_group, 'red', linewidth=1.0, label='Forward Kinematics')
axs[1, 0].plot(
    CD_alpha_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)
# axs[1, 0].plot(IK_alpha_group, 'blue', linewidth=1.0, label='Inverse Kinematics')
axs[1, 0].plot(
    IK_alpha_group,
    color='blue',
    linestyle='--',
    linewidth=1.2,
    marker='s',
    markersize=2,
    label='IK'
)
axs[1, 0].legend(fontsize=3)
axs[1, 0].set_title("Orientation Euler angles: alpha", fontsize=8)
axs[1, 0].set_xlabel("time", fontsize=8)
axs[1, 0].set_ylabel("Forearm yaw (deg)", fontsize=8)
axs[1, 0].grid(False)
axs[1, 0].legend()

# Graph 4 - beta
# axs[1, 1].plot(CD_beta_group, 'red', linewidth=1.0, label='Forward Kinematics')
axs[1, 1].plot(
    CD_beta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)
# axs[1, 1].plot(IK_beta_group, 'blue', linewidth=1.0, label='Inverse Kinematics')
axs[1, 1].plot(
    IK_beta_group,
    color='blue',
    linestyle='--',
    linewidth=1.2,
    marker='s',
    markersize=2,
    label='IK'
)
axs[1, 1].legend(fontsize=3)
axs[1, 1].set_title("Orientation Euler angles: beta", fontsize=8)
axs[1, 1].set_xlabel("time", fontsize=8)
axs[1, 1].set_ylabel("Forearm elevation (deg)", fontsize=8)
axs[1, 1].grid(False)
axs[1, 1].legend()

plt.tight_layout()
plt.show()

######################################################################################################
#                                   ERROR and STANDARD DEVIATION
#####################################################################################################

# theta
Erro_y_theta = []
for i in range (len(CD_theta_group)):
    Erro_y_theta.append(IK_theta_group[i]-CD_theta_group[i])
# print("Erro_y =", Erro_y_theta)

CD_theta_group_array = np.array(CD_theta_group)
y_array_theta = np.array(Erro_y_theta)

n = y_array_theta.size
media_theta = y_array_theta.mean()                                # mean error
# mediana = np.median(y_array_theta)                              # median
desvios = y_array_theta - media_theta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                # sample variance
std_samp_theta = np.sqrt(var_samp)                                 # sample standard deviation (sample variance) - USE

print("\nSummary of errors between Forward and Inverse Kinematics:")
print(f"Mean_theta = {media_theta:.2f}")
# print(f"Median = {mediana:.2f}")
print(f"Sample Standard Deviation = {std_samp_theta:.2f}")

media_theta = np.round(media_theta,2)
std_samp_theta = np.round(std_samp_theta,2)

# eta
Erro_y_eta = []
for i in range (len(CD_eta_group)):
    # print("CD_eta_group =", CD_eta_group[i])
    Erro_y_eta.append(IK_eta_group[i]-CD_eta_group[i])
# print("Erro_y_eta =", Erro_y_eta)

CD_eta_group_array = np.array(CD_eta_group)
y_array_eta = np.array(Erro_y_eta)

n = y_array_eta.size
media_eta = y_array_eta.mean()                                       # mean error
# mediana = np.median(y_array_theta)                                 # median
desvios = y_array_eta - media_eta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                  # sample variance
std_samp_eta = np.sqrt(var_samp)                                     # sample standard deviation (sample variance) - USE

# print("\nSummary:")
print(f"\nMean_eta = {media_eta:.2f}")
# print(f"Median = {mediana:.2f}")
print(f"Sample Standard Deviation = {std_samp_eta:.2f}")

media_eta = np.round(media_eta,2)
std_samp_eta = np.round(std_samp_eta,2)


# alpha
Erro_y_alpha = []
# for i in range (25):
for i in range (len(CD_alpha_group)):
    # print("CD_alpha_group =", CD_alpha_group[i])
    Erro_y_alpha.append(IK_alpha_group[i]-CD_alpha_group[i])
# print("Erro_y_alpha =", Erro_y_alpha)

CD_alpha_group_array = np.array(CD_alpha_group)
y_array_alpha = np.array(Erro_y_alpha)

n = y_array_alpha.size
media_alpha = y_array_alpha.mean()                                # mean error
# mediana = np.median(y_array_theta)                              # median
desvios = y_array_alpha - media_alpha
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                # sample variance
std_samp_alpha = np.sqrt(var_samp)                                 # sample standard deviation (sample variance) - USE

# print("\nSummary:")
print(f"\nMean_alpha = {media_alpha:.2f}")
# print(f"Median = {mediana:.2f}")
print(f"Sample Standard Deviation = {std_samp_alpha:.2f}")

media_alpha = np.round(media_alpha,2)
std_samp_alpha = np.round(std_samp_alpha,2)

# beta
Erro_y_beta = []
# for i in range (25):
for i in range (len(CD_beta_group)):
    # print("CD_beta_group =", CD_beta_group[i])
    Erro_y_beta.append(IK_beta_group[i]-CD_beta_group[i])
# print("Erro_y_beta =", Erro_y_beta)

CD_beta_group_array = np.array(CD_beta_group)
y_array_beta = np.array(Erro_y_beta)

n = y_array_beta.size
media_beta = y_array_beta.mean()                                  # mean error
# mediana = np.median(y_array_theta)                              # median
desvios = y_array_beta - media_beta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                # sample variance
std_samp_beta = np.sqrt(var_samp)                                  # sample standard deviation (sample variance) - USE

# print("\nSummary:")
print(f"\nMean_beta = {media_beta:.2f}")
# print(f"Median = {mediana:.2f}")
print(f"Sample Standard Deviation = {std_samp_beta:.2f}")

media_beta = np.round(media_beta,2)
std_samp_beta = np.round(std_samp_beta,2)

# #######################################################################################

mfm_x = np.array(['theta', "eta", "alpha", "beta"])
mfm_y = []
mfm_desvios = []

mfm_y.append(media_theta)
mfm_desvios.append(std_samp_theta)

mfm_y.append(media_eta)
mfm_desvios.append(std_samp_eta)

mfm_y.append(media_alpha)
mfm_desvios.append(std_samp_alpha)

mfm_y.append(media_beta)
mfm_desvios.append(std_samp_beta)

plt.style.use('_mpl-gallery')
# make data:
np.random.seed(1)
# plot:
fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
ax.set_title("Type 1: Comparison of angles: position and orientation", fontsize=8)
ax.errorbar(mfm_x, mfm_y, mfm_desvios, fmt='o', linewidth=2, capsize=6)
ax.set_xlabel("Intrinsic Coordinates", fontsize=8)
ax.set_ylabel("Mean absolute error (deg)", fontsize=8)
ax.grid(False)
plt.tight_layout()
plt.show()