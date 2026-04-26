##################################################################################################
#                           FORWARD KINEMATICS 
##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar

# DH homogeneous transformation matrix function
def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# DH parameters for the DLR 7 DOF arm (theta, d, a, alpha)
dh_params = [
    (0,    0.0,  0.0, -np.pi/2),
    (0,    0.0,  0.0,  np.pi/2),
    (0,    0.4,  0.0,  np.pi/2),
    (0,    0.0,  0.0, -np.pi/2),
    (0,    0.4,  0.0, -np.pi/2),
    (0,    0.0,  0.0,  np.pi/2),
    (0,    0.126,  0.0,  0.0)
]

# Forward kinematics
def forward_kinematics(dh_params, thetas):
    Tw = np.eye(4)
    T = np.eye(4)
    positions = [T[:3, 3]]                                      # initial position (base)
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



    L1 = dh_params[2][1]
    L2 = dh_params[4][1]
    ########             # Calculation of q4 (using Pw and Ps)        ########
    # print("\n    2.2 - Calculation of q4 - Law of cosines - Triangle geometry")
    v = Pw - Ps
    d = np.linalg.norm(v)
    a = (d**2 -L1**2 - L2**2) / (-2 * L1 * L2)                                              # law of cosines
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
    while tamanho_reta < reta_total:                                                        # calculating position of Pei
        Pei_x = Ps[0] + v_unitario[0] * tempo
        Pei_y = Ps[1] + v_unitario[1] * tempo
        Pei_z = Ps[2] + v_unitario[2] * tempo
        tamanho_reta = np.sqrt( ((Ps[0]-Pei_x)**2) + ((Ps[1]-Pei_y)**2)  +  ((Ps[2]-Pei_z)**2)   )
        tempo = tempo + 0.001
    Pei = np.array([Pei_x,Pei_y,Pei_z])                                                     # position Pei

    # --- Circumference data ---
    C = Pei                                                                                 # center
    # r = np.sqrt( ((Pw[0]-Pei[0])**2) + ((Pw[1]-Pei[1])**2)  +  ((Pw[2]-Pei[2])**2) )        # radius
    # n = np.array([0.0, 0.0, 1.0])                                                           # plane normal vector (here, XY plane)

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
    # --- Dot and cross product ---
    dot = np.dot(v1, v2)
    # cross = np.cross(v1, v2)

    # --- Unsigned angle ---
    q3 = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # --- Convert to degrees ---
    # junta3 = np.round(np.degrees(q3),2)
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
    R04 = T04[:3,:3]                                                    # gets orientation from 0 to 4
    R07 = T07[:3,:3]                                                    # gets orientation from 4 to 7

    R47 = R04.T @ R07                                                   # calculates target

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
def wrist_ik_general(R):                                            # General procedure

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
    vetor_a_transposto = np.transpose(a)                           # using the np.transpose() method for transpose matrix to calculate dot product
    alpha = np.dot(n, vetor_a_transposto) 
    beta = alpha * n
    gama = a - beta
    omega = np.linalg.norm(gama)
    u = gama / omega
    v = np.cross(n, u) 
    # triangle geometry
    a = (L1**2 - L2**2 + d**2) / (2 * d)
    R = np.sqrt(max(L1**2 - a**2, 0))                               # distance between Pe and hypotenuse, which will be the circumference radius        
    # psi_graus = (psi*180)/np.pi   
    cos_alpha = (L2**2 - L1**2 - d**2) / (-2 * L1 * d)              # law of cosines
    Pc = Ps + L1 * cos_alpha * n    
    Pe = Pc + R * (np.cos(psi) * u + np.sin(psi) * v)    
    return Pe


# # (ELLIPSOID) Special Jacobian based on swivel angle (ψ) in a humanoid robotic arm (7 DOF, shoulder–elbow–wrist).
# def compute_jacobiano(psi):
#     L1 = dh_params[2][1]
#     L2 = dh_params[4][1]
#     p_elbow = calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi)         # calculation of Pe
#     z1 = np.cross(p_elbow - p_shoulder, [1, 0, 0])                  # axis associated with shoulder. Creates the first Jacobian vector associated with shoulder rotation.
#     z2 = np.cross(p_wrist - p_elbow, [0, 1, 0])                     # axis associated with elbow. Represents the elbow rotation.
#     z3 = np.cross(p_wrist - p_elbow, [0, 0, 1])                     # axis associated with wrist. Represents the effect of rotation at the wrist.
#     return np.vstack([z1, z2, z3]).T                                # Stacks vectors z1, z2, z3 as a column. Result: a 3×3 matrix

def compute_jacobian(q):

    delta = 1e-5
    J = np.zeros((3, 7))
    T_base = forward_kinematics1(q)                                  # End-effector position with joints q
    p0 = T_base[:3, 3]                                               # Extract only position (x,y,z) of the end-effector
    for i in range(7):
        dq = np.copy(q)                                              # Creates a copy of q → dq (preserving original q)
        dq[i] += delta                                               # Increments only joint i by δ
        T_i = forward_kinematics1(dq)                                # Recalculates end-effector position only with change in joint i
        pi = T_i[:3, 3]                                              # extracts linear position (x,y,z)
        J[:, i] = (pi - p0) / delta                                  # calculates Jacobian and stores in matrix J at joint i position
         
    return J

def forward_kinematics1(q):
    L1 = dh_params[2][1]
    L2 = dh_params[4][1]
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
            q = inverse_kinematics_from_points(p_shoulder, p_elbow, p_wrist, 0, 0, 0)                           # IK
            J = compute_jacobian(q)                                                                             # MANIPULABILITY - JACOBIAN
            w = manipulability(J)                                                                               # MANIPULABILITY - ELLIPSOID
            values.append((psi, w))
            return -w
        except:            
            return np.inf

    result = minimize_scalar(cost, bounds=(0, np.pi), method='bounded')
    return result.x, -result.fun, values


#########################################################################################################################
###########################                         START                        ########################################
#########################################################################################################################

############################                 TEST TYPE 5                                  ###############################

def swivel_manip(Pw):
    Ps = np.array([0., 0., 0.])
    L1 = dh_params[2][1]
    L2 = dh_params[4][1]
    best_psi, max_w, _ = optimize_swivel(Ps, Pw, L1, L2)
    print("best_psi =", best_psi)
    print("max_w =", max_w)            
    return best_psi, max_w



psi_m_list = []

# Type 5: Touch face (smooth, close to body)
traj_type5 = [
    [0.00, 0.00, 0.10],   # 0: rest
    [0.00, 0.00, 0.25],   # 1: goes up
    [0.00, 0.00, 0.45],   # 2: FACE ✓
    [-0.05,0.05, 0.42],   # 3: lateral head touch
    [0.00, 0.00, 0.45],   # 4: face center
    [0.00, 0.00, 0.25],   # 5: goes down
    [0.00, 0.00, 0.10]    # 6: rest
]

# ϕ_s expected values (based on validated regression)
phi_s_vals = [52, 28, 18, 22, 18, 28, 52]

print("=== TYPE 5: TOUCH FACE ===")
print(f"{'#':<2} {'Pose':<20} {'ϕ_s':<4}")
print("-"*28)

for i, (pose, phi) in enumerate(zip(traj_type5, phi_s_vals)):
    print(f"{i:<2} {f'({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})':<20} {phi:<4}°")

    # manipulability ψ
    psi_m, w_m = swivel_manip(pose)
    # print("psi_m =",psi_m)
    psi_m_list.append(psi_m)




# **IMAGE 1: Evolution of ϕ_s**
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(range(7), phi_s_vals, 'co-', label='ψ_h (human)', linewidth=3, markersize=10)
ax1.plot(range(7), psi_m_list, 'bo-', label='ψ_m (manipulability)', linewidth=3, markersize=10)
ax1.set_xticks(range(7))
ax1.set_xticklabels(['Rest','Up','Face','Head','Face','Down','Rest'])
ax1.set_ylabel('ϕ_s (°)')
ax1.set_title('Type 5: minimum ϕ_s 18°')
ax1.grid(True)
ax1.legend()
plt.tight_layout()
plt.savefig('phi_s_evolution_type5.png')

# **IMAGE 2: Trajectory XY**
fig2, ax2 = plt.subplots(figsize=(6, 4))
x_vals = [p[0] for p in traj_type5]
y_vals = [p[1] for p in traj_type5]
ax2.plot(x_vals, y_vals, 'co-', linewidth=3, markersize=10)
ax2.scatter(0.00, 0.00, c='red', s=150, label='Face')
ax2.scatter(-0.05, 0.05, c='orange', s=100, label='Head')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Trajectory XY Type 5')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')
plt.tight_layout()
plt.savefig('trajectory_xy_type5.png')

plt.show()