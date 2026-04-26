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
    while tamanho_reta < reta_total:                                                        # calculating the position of Pei
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
    vetor_a_transposto = np.transpose(a)                           # using the np.transpose() method for transpose matrix to calculate the dot product
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
#     z1 = np.cross(p_elbow - p_shoulder, [1, 0, 0])                  # axis associated with the shoulder. Creates the first Jacobian vector associated with shoulder rotation.
#     z2 = np.cross(p_wrist - p_elbow, [0, 1, 0])                     # axis associated with the elbow. Represents the elbow rotation.
#     z3 = np.cross(p_wrist - p_elbow, [0, 0, 1])                     # axis associated with the wrist. Represents the effect of wrist rotation.
#     return np.vstack([z1, z2, z3]).T                                # Stacks vectors z1, z2, z3 as columns. Result: a 3×3 matrix

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


# plotting 3D graph
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
    # ax.set_box_aspect([1, 1, 1])  # equal aspect
    plt.show()

#########################################################################################################################
###########################                         START                        ########################################
#########################################################################################################################

############################ Type 2 COMPLETE: rest → bottle → side table → rest     ###############################

def psi_humano(pw):
    # Article/regression coefficients
    a, b, c = 0.5, 0.9, 0.0

    chi = np.arctan2(pw[1], pw[0])
    psi = np.arccos(pw[2] / np.linalg.norm(pw))
    
    phi_s = np.degrees(a*chi + b*psi + c)
    return phi_s

def swivel_manip(Pw):
    Ps = np.array([0., 0., 0.])
    L1 = dh_params[2][1]
    L2 = dh_params[4][1]
    best_psi, max_w, _ = optimize_swivel(Ps, Pw, L1, L2)
    print("best_psi =", best_psi)
    print("max_w =", max_w)            
    return best_psi, max_w

# Type 2 COMPLETE: rest → bottle → side table → rest
traj_type2_completa = np.array([
    [0.00,  0.00, 0.10],  # 0: rest
    [0.20,  0.00, 0.20],  # 1: grab bottle (front)
    [0.25, -0.10, 0.25],  # 2: lift bottle
    [0.30, -0.15, 0.28],  # 3: lateral move
    [0.35, -0.20, 0.30],  # 4: FINAL POSE table
    [0.30, -0.15, 0.28],  # 5: lift from table
    [0.20,  0.00, 0.20],  # 6: return front
    [0.00,  0.00, 0.10]   # 7: rest
])


psi_h_list = []
psi_m_list = []

# Type 2: SIMPLE LISTS (no numpy!)
traj_xy = [[0.00, 0.00], [0.25, 0.00], [0.35, -0.20], [0.25, 0.00], [0.00, 0.00]]
# phi_s_vals = [52, 45, 62, 45, 52]  # pre-calculated

for pw in traj_type2_completa:
    
    # human ψ
    psi_h = psi_humano(pw)
    psi_h_list.append(psi_h)
    
    # manipulability ψ
    psi_m, w_m = swivel_manip(pw)
    psi_m_list.append(psi_m)

print("=== TYPE 2: WATER BOTTLE ===")
print(f"{'Point':<4} {'Pose':<20} {'ϕ_s':<6}")
print("-"*35)
for i, (pw, phi_s) in enumerate(zip(traj_type2_completa, psi_h_list)):
    print(f"{i:<4} {f'({pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f})':<20} {phi_s:<6.1f}°")
      
 

# **TWO GRAPHS**
plt.figure(figsize=(12,4))


# Graph 1: Evolution of ϕ_s
plt.subplot(1, 2, 1)
plt.plot(psi_h_list, 'go-', linewidth=3, markersize=8)
plt.plot(psi_m_list, 'blue', linewidth=3, markersize=8)
plt.xlabel('Point in the trajectory')
plt.ylabel('ϕ_s (°)')
plt.title('Type 2: Evolution of ϕ_s (Bottle)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
x_vals = [p[0] for p in traj_xy]
y_vals = [p[1] for p in traj_xy]
plt.plot(x_vals, y_vals, 'go-', linewidth=4, markersize=12)
plt.scatter(0.35, -0.20, c='red', s=200, label='Bottle')
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.title('Trajectory XY')
plt.legend(); plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()


# **IMAGE 1: Evolution of ϕ_s**
# Create the first plot
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(psi_h_list, 'go-', linewidth=3, markersize=8, label='Human')
ax1.plot(psi_m_list, 'blue', linewidth=3, markersize=8, label='Manipulability')
ax1.set_xlabel('Point in the trajectory')
ax1.set_ylabel('ϕ_s (°)')
ax1.set_title('Type 2: Evolution of ϕ_s (Bottle)')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_phi_s.png')  # Saves the first image

# **IMAGE 2: Trajectory XY**
# Create the second plot
fig2, ax2 = plt.subplots(figsize=(6, 4))
x_vals = [p[0] for p in traj_xy]
y_vals = [p[1] for p in traj_xy]
ax2.plot(x_vals, y_vals, 'go-', linewidth=4, markersize=12)
ax2.scatter(0.35, -0.20, c='red', s=200, label='Bottle')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Trajectory XY')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')
plt.tight_layout()
plt.savefig('trajectory_xy.png')    # Saves the second image