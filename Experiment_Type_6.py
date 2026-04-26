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
        if i == 3:
            Tw = T
    return np.array(positions), Tw

# INVERSE KINEMATICS: Q1, Q2, Q3 and Q4 (using Pw)
def inverse_kinematics_from_points(Ps, Pe, Pw, t5, t6, t7):    
    q = np.zeros(7)

    ########       Calculation of angles q1 and q2              #######
    if Pe[0]==0 and Pe[1] == 0:
        q1 = np.arctan2(Pe[1],Pe[0])
        junta1 = (q1*180)/np.pi
        q2 = 0
        junta2 = (q2*180)/np.pi
    else:
        q1 = np.arctan2(Pe[1],Pe[0])
        junta1 = (q1*180)/np.pi
        q2 = np.arctan2(np.sqrt(np.power(Pe[1],2) + np.power(Pe[0],2)), Pe[2])
        junta2 = (q2*180)/np.pi

    L1 = dh_params[2][1]
    L2 = dh_params[4][1]
    
    ########             Calculation of q4 (using Pw and Ps)        ########
    v = Pw - Ps
    d = np.linalg.norm(v)
    a = (d**2 -L1**2 - L2**2) / (-2 * L1 * L2)                                              # law of cosines
    arco_q4 = np.arccos(a)

    q4 = np.pi - arco_q4
    junta4 = (q4*180)/np.pi

    ########                     Calculation of q3                #######
    v = Pe - Ps
    norma = np.linalg.norm(v)
    v_unitario = v / norma

    h =  L2 * np.cos(q4)                                                                    # triangle height
    reta_total = L1 + h                                                                     # total arm length + triangle height

    tempo = 0
    tamanho_reta = 0
    while tamanho_reta < reta_total:                                                        # calculating Pei position
        Pei_x = Ps[0] + v_unitario[0] * tempo
        Pei_y = Ps[1] + v_unitario[1] * tempo
        Pei_z = Ps[2] + v_unitario[2] * tempo
        tamanho_reta = np.sqrt( ((Ps[0]-Pei_x)**2) + ((Ps[1]-Pei_y)**2)  +  ((Ps[2]-Pei_z)**2)   )
        tempo = tempo + 0.001
    Pei = np.array([Pei_x,Pei_y,Pei_z])                                                     # Pei position

    C = Pei                                                                                 # center

    # Simulating with theta3 = 0
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = np.radians([junta1, junta2, 0, junta4, 0, 0, 0])
    theta_test = [theta1, theta2, theta3, -theta4, theta5, theta6, theta7]

    points_Pw_zero, _ = forward_kinematics(dh_params, theta_test)
    
    Pe = np.array(points_Pw_zero[4])
    Pw_zero = np.array(points_Pw_zero[5])

    P1 = Pw_zero
    P2 = Pw
    # --- Radial vectors ---
    v1 = P1 - C
    v2 = P2 - C

    # --- Unsigned angle ---
    q3 = np.arccos(dot := np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    q[0], q[1], q[2], q[3] = q1, q2, q3, q4
    q[4], q[5], q[6] = t5, t6, t7

    # Calculate T04
    T = np.eye(4)
    for i in range(4):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
    T04 = T.copy()

    # Calculate T47
    T = np.eye(4)
    for i in range(4, 7):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
    T47 = T.copy()

    T07 = T04 @ T47                                                     
    R04 = T04[:3,:3]                                                    
    R07 = T07[:3,:3]                                                    
    R47 = R04.T @ R07                                                   

    q5_calc, q6_calc, q7_calc = wrist_ik_general(R47)

    q[4], q[5], q[6] = q5_calc, q6_calc, q7_calc
    return q

def wrist_ik_general(R_mat):
    s6 = np.sqrt(R_mat[2,0]**2 + R_mat[2,1]**2)
    q6 = np.arctan2(s6, R_mat[2,2])

    if np.isclose(s6, 0):
        print("Singularity detected")
        return 0, q6, np.arctan2(-R_mat[0,1], R_mat[0,0])

    q7 = np.arctan2(R_mat[2,1], -R_mat[2,0])
    q5 = np.arctan2(R_mat[1,2], R_mat[0,2])
    return q5, q6, q7

def calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi):
    v = Pw - Ps
    d = np.linalg.norm(v)
    n = v / d
    m = Pwa - Ps
    da = np.linalg.norm(m)
    a = m / da    
    vetor_a_transposto = np.transpose(a)
    alpha = np.dot(n, vetor_a_transposto) 
    beta = alpha * n
    gama = a - beta
    u = gama / np.linalg.norm(gama)
    v_vec = np.cross(n, u) 
    
    a_tri = (L1**2 - L2**2 + d**2) / (2 * d)
    R_circ = np.sqrt(max(L1**2 - a_tri**2, 0))
    cos_alpha = (L2**2 - L1**2 - d**2) / (-2 * L1 * d)
    Pc = Ps + L1 * cos_alpha * n    
    Pe = Pc + R_circ * (np.cos(psi) * u + np.sin(psi) * v_vec)    
    return Pe

def compute_jacobian(q):
    delta = 1e-5
    J = np.zeros((3, 7))
    T_base = forward_kinematics1(q)
    p0 = T_base[:3, 3]
    for i in range(7):
        dq = np.copy(q)
        dq[i] += delta
        T_i = forward_kinematics1(dq)
        pi = T_i[:3, 3]
        J[:, i] = (pi - p0) / delta
    return J

def forward_kinematics1(q):
    L1, L2 = dh_params[2][1], dh_params[4][1]
    q1, q2, q3, q4, q5, q6, q7 = q
    R_shoulder = R.from_euler('zyx', [q1, q2, q3]).as_matrix()
    p_elbow = R_shoulder @ np.array([0, 0, L1])
    R_elbow = R.from_rotvec([0, q4, 0]).as_matrix()
    R_wrist = R.from_euler('zyx', [q5, q6, q7]).as_matrix()
    R_hand = R_shoulder @ R_elbow @ R_wrist
    p_wrist = p_elbow + (R_shoulder @ R_elbow @ np.array([0, 0, L2]))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_hand, p_wrist
    return T

def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))

def optimize_swivel(p_shoulder, p_wrist, L1, L2):
    Pwa = np.array([p_wrist[0], p_wrist[1], 0])
    def cost(psi):
        try:
            p_elbow = calc_elbow_position(p_shoulder, p_wrist, Pwa, L1, L2, psi)
            q = inverse_kinematics_from_points(p_shoulder, p_elbow, p_wrist, 0, 0, 0)
            J = compute_jacobian(q)
            return -manipulability(J)
        except:            
            return np.inf
    result = minimize_scalar(cost, bounds=(0, np.pi), method='bounded')
    return result.x, -result.fun, []

##################################################################################################
#                                          START                                                 #
##################################################################################################

def swivel_manip(Pw):
    Ps = np.array([0., 0., 0.])
    L1, L2 = dh_params[2][1], dh_params[4][1]
    best_psi, max_w, _ = optimize_swivel(Ps, Pw, L1, L2)
    print(f"best_psi = {best_psi}, max_w = {max_w}")            
    return best_psi, max_w

psi_m_list = []

# Type 6: Writing (low, precise trajectory)
traj_type6 = [
    [0.00, 0.00, 0.20],   # 0: rest
    [0.15, 0.15, 0.22],   # 1: grab pen
    [0.10, 0.25, 0.20],   # 2: WRITING POSE ✓
    [0.12, 0.22, 0.21],   # 3: small writing movement
    [0.10, 0.25, 0.20],   # 4: return position
    [0.15, 0.15, 0.22],   # 5: release pen
    [0.00, 0.00, 0.20]    # 6: rest
]

phi_s_vals = [52, 42, 38, 30, 38, 42, 52]  # Pre-calculated human values

print("=== TYPE 6: WRITING ===")
print(f"{'#':<2} {'Pose':<20} {'ϕ_s':<4}")
print("-" * 28)

for i, (pose, phi) in enumerate(zip(traj_type6, phi_s_vals)):
    print(f"{i:<2} {f'({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})':<20} {phi:<4}°")
    psi_m, _ = swivel_manip(pose)
    psi_m_list.append(np.degrees(psi_m))

# **IMAGE 1: Evolution of ϕ_s**
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(range(7), phi_s_vals, 'ro-', label='ψ_h (human)', linewidth=3, markersize=10)
ax1.plot(range(7), psi_m_list, 'bo-', label='ψ_m (manipulability)', linewidth=3, markersize=10)
ax1.set_xticks(range(7))
ax1.set_xticklabels(['Rest','Grab','Write','Move','Return','Release','Rest'])
ax1.set_ylabel('ϕ_s (°)')
ax1.set_title('Type 6: Evolution of ϕ_s (Writing)')
ax1.grid(True)
ax1.legend()
plt.tight_layout()
plt.savefig('phi_s_evolution_type6.png')

# **IMAGE 2: Trajectory XY**
fig2, ax2 = plt.subplots(figsize=(6, 4))
x_vals = [p[0] for p in traj_type6]
y_vals = [p[1] for p in traj_type6]
ax2.plot(x_vals, y_vals, 'mo-', linewidth=3, markersize=10)
ax2.scatter(0.10, 0.25, c='red', s=150, label='Writing Area')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Trajectory XY Type 6')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')
plt.tight_layout()
plt.savefig('trajectory_xy_type6.png')

plt.show()