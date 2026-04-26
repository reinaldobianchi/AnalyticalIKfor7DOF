##################################################################################################
#                           CINEMÁTICA DIRETA 
##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar

# Homogeneous transformation matrix function DH
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



def forward_kinematics(dh_params, thetas):
    Tw = np.eye(4)
    T = np.eye(4)
    positions = [T[:3, 3]]                                  
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        Ti = dh_matrix(thetas[i], d, a, alpha)
        T = T @ Ti
        positions.append(T[:3, 3])
        # print("T")
        if i == 3:
            Tw = T
    return np.array(positions), Tw

def inverse_kinematics_from_points(Ps, Pe, Pw, _, t5, t6, t7):    
    q = np.zeros(7)

    ########    Calculation of angles q1 and q2      #######
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



    ########     Calculation of q4 (using Pw and Ps)     ########
    v = Pw - Ps
    d = np.linalg.norm(v)
    a = (d**2 -L1**2 - L2**2) / (-2 * L1 * L2)                            # Law of cosines
    arco_q4 = np.arccos(a)

    q4 = np.pi - arco_q4
    junta4 = (q4*180)/np.pi


    ########                     Calculation of q3                #######
    #Calculation of Pei (imaginário)
    # Vector diretor (difference between the points)
    v = Pe - Ps
    norma = np.linalg.norm(v)
    v_unitario = v / norma

    h =  L2 * np.cos(q4)                                                   # triangle height
    reta_total = L1 + h                                                    # total arm length + triangle height

    tempo = 0
    tamanho_reta = 0
    while tamanho_reta < reta_total:                                        # calculating Pei's position
        Pei_x = Ps[0] + v_unitario[0] * tempo
        Pei_y = Ps[1] + v_unitario[1] * tempo
        Pei_z = Ps[2] + v_unitario[2] * tempo
        tamanho_reta = np.sqrt( ((Ps[0]-Pei_x)**2) + ((Ps[1]-Pei_y)**2)  +  ((Ps[2]-Pei_z)**2)   )
        tempo = tempo + 0.001
    Pei = np.array([Pei_x,Pei_y,Pei_z])                                      # position Pei

    # Circumference data
    C = Pei                                                                  # centro

    #Simulating with theta3 = 0
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = np.radians([junta1, junta2, 0, junta4, 0, 0, 0])
    theta_test = [theta1, theta2, theta3, -theta4, theta5, theta6, theta7]

    points_Pw_zero, _ = forward_kinematics(dh_params, theta_test)
    
    Pe = np.array(points_Pw_zero[4])
    Pw_zero = np.array(points_Pw_zero[5])

    P1 = Pw_zero
    P2 = Pw

    v1 = P1 - C
    v2 = P2 - C
 
    dot = np.dot(v1, v2) 
    q3 = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    junta3 = np.round(np.degrees(q3),2)


    #############     Calculation of Q5, Q6 e Q7          #################
    Ps = np.array([0, 0, 0])
    Pe = np.array(positions[4])
    Pw = np.array(positions[5])
   
    q[0] = q1
    q[1] = q2
    q[2] = q3
    q[3] = q4

    q[4] = t5
    q[5] = t6
    q[6] = t7


    # Calcula T04
    T = np.eye(4)
    for i in range(4):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)                            # Get the transformation matrix: T0_4

    T04 = T.copy()                                                       

    # Calcula T47
    T = np.eye(4)
    for i in range(4, 7):
        theta, d, a, alpha = dh_params[i]
        T = T @ dh_matrix(q[i], d, a, alpha)                            # Get the transformation matrix: T4_7

    T47 = T.copy()                                                      

    # Matrix multiplication
    T07 = T04 @ T47                                                     # Multiply T04 by T0_7 = total transformation matrix (T07)

    # Extracting R47 via correct definition
    R04 = T04[:3,:3]                                                    # take a direction from 0 to 4
    R07 = T07[:3,:3]                                                    # take a direction from 0 to 7

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

# General Procedure: Calculate q5... q7 from the residual orientation (after q1–q4)
def wrist_ik_general(R):                                           

    s6 = np.sqrt(R[2,0]**2 + R[2,1]**2)
    q6 = np.arctan2(s6, R[2,2])

    if np.isclose(s6, 0):
        # print("Singularity detected")
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
    vetor_a_transposto = np.transpose(a)                             
    alpha = np.dot(n, vetor_a_transposto) 
    beta = alpha * n
    gama = a - beta
    omega = np.linalg.norm(gama)
    u = gama / omega
    v = np.cross(n, u) 

    a = (L1**2 - L2**2 + d**2) / (2 * d)
    R = np.sqrt(max(L1**2 - a**2, 0))                                 # distance between Pe and the hypotenuse, which will be the radius of the circle.        
    cos_alpha = (L2**2 - L1**2 - d**2) / (-2 * L1 * d)                # law of cosines.
    Pc = Ps + L1 * cos_alpha * n    
    Pe = Pc + R * (np.cos(psi) * u + np.sin(psi) * v)    
    return Pe

# Plotting the 3D Graph
def plot_3d(positions, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', linewidth=3, markersize=8, color='blue')

    # Adjust chart limits
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
    ax.set_title(title)
    ax.grid(True)
    plt.show()

#########################################################################################################################
#                                                   START                       
#########################################################################################################################

# DIRECT KINEMATICS  
print("\n\n1. - Direct Kinematics (Target Coordinates).")
# Joint angles (target)
theta1, theta2, theta3, theta4, theta5, theta6, theta7 = np.radians([20, 0, 25, 90, 0, 45, 0])         
theta_values = [theta1, theta2, theta3, -theta4, theta5, theta6, theta7]

# Execute direct kinematics
positions, Tw = forward_kinematics(dh_params, theta_values)

Ps = np.array([0., 0., 0.])                                 # shoulder position
p_elbow = np.array(positions[4])                                                                    
Pe = np.array([p_elbow[0], p_elbow[1], p_elbow[2]])         # elbow position
p_wrist = np.array(positions[5])
Pw = np.array([p_wrist[0], p_wrist[1], p_wrist[2]])         # desired wrist position
Pwa =np.array([p_wrist[0], p_wrist[1], 0])                  # image in the xy plane
p_hand = np.array(positions[7])                                                                     
Ph = np.array([p_hand[0], p_hand[1], p_hand[2]])            # hand position

print("\n     1.1 - Position: shoulder, elbow, wrist, and hand")
np.set_printoptions(precision=2, suppress=True)
print("           Ps = ", Ps)
print("           Pe = ", Pe)
print("           Pw = ", Pw)
print("           Ph = ", Ph)

L1 = dh_params[2][1]                                        # length from shoulder to elbow
L2 = dh_params[4][1]                                        # length from elbow to wrist
L3 = dh_params[6][1]                                        # wrist length to the tip of the hand

plot_3d(positions, 'Target Coordinates (Direct Kinematics).')

 
#####################################  SWIVEL ANGLE ANIMATION   #########################################  

print("\n2. Inverse Kinematics (calculation of angles: q1... q7).")

psi_values = np.linspace(0, 2*np.pi, 360)                   # Generation of Pe points for various values ​​of ψ
elbow_positions = np.array([calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi) for psi in psi_values])


# Plotting
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(elbow_positions[:, 0], elbow_positions[:, 1], elbow_positions[:, 2], label='Possible elbow positions in 3D space', color='blue')
ax.scatter(*Ps, color='green', label='Ps (shoulder)')
ax.scatter(*Pw, color='red', label='Pw (wrist)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title("Elbow position variation (Pe) with Swivel Angle (ψ)")
ax.legend()

# Graph elements
line, = ax.plot([], [], [], lw=3, color='blue')
point_shoulder, = ax.plot([], [], [], 'ko', markersize=6)
point_elbow, = ax.plot([], [], [], 'ro', markersize=6)
point_hand, = ax.plot([], [], [], 'go', markersize=6)

# List of values ​​of ψ
psi_values = np.linspace(0, 2*np.pi, 60)

# Update function
def update(frame):
    psi = psi_values[frame]
    Pwa =np.array([Pw[0], Pw[1], 0])                            
    try:
        p_elbow = calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi)             
    except ValueError:
        p_elbow = Ps

    # Update positions
    xs = [Ps[0], p_elbow[0], p_wrist[0]]
    ys = [Ps[1], p_elbow[1], p_wrist[1]]
    zs = [Ps[2], p_elbow[2], p_wrist[2]]

    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    point_shoulder.set_data([Ps[0]], [Ps[1]])
    point_shoulder.set_3d_properties([Ps[2]])
    point_elbow.set_data([p_elbow[0]], [p_elbow[1]])
    point_elbow.set_3d_properties([p_elbow[2]])
    point_hand.set_data([p_wrist[0]], [p_wrist[1]])
    point_hand.set_3d_properties([p_wrist[2]])

    ax.set_title(f'Swivel Angle $\\psi$ = {psi:.2f} rad')
    return line, point_shoulder, point_elbow, point_hand

# Animation
anim = FuncAnimation(fig, update, frames=len(psi_values), interval=100, blit=False)
plt.show()

#####################################  ANIMATION WITH THE ELLIPSOID   #########################################

# Manipulator positions
p_shoulder = np.array([0.0, 0.0, 0.0])
p_wrist = np.array([Pw[0], Pw[1], Pw[2]])

psi_anim = np.linspace(0, 2 * np.pi, 60)

def compute_jacobiano(psi):
    p_elbow = calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi)         
    z1 = np.cross(p_elbow - p_shoulder, [1, 0, 0])                  # axis associated with the shoulder. Creates the first Jacobian vector associated with shoulder rotation.
    z2 = np.cross(p_wrist - p_elbow, [0, 1, 0])                     # axis associated with the elbow. Represents the rotation of the elbow.
    z3 = np.cross(p_wrist - p_elbow, [0, 0, 1])                     # axis associated with the wrist. Represents the effect of rotation on the wrist.
    return np.vstack([z1, z2, z3]).T                                # Stack the vectors z1, z2, z3 as a column. Result: a 3x3 matrix.

def update(frame):
    psi = psi_anim[frame]
    p_elbow = calc_elbow_position(Ps, Pw, Pwa, L1, L2, psi)
    xs = [p_shoulder[0], p_elbow[0], p_wrist[0]]
    ys = [p_shoulder[1], p_elbow[1], p_wrist[1]]
    zs = [p_shoulder[2], p_elbow[2], p_wrist[2]]
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    point_shoulder.set_data([p_shoulder[0]], [p_shoulder[1]])
    point_shoulder.set_3d_properties([p_shoulder[2]])
    point_elbow.set_data([p_elbow[0]], [p_elbow[1]])
    point_elbow.set_3d_properties([p_elbow[2]])
    point_hand.set_data([p_wrist[0]], [p_wrist[1]])
    point_hand.set_3d_properties([p_wrist[2]])

    J = compute_jacobiano(psi)
    U, S, _ = np.linalg.svd(J @ J.T)
    radii = np.sqrt(S)
    directions = U

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.zeros(x.shape + (3,))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ellipsoid[i, j, :] = (
                radii[0] * x[i, j] * directions[:, 0] +
                radii[1] * y[i, j] * directions[:, 1] +
                radii[2] * z[i, j] * directions[:, 2]
            ) + p_wrist

    if ellipsoid_surface[0] is not None:
        ellipsoid_surface[0].remove()

    ellipsoid_surface[0] = ax.plot_surface(
        ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
        color='cyan', alpha=0.4, edgecolor='k'
    )

    ax.set_title(f'Swivel Angle ψ = {psi:.2f} rad')
    return line, point_shoulder, point_elbow, point_hand, ellipsoid_surface[0]

# Plotting a 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0,0.6)
ax.set_ylim(0,0.6)
ax.set_zlim(0,0.8)
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robotic Arm with Manipulability Ellipsoid')

line, = ax.plot([], [], [], lw=3, color='blue')
point_shoulder, = ax.plot([], [], [], 'ko', markersize=6)
point_elbow, = ax.plot([], [], [], 'ro', markersize=6)
point_hand, = ax.plot([], [], [], 'go', markersize=6)
ellipsoid_surface = [None]

ani = FuncAnimation(fig, update, frames=len(psi_anim), interval=150, blit=False)
ani.save('robotic_arm_ellipsoid.gif', writer='pillow', fps=10)
plt.show()

#########################################################################################################################
#                                   Calculating the best ψ (Swivel Angle) based on Manipulation
#########################################################################################################################

def compute_jacobian(q):    
    delta = 1e-5
    J = np.zeros((3, 7))
    T_base = forward_kinematics1(q)                                  # Position of the actuator with the joints q.
    p0 = T_base[:3, 3]                                               # Extracts only the (x,y,z) position from the end effector.
    for i in range(7):
        dq = np.copy(q)                                              # Creates a copy of q → dq (keeping the original q)
        dq[i] += delta                                               # It only increases the i-joint in δ
        T_i = forward_kinematics1(dq)                                # Recalculates the position of the effected work only with the change in joint i.
        pi = T_i[:3, 3]                                              # Extracts the position (x,y,z)
        J[:, i] = (pi - p0) / delta                                  # Performs the Jacobian calculation and stores it in matrix J, in the joint position i
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

def optimize_swivel(p_shoulder, p_wrist, p_hand, L1, L2):
    Pwa =np.array([p_wrist[0], p_wrist[1], 0])
    values = []
    def cost(psi):
        try:
            p_elbow = calc_elbow_position(p_shoulder, p_wrist, Pwa, L1, L2, psi)
            q = inverse_kinematics_from_points(p_shoulder, p_elbow, p_wrist, Tw, theta5, theta6, theta7)
            J = compute_jacobian(q)
            w = manipulability(J)
            values.append((psi, w))
            return -w
        except:
            return np.inf

    result = minimize_scalar(cost, bounds=(0, 2*np.pi), method='bounded')
    return result.x, -result.fun, values

# Calculate the new Pe using manipulability (best_psi).
print("\n    2.1 - Manipulability:  Calculating the best Pe.")
best_psi, max_w, psi_values = optimize_swivel(Ps, Pw, Ph, L1, L2) 
print("          . The best 𝜓 = %f rd" %round(best_psi,2))
print("          . The best Manipulability = %f" %round(max_w,4))

# Graph: manipulability x ψ
psis = [v[0] for v in psi_values]
ws = [v[1] for v in psi_values]

# Sort for best viewing
psis, ws = zip(*sorted(zip(psis, ws)))

# Plotting manipulability as a function of ψ
plt.figure(figsize=(8, 5))
plt.plot(psis, ws, label='Manipulability $w(\\psi)$', color='blue')
plt.axvline(best_psi, color='red', linestyle='--', label=f'ψ* = {best_psi:.2f} rad')
plt.title('Manipulability vs Swivel Angle (ψ)')
plt.xlabel('Swivel Angle ψ (rad)')
plt.ylabel('Manipulability w')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

######################################################################################################
#                                   Calculate new Pe
######################################################################################################
print("\n    2.2 - Calculate the best Pe:")
Pe = calc_elbow_position(Ps, Pw, Pwa, L1, L2, -best_psi)                                             # Calcula o novo com o melhor psi
print("          . Pe_NEW  =    ", Pe)

# Final Position
print("\n            2.3.2 - Final Position: Inverse Kinematics for Ps, Pe, Pw, and Ph: Joint Angles (rd)")
print("                Ps =     ",  Ps)
print("                Pe_new = ",  Pe)
print("                Pw =     ",  Pw)
print("                Ph =     ",  Ph)


#################################      Verification         ##############################################
print("\n3. - Verification")

print("\n     3.1 - Inverse kinematics for final position (rd)")
qb = inverse_kinematics_from_points(Ps, Pe, Pw, Tw, theta5, theta6, theta7)
print("\n            q     =", (qb))                                                 
print("            q (gr)=", np.degrees(qb))

print("\n     3.2 - Position using Direct Kinematics with new angles.")
theta_values = [qb[0], qb[1], qb[2], -qb[3], qb[4], qb[5], qb[6]]
positions, _ = forward_kinematics(dh_params, theta_values)                                          # calculates positions using new angles.

Ps = np.array([0, 0, 0])
Pe = np.array(positions[4])
Pw = np.array(positions[5])
Ph = np.array(positions[7])

np.set_printoptions(precision=2, suppress=True)
print("\n            Ps = ", Ps)
print("            Pe = ", Pe)
print("            Pw = ", Pw)
print("            Ph = ", Ph)

# Trajectory - Linear Interpolation between qa and qb

print("\n4. - Trajectory of the manipulator:")
print("     4.1 - Starting position = [0, 0, 0, 0, 0, 0, 0]")
qa = np.radians([0., 0., 0., 0., 0., 0., 0.])                                                       # Initial position of the manipulator.

print("\n     4.2 - Final position:")
print("           q (rd)=", (qb))                                                 
print("           q (gr)=", np.degrees(qb))

steps = 25                                                                                          # number of steps between the starting and ending points.
trajectory = np.linspace(qa, theta_values, steps)                                                   # Trajectory - Linear Interpolation

plot_3d(positions, 'Direct Kinematics - Final Position')

############################     Static chart with the best position within Ellipsoid

# Handler parameters 
L1 = dh_params[2][1]       
L2 = dh_params[4][1] 

p_shoulder = np.array([0.0, 0.0, 0.0])
p_wrist =    np.array([Pw[0], Pw[1], Pw[2]])
p_elbow =    np.array([Pe[0], Pe[1], Pe[2]])

psi_anim =   np.array([best_psi])

def compute_jacobianoo(psi):
    p_elbow = calc_elbow_position(p_shoulder, p_wrist, Pwa, L1, L2, psi)
    z1 = np.cross(p_elbow - p_shoulder, [1, 0, 0])
    z2 = np.cross(p_wrist - p_elbow, [0, 1, 0])
    z3 = np.cross(p_wrist - p_elbow, [0, 0, 1])
    return np.vstack([z1, z2, z3]).T

# Plotting a 3D figure
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

line, = ax.plot([], [], [], lw=3, color='blue')
point_shoulder, = ax.plot([], [], [], 'ko', markersize=6)
point_elbow, = ax.plot([], [], [], 'ro', markersize=6)
point_hand, = ax.plot([], [], [], 'go', markersize=6)
ellipsoid_surface = [None]

def update(frame):
    psi = psi_anim[frame]
    p_elbow = calc_elbow_position(p_shoulder, p_wrist, Pwa, L1, L2, psi)
    xs = [p_shoulder[0], p_elbow[0], p_wrist[0]]
    ys = [p_shoulder[1], p_elbow[1], p_wrist[1]]
    zs = [p_shoulder[2], p_elbow[2], p_wrist[2]]
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    point_shoulder.set_data([p_shoulder[0]], [p_shoulder[1]])
    point_shoulder.set_3d_properties([p_shoulder[2]])
    point_elbow.set_data([p_elbow[0]], [p_elbow[1]])
    point_elbow.set_3d_properties([p_elbow[2]])
    point_hand.set_data([p_wrist[0]], [p_wrist[1]])
    point_hand.set_3d_properties([p_wrist[2]])

    J = compute_jacobianoo(psi)
    U, S, _ = np.linalg.svd(J @ J.T)
    radii = np.sqrt(S)
    directions = U

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.zeros(x.shape + (3,))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ellipsoid[i, j, :] = (
                radii[0] * x[i, j] * directions[:, 0] +
                radii[1] * y[i, j] * directions[:, 1] +
                radii[2] * z[i, j] * directions[:, 2]
            ) + p_wrist

    if ellipsoid_surface[0] is not None:
        ellipsoid_surface[0].remove()

    ellipsoid_surface[0] = ax.plot_surface(
        ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
        color='cyan', alpha=0.4, edgecolor='k'
    )

    ax.set_title(f'Robotic Arm with Manipulability Ellipsoid. Swivel Angle ψ = {psi:.2f} rad')
    return line, point_shoulder, point_elbow, point_hand, ellipsoid_surface[0]

# Show
ani = FuncAnimation(fig, update, frames=len(psi_anim), interval=150, blit=False)
plt.show()

###########################################################################################
#                                   ANIMATION OF THE TRAJECTORY
###########################################################################################

# PLOTING 3D IMAGE 
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

line, = ax.plot([], [], [], 'o-', lw=2, color='blue')

def update(frame):
    q = trajectory[frame]
    positions, _ = forward_kinematics(dh_params, q)
    line.set_data(positions[:,0], positions[:,1])
    line.set_3d_properties(positions[:,2])
    return line,

ani = FuncAnimation(fig, update, frames=steps, interval=200, blit=True)

plt.show()

#########################################################################################################
#                                    POSITION TESTS
#########################################################################################################

print("\n5. Trajetory (rad)")
print("qa =", qa)
print("qb =", np.array(theta_values))
steps = 25

trajectory = np.linspace(qa, theta_values, steps, axis=0)

CD_theta_group = []
IK_theta_group = []

CD_eta_group = []
IK_eta_group = []

CD_alpha_group = []
IK_alpha_group = []

CD_beta_group = []
IK_beta_group = []

for i in trajectory:

    CD_theta_group.append(np.degrees(i[3]))                                               # stores theta in the CD (trajectory) list
    CD_eta_group.append(np.degrees(i[2]))                                                 # stores eta in the CD (trajectory) list
    CD_alpha_group.append(np.degrees(i[0]))                                               # stores alpha in the CD (trajectory) list
    CD_beta_group.append(np.degrees(i[1]))                                                # stores beta in the CD (trajectory) list
   
    # Direct Kinematics
    positions1, _ = forward_kinematics(dh_params, i)                                      # obtains the position of each trajectory

    Ps = np.array([0, 0, 0])
    Pe = np.array(positions1[4])
    Pw = np.array(positions1[5])
    Ph = np.array(positions1[7])

    # Inverse Kinematics
    qb = inverse_kinematics_from_points(Ps, Pe, Pw,_, 0, 0, 0)                             # obtains new angles with IK from the end positions of the end effector. 

    IK_theta_group.append(np.degrees(-qb[3]))                                              # stores theta in the IK list
    IK_eta_group.append(np.degrees(qb[2]))                                                 # stores eta in the IK list
    IK_alpha_group.append(np.degrees(qb[0]))                                               # stores alpha in the IK list
    IK_beta_group.append(np.degrees(qb[1]))                                                # stores beta in the IK list



######################################################################################################
#                              Plotting curves: Direct vs Inverse Kinematics
######################################################################################################

plt.style.use('_mpl-gallery')
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Gráfico 1 - theta
axs[0, 0].plot(
    CD_theta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)

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
axs[0, 0].set_title("Comparison trajectory vs Inverse Kinematics - theta", fontsize=8)
axs[0, 0].set_xlabel("time", fontsize=8)
axs[0, 0].set_ylabel("Upper arm elevation (deg)", fontsize=8)
axs[0, 0].grid(False)
axs[0, 0].legend()

# Graphic 2 - eta
axs[0, 1].plot(
    CD_eta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)

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
axs[0, 1].set_title("Comparison trajectory vs Inverse Kinematics - eta", fontsize=8)
axs[0, 1].set_xlabel("time", fontsize=8)
axs[0, 1].set_ylabel("Upper arm yaw (deg)", fontsize=8)
axs[0, 1].grid(False)
axs[0, 1].legend()

# Graphic 3 - alpha

axs[1, 0].plot(
    CD_alpha_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)

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
axs[1, 0].set_title("Comparison trajectory vs Inverse Kinematics - alpha", fontsize=8)
axs[1, 0].set_xlabel("time", fontsize=8)
axs[1, 0].set_ylabel("Forearm yaw (deg)", fontsize=8)
axs[1, 0].grid(False)
axs[1, 0].legend()

# Graphic 4 - beta

axs[1, 1].plot(
    CD_beta_group,
    color='red',
    linestyle='-',
    linewidth=3.2,
    marker='o',
    markersize=4,
    label='Trajectory'
)

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
axs[1, 1].set_title("Comparison trajectory vs Inverse Kinematics - beta", fontsize=8)
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

CD_theta_group_array = np.array(CD_theta_group)
y_array_theta = np.array(Erro_y_theta)

n = y_array_theta.size
media_theta = y_array_theta.mean()                                   # average error

desvios = y_array_theta - media_theta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                  # sample variance
std_samp_theta = np.sqrt(var_samp)                                   # sample standard deviation (sample variance)

print("\n6. Errors:  Trajetory vs Inverse Kinematics:")
print(f"   Average_theta = {media_theta:.2f}")
print(f"   Sample standard deviation = {std_samp_theta:.2f}")

media_theta = np.round(media_theta,2)
std_samp_theta = np.round(std_samp_theta,2)

# eta
Erro_y_eta = []
for i in range (len(CD_eta_group)):
    Erro_y_eta.append(IK_eta_group[i]-CD_eta_group[i])


CD_eta_group_array = np.array(CD_eta_group)
y_array_eta = np.array(Erro_y_eta)

n = y_array_eta.size
media_eta = y_array_eta.mean()                                       # average error
desvios = y_array_eta - media_eta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                  # sample variance
std_samp_eta = np.sqrt(var_samp)                                     # sample standard deviation (sample variance)

print(f"\n   Average_eta = {media_eta:.2f}")
print(f"   Sample standard deviation = {std_samp_eta:.2f}")

media_eta = np.round(media_eta,2)
std_samp_eta = np.round(std_samp_eta,2)


# alpha
Erro_y_alpha = []
for i in range (len(CD_alpha_group)):
    Erro_y_alpha.append(IK_alpha_group[i]-CD_alpha_group[i])

CD_alpha_group_array = np.array(CD_alpha_group)
y_array_alpha = np.array(Erro_y_alpha)

n = y_array_alpha.size
media_alpha = y_array_alpha.mean()                                # average error
desvios = y_array_alpha - media_alpha
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                # sample variance
std_samp_alpha = np.sqrt(var_samp)                                 # sample standard deviation (sample variance)

print(f"\n   Average_alpha = {media_alpha:.2f}")
print(f"   Sample standard deviation = {std_samp_alpha:.2f}")

media_alpha = np.round(media_alpha,2)
std_samp_alpha = np.round(std_samp_alpha,2)

# beta
Erro_y_beta = []
for i in range (len(CD_beta_group)):
    Erro_y_beta.append(IK_beta_group[i]-CD_beta_group[i])

CD_beta_group_array = np.array(CD_beta_group)
y_array_beta = np.array(Erro_y_beta)

n = y_array_beta.size
media_beta = y_array_beta.mean()                                   # average error
desvios = y_array_beta - media_beta
desvios2 = desvios**2
soma_quadrados = desvios2.sum()

var_samp = soma_quadrados / (n - 1)                                # sample variance
std_samp_beta = np.sqrt(var_samp)                                  # sample standard deviation (sample variance)

print(f"\n   Average_beta = {media_beta:.2f}")
print(f"   Sample standard deviation = {std_samp_beta:.2f}")

media_beta = np.round(media_beta,2)
std_samp_beta = np.round(std_samp_beta,2)

########################################################################################

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
ax.set_title("Errors: trajectory vs IK", fontsize=8)
ax.errorbar(mfm_x, mfm_y, mfm_desvios, fmt='o', linewidth=2, capsize=6)
ax.set_xlabel("Intrinsic Coordinates", fontsize=8)
ax.set_ylabel("Mean absolute error (deg)", fontsize=8)
ax.grid(False)
plt.tight_layout()
plt.show()
