from numpy import float32
import numpy as np

np.set_printoptions(floatmode='unique')

def toFloat(x):
    return np.frombuffer(np.array(x, dtype=np.uint32).tobytes(), dtype=float32)[0]

motions = [ # (distance moved, joystick along same axis, joystick along opposite axis, is jumping)
    ( float32(0.65921944),  25,  -6, True),
    ( float32(0.9732757) ,  35,  -7, True),
    ( float32(1.0043554) ,  36,  -7, True),
    ( float32(1.2872953) ,  45,  -8, True),
    ( float32(1.3184389) ,  46,  -8, True),
    ( float32(1.3495119) ,  47,  -8, True),
    ( float32(1.601344)  ,  55,  -9, True),
    ( float32(1.6324947) ,  56,  -9, True),
    ( float32(1.6636039) ,  57,  -9, True),
    ( float32(1.6946687) ,  58,  -9, True),
    ( float32(1.915393)  ,  65, -10, True),
    ( float32(1.9465514) ,  66, -10, True),
    ( float32(1.9776584) ,  67, -10, True),
    ( float32(2.0087109) ,  68, -10, True),
    ( float32(2.164805)  ,  25,  -6, False),
    ( float32(2.1874359) , 121, -15, True),
    ( float32(2.1875)    , 120, -15, True),
    ( float32(2.243319)  ,  35,  -7, False),
    ( float32(2.2510302) ,  36,  -7, False),
    ( float32(2.321765)  ,  45,  -8, False),
    ( float32(2.3296096) ,  46,  -8, False),
    ( float32(2.3373194) ,  47,  -8, False),
    ( float32(2.4002774) ,  55,  -9, False),
    ( float32(2.4081237) ,  56,  -9, False),
    ( float32(2.415901)  ,  57,  -9, False),
    ( float32(2.4236085) ,  58,  -9, False),
    
    ( float32(-0.65921944),  -25,  6, True),
    ( float32(-0.9732757) ,  -35,  7, True),
    ( float32(-1.0043554) ,  -36,  7, True),
    ( float32(-1.2872953) ,  -45,  8, True),
    ( float32(-1.3184389) ,  -46,  8, True),
    ( float32(-1.3495119) ,  -47,  8, True),
    ( float32(-1.601344)  ,  -55,  9, True),
    ( float32(-1.6324947) ,  -56,  9, True),
    ( float32(-1.6636039) ,  -57,  9, True),
    ( float32(-1.6946687) ,  -58,  9, True),
    ( float32(-1.915393)  ,  -65, 10, True),
    ( float32(-1.9465514) ,  -66, 10, True),
    ( float32(-1.9776584) ,  -67, 10, True),
    ( float32(-2.0087109) ,  -68, 10, True),
    ( float32(-2.164805)  ,  -25,  6, False),
    ( float32(-2.1874359) , -121, 15, True),
    ( float32(-2.1875)    , -120, 15, True),
    ( float32(-2.243319)  ,  -35,  7, False),
    ( float32(-2.2510302) ,  -36,  7, False),
    ( float32(-2.321765)  ,  -45,  8, False),
    ( float32(-2.3296096) ,  -46,  8, False),
    ( float32(-2.3373194) ,  -47,  8, False),
    ( float32(-2.4002774) ,  -55,  9, False),
    ( float32(-2.4081237) ,  -56,  9, False),
    ( float32(-2.415901)  ,  -57,  9, False),
    ( float32(-2.4236085) ,  -58,  9, False),
]

start_position = [float32(-700), float32(0)]

goal_positions = [
    [toFloat(0xC41C7A04), toFloat(0xC309C8E1)],
    [toFloat(0xC41C7A24), toFloat(0xC309CB64)],
    [toFloat(0xC432D701), toFloat(0xC2CBEFF1)],
    [toFloat(0xC41DFA31), toFloat(0xC3055CE6)],
    [toFloat(0xC417FA1C), toFloat(0xC31058FA)],
    [toFloat(0xC41EFA6D), toFloat(0xC30D50FD)],
]

print(goal_positions)

position = start_position

def printMotion(motion, position, axis):
    distance, sameAxis, oppositeAxis, jump = motion
    if axis == 0:
        x = sameAxis
        y = oppositeAxis
    else:
        y = -sameAxis
        x = oppositeAxis
    print('X: %4d, Y: %4d, Jump: %-5s -- Position: %11.6f, %11.6f (%s%s)' % (x, y, jump, position[0], position[1], '+' if distance >= 0 else '', distance))

for goal_position in goal_positions:

    print('Start:', position)
    print('Goal:', goal_position)

    stepcount = 0
    for axis in range(2):

        while position[axis] != goal_position[axis]:
            closest_distance = abs(goal_position[axis] - position[axis])
            chosen_motion1 = None
            chosen_motion2 = None
            chosen_motion3 = None
            
            for motion1 in motions:
                dist = abs(goal_position[axis] - (position[axis]+motion1[0]))
                if dist < closest_distance:
                    chosen_motion1 = motion1
                    chosen_motion2 = None
                    chosen_motion3 = None
                    closest_distance = dist
                    
                for motion2 in motions:
                    dist = abs(goal_position[axis] - ((position[axis]+motion1[0])+motion2[0]))
                    if dist < closest_distance:
                        chosen_motion1 = motion1
                        chosen_motion2 = motion2
                        chosen_motion3 = None
                        closest_distance = dist
                        
                    for motion3 in motions:
                        dist = abs(goal_position[axis] - (((position[axis]+motion1[0])+motion2[0])+motion3[0]))
                        if dist < closest_distance:
                            chosen_motion1 = motion1
                            chosen_motion2 = motion2
                            chosen_motion3 = motion3
                            closest_distance = dist

            if chosen_motion1 is None:
                print('Giving up at dist=%s'%abs(goal_position[axis] - position[axis]))
                break

            position[axis] += chosen_motion1[0]
            printMotion(chosen_motion1, position, axis)
            stepcount += 1
            if chosen_motion2 is not None:
                position[axis] += chosen_motion2[0]
                printMotion(chosen_motion2, position, axis)
                stepcount += 1
            if chosen_motion3 is not None:
                position[axis] += chosen_motion3[0]
                printMotion(chosen_motion3, position, axis)
                stepcount += 1

            if position[axis] == goal_position[axis]:
                print(['Solved X axis.','Solved Z Axis.'][axis])
                break

    print('Total steps:', stepcount)
    print()
