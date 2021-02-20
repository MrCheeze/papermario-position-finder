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
    ( float32(2.1874359) , 121, -15, True),
    ( float32(2.1875)    , 120, -15, True),
    
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
    ( float32(-2.1874359) , -121, 15, True),
    ( float32(-2.1875)    , -120, 15, True),
]

start_position = [toFloat(0xC41C8000), toFloat(0xC30A0000)]
goal_position = [toFloat(0xC41C7A04), toFloat(0xC309C8E1)]


def printMotion(motion, position, axis):
    distance, sameAxis, oppositeAxis, _ = motion
    if axis == 0:
        x = sameAxis
        y = oppositeAxis
    else:
        y = -sameAxis
        x = oppositeAxis
    print('X: %4d, Y: %4d -- Position: %11.6f, %11.6f (%s%s)' % (x, y, position[0], position[1], '+' if distance >= 0 else '', distance))
    lua_file.write('    nextFrame(%d, %d)\n'%(x,y))



print('Start:', start_position)
print('Goal:', goal_position)

lua_file = open('go_to_positions.lua','w')
lua_file.write('''event.onexit(function()
	joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
end)

a_frames = 0
function nextFrame(x_input, y_input)
    prev_x = mainmemory.read_u32_be(0x10F1B0)
    prev_z = mainmemory.read_u32_be(0x10F1B8)
    savestate.save("temp.State")
    while mainmemory.read_u32_be(0x10F1B0) == prev_x and mainmemory.read_u32_be(0x10F1B8) == prev_z do
        while mainmemory.readbyte(0x10F23C) == 0xA do
            emu.frameadvance()
        end
        if mainmemory.readbyte(0x10F23C) == 0 then
            a_frames = 4
        end
        if a_frames > 0 then
            joypad.set({["P1 A"]="True"})
        end
        joypad.setanalog({["P1 X Axis"]=x_input, ["P1 Y Axis"]=y_input})
        emu.frameadvance()
        if mainmemory.readbyte(0x10F23C) == 1 or mainmemory.readbyte(0x10F23C) == 2 then
            savestate.load("temp.State")
            joypad.set({["P1 A"]="False"})
            a_frames = 0
            joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
            while mainmemory.readbyte(0x10F23C) ~= 0xA do
                emu.frameadvance()
            end
        end
    end
    if a_frames > 0 then
        a_frames = a_frames - 1
    end
    joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
end

function goToGoal()\n''')

position = start_position
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

lua_file.write('end\ngoToGoal()\n')
lua_file.close()
