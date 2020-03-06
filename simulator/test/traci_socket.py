import time
import set_sumo_home
set_sumo_home.set_sumo_home()
import libsumo as traci

traci.start(["sumo", "-c", "../sumocfg/binjiang1.sumocfg"])  # , "--step-length", "1"])

stepz = 0

time_start = time.time()
while stepz < 3600:
    print("step: ", stepz)
    for veh_id in traci.vehicle.getIDList():
        position = traci.vehicle.getPosition(veh_id)
        print(position)
    traci.simulationStep()
    stepz += 1

time_end = time.time()
print('totally cost', time_end - time_start)
traci.close()

