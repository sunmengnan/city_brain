import test.set_sumo_home as st

st.set_sumo_home()
import test.libsumo as libsumo
import traci
import time

interface = [libsumo]
network = ["binjiang0"]
# network = ["1_1_turn"]

retrieval = [0, "pos", "sv", "all"]


def runSumo(mode, network, retrieval):
    mode.start(["sumo", "-c", "./sumocfg/{}.sumocfg".format(network), "--no-warnings"])
    stepz = 0
    time_start = time.time()
    cpu_start = time.clock()
    while stepz < 3600:
        mode.simulationStep()
        if retrieval == "pos":
            print("step: ", stepz)
            for veh_id in mode.vehicle.getIDList():
                position = mode.vehicle.getPosition(veh_id)
        elif retrieval == "sv":
            print("step: ", stepz)
            for veh_id in mode.vehicle.getIDList():
                speed = mode.vehicle.getSpeed(veh_id)
            for edge_id in mode.edge.getIDList():
                number = mode.edge.getLastStepVehicleNumber(edge_id)
        elif retrieval == "all":
            print("step: ", stepz)
            for veh_id in mode.vehicle.getIDList():
                position = mode.vehicle.getPosition(veh_id)
                speed = mode.vehicle.getSpeed(veh_id)
            for edge_id in mode.edge.getIDList():
                number = mode.edge.getLastStepVehicleNumber(edge_id)

        stepz += 1
    time_end = time.time()
    cpu_end = time.clock()
    mode.close()
    with open('./log/log_libsumo', 'a') as file:
        if hasattr(mode, "Connection"):
            file.write("traci,{},{},{},{}\n".format(network, retrieval, time_end - time_start, cpu_end - cpu_start))
        else:
            file.write("libsumo,{},{},{},{}\n".format(network, retrieval, time_end - time_start, cpu_end - cpu_start))


if __name__ == "__main__":
    for md in interface:
        for nt in network:
            for rt in retrieval:
                runSumo(md, nt, rt)
