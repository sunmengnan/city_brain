import traci


def runTraci():
	traci.start(["sumo", "-c", "./sumocfg/6_6_turn.sumocfg", "--no-warnings"])
	step = 0
	while step < 3599:
		traci.simulationStep()
		vehicle_count = traci.vehicle.getIDCount()
		lane_vehicle_count = 0
		for laneID in traci.lane.getIDList():
			lane_vehicle_count += traci.lane.getLastStepVehicleNumber(laneID)
		lane_waiting_vehicle_count = 0
		for laneID in traci.lane.getIDList():
			for vehicleID in traci.lane.getLastStepVehicleIDs(laneID):
				if traci.vehicle.getSpeed(vehicleID) < 0.1:
					lane_waiting_vehicle_count += 1
		lane_vehicles = []
		for laneID in traci.lane.getIDList():
			lane_vehicles += traci.lane.getLastStepVehicleIDs(laneID)
		vehicle_speed = {}
		for vehicleID in traci.vehicle.getIDList():
			vehicle_speed[vehicleID] = traci.vehicle.getSpeed(vehicleID)
		vehicle_distance = {}
		for vehicleID in traci.vehicle.getIDList():
			vehicle_distance[vehicleID] = traci.vehicle.getDistance(vehicleID)
		print("Total: {}  On lane: {}  Waiting on lane: {}\n".format(vehicle_count, lane_vehicle_count, lane_waiting_vehicle_count))
		step += 1
	traci.close()


if __name__=="__main__":
	runTraci()
