import numpy as np

def calculate():
    global SUMOCFG
    logFile = open("./log/log", 'w')
    for sumocfg in SUMOCFG:
        time = []
        for line in open("./log/log_"+sumocfg, 'r'):
            line = line.strip()
            lineSplit = line.split(':')
            minute = int(lineSplit[0])
            second = float(lineSplit[1])
            t = minute * 60.0 + second
            time.append(3600.0/t)
        time = np.array(time)
        logFile.write(sumocfg+": "+str(time.mean())+"/"+str(time.std())+"\n")
        time = time.tolist()
    time = []
    for line in open("./log/log_6_6_interface", 'r'):
        line = line.strip()
        lineSplit = line.split(':')
        minute = int(lineSplit[0])
        second = float(lineSplit[1])
        t = minute * 60.0 + second
        time.append(3600.0/t)
    time = np.array(time)
    logFile.write("6_6_interface: "+str(time.mean())+"/"+str(time.std())+"\n")
    logFile.close()

if __name__=="__main__":
    calculate()