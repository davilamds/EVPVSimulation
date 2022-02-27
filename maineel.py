import sys

import numpy
import py_dss_interface
import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter
import eel


#***********CLEAR ALL (REMOVE IF NOT WANTED)
import os
os.system('cls' if os.name == 'nt' else 'clear')
print(chr(27) + "[2J")
#*********************

@eel.expose
def printhello():
    print("Hello World")

@eel.expose
def graph_irradiation2():
    radiation_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\fotovoltaico\Radiacion.dss"
    gr1 = pd.read_csv(radiation_path, delimiter=' ', header=None)
    gr = gr1.to_numpy()
    gr1_size = len(gr)

    tgi = np.arange(0, 24, 24 / len(gr))
    mp.plot(tgi, gr)
    mp.xticks(range(25))
    mp.xlim(0, 24)
    mp.grid()
    mp.ylabel("Solar irradiance [W/" + r'$m^3$' + "]")
    mp.xlabel("Time [Hours]")
    mp.savefig('irradiation.png')
    mp.show()
    
@eel.expose
def graph_pvefficiency2():
    efficiencypv_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\fotovoltaico\EffvsP.dss"
    ef1 = pd.read_csv(efficiencypv_path, delimiter=',', header=None)
    ef = ef1.to_numpy()
    ef1_size = len(ef)
    efx=ef[:,0]
    efy=ef[:,1]
    mp.plot(efx, efy)
    mp.grid()
    mp.ylabel("Efficiency [%]")
    mp.xlabel("Power [p.u.]")
    mp.show()
    
@eel.expose
def graph_pvpvst2():
    pvstpv_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\fotovoltaico\PvsT.dss"
    pvst1 = pd.read_csv(pvstpv_path, delimiter=',', header=None)
    pvst = pvst1.to_numpy()
    pvst1_size = len(pvst)
    pvstx=pvst[:,0]
    pvsty=pvst[:,1]
    mp.plot(pvstx, pvsty)
    mp.grid()
    mp.ylabel("Power [p.u.]")
    mp.xlabel("Temperature [°C]")
    mp.show()

@eel.expose
def graph_pvtemperature2():
    pvtemperature_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\fotovoltaico\Temperatura.dss"
    pvtemp1 = pd.read_csv(pvtemperature_path, delimiter=' ', header=None)
    pvtemp = pvtemp1.to_numpy()
    pvtemp1_size = len(pvtemp)

    tgi = np.arange(0, 24, 24 / len(pvtemp))
    mp.plot(tgi, pvtemp)
    mp.xticks(range(25))
    mp.xlim(0, 24)
    mp.grid()
    mp.ylabel("Temperature [°C]")
    mp.xlabel("Time [Hours]")
    mp.show()

@eel.expose
def graph_chargeprobability2():
    First_charge_pdf_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\curvas\firstchargingprobability.csv"
    First_charge_pdf1 = pd.read_csv(First_charge_pdf_path, delimiter=',', header=None)
    First_charge_pdf=First_charge_pdf1.to_numpy()
    First_charge_pdf_size=len(First_charge_pdf)
    First_charge_probability=First_charge_pdf[:,1]
    #show pdf
    tpf=np.arange(0,24,24/len(First_charge_pdf))
    mp.bar(tpf,First_charge_probability, align='edge', width=0.1)
    mp.xlabel("Time [Hours]")
    mp.ylabel("Probability [%]")
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.grid()
    mp.rcParams["font.family"]="times"
    mp.savefig('probability.png')
    mp.show()

@eel.expose
def graph_residentialloadshape2():   
    residentialloadshape_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\curvas\load_profile_1.txt"
    residentialloadshape1 = pd.read_csv(residentialloadshape_path, delimiter=' ', header=None)
    residentialloadshape=residentialloadshape1.to_numpy()
    residentialloadshape_size=len(residentialloadshape)
    
    trl1=np.arange(0,24,24/len(residentialloadshape))
    mp.plot(trl1,residentialloadshape)
    mp.ylabel("Load Power [p.u.]")
    mp.xlabel("Time [Hours]")
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.grid()
    mp.rcParams["font.family"]="times"
    mp.savefig('loadprofile.png')
    mp.show()

 
@eel.expose
def programa(js_simulationtime,
             js_simulationtimegroup,
             js_montecarlo,
             js_masterdssfile,
             js_simulationdaygroup,
             js_pvpenetration,
             js_enablepvpenetration,
             js_enablepv,
             js_irradiationdssfile,
             js_efficiencydssfile,
             js_ptdssfile,
             js_temperaturedssfile,
             js_setpointdelay,
             js_averagechargingtime,
             js_evpenetration,
             js_enableevpenetration,
             js_enableev,
             js_chargeprobabilityfile,
             js_defaultchargeprobability,
             js_firstpeak,
             js_secondpeak,
             js_transmissiondelay,
             js_enabletransmissiondelay,
             js_bandwidth,
             js_paquetsize,
             js_npackets,
             js_propagationdelay,
             js_enablepropagationdelay,
             js_distance,
             js_velocity,
             js_queuingdelay,
             js_processingdelay,
             js_commnodes,
             js_loaddssfile,
             js_golaywindowsize,
             js_golaypoliorder,
             js_enablecontrol,
             js_enablecommdelays):
    print("simulation time: " +js_simulationtime)
    print("simulation time steps: " + js_simulationtimegroup)
    print("monte carlo iterations: " + js_montecarlo)
    print("master dss file:" + js_masterdssfile)
    print("simulation day selected: " + js_simulationdaygroup)
    print("PV penetration value: " + js_pvpenetration)
    print("PV penetration enabled?: " + str(js_enablepvpenetration))
    print("PV systems enabled?: " + str(js_enablepv))
    print("irradiation dss file: " + js_irradiationdssfile)
    print("efficiency dss file: " + js_efficiencydssfile)
    print("P-T dss file: " + js_ptdssfile)
    print("Temperature dss file: " + js_temperaturedssfile)
    print("EV Setpoint change delay: " + js_setpointdelay)
    print("EV average charging time: "+js_averagechargingtime)
    print("EV penetration value: " + js_evpenetration)
    print("EV penetration enabled?: " + str(js_enableevpenetration))
    print("EVs enabled?: " + str(js_enableev))
    print("Charge probability file: " + js_chargeprobabilityfile)
    print("Use default charge probability?: " + str(js_defaultchargeprobability))
    print("Charge probability first peak: " + js_firstpeak)
    print("Charge probability second peak: " + js_secondpeak)
    print("Transmission delay: " + js_transmissiondelay)
    print("Transmission delay enabled?: " + str(js_enabletransmissiondelay))
    print("Bandwidth: " + js_bandwidth)
    print("Packet size: " + js_paquetsize)
    print("Number of packets: " + js_npackets)
    print("Propagation delay: " + js_propagationdelay)
    print("Propagation delay enabled?: " + str(js_enablepropagationdelay))
    print("distance: " + js_distance)
    print("Velocity: " + js_velocity)
    print("Queuing delay: " + js_queuingdelay)
    print("Processing delay: " + js_processingdelay)
    print("Communication nodes: " + js_commnodes)
    print("Load dss file: " + js_loaddssfile)
    print("Golay filter window size: " + js_golaywindowsize)
    print("Golay filter poli order: " + js_golaypoliorder)
    print("Control filter enabled?"+str(js_enablecontrol))
    print("Communication delays enabled?" + str(js_enablecommdelays))

    dss = py_dss_interface.DSSDLL()  #DSS object
    dss_file = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\Master.dss" #DSS file path
    #lineMV_data_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\FEL_LinesMV.dss" #DSS file path
    TX_data_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\FEL_Transformers.dss" #DSS file path
    EV_data_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\FEL_EV_Storage.dss" #DSS file path
    First_charge_pdf_path = r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\curvas\firstchargingprobability.csv"

    filter_control_path=r"C:\Users\davil\Documents\pythonenv1\hibridsim\circuit\FELIZ_DSS\curvas\filterresponse.csv"
    #************SIMULATION CONFIGURATION**************
    #simulation_time=10 #15
    simulation_time=int(js_simulationtime)
    #simulation_time_units="m"
    simulation_time_units=js_simulationtimegroup
    if (simulation_time_units=="m"):
        time_steps=int(1440/simulation_time)
    elif (simulation_time_units=="s"):
        time_steps=int(86400/simulation_time)
    
    #average_charging_time=5 #hours Average charging time of clients
    average_charging_time=int(js_averagechargingtime) #hours Average charging time of clients
    #**communication delay times
    #transmission delay
    #transmission_delay_time= 0.100 #seconds
    #transmission_delay_time=int(js_transmissiondelay)
    #communication_nodes=2
    communication_nodes=int(js_commnodes)
    #bandwidth=1 #MPbs
    bandwidth=int(js_bandwidth)
    #packet_size= 8 #Bits
    packet_size=int(js_paquetsize)
    #number_of_packets=5
    number_of_packets=int(js_npackets)
    t=packet_size/(bandwidth*10000)

    if str(js_enablecommdelays) == "True":
        if str(js_enabletransmissiondelay) == "True":
            transmission_delay_time = float(js_transmissiondelay)
        elif str(js_enabletransmissiondelay) == "False":
            transmission_delay_time= t*number_of_packets
    elif str(js_enablecommdelays)=="False":
        transmission_delay_time = 0


    #propagation delay
    #propagation_delay_time= 0.100 #seconds
    #propagation_delay_time=int(js_propagationdelay)
    #distance=100 #meters
    distance=int(js_distance)

    #velocity=2.1*(10**8) #optical fiber
    #velocity=js_velocity2.1*(10**8)
    testvel=str(js_velocity).find("E")
    testvel1=len(str(js_velocity))
    testvel2=float(str(js_velocity)[0:testvel])
    testvel3 = float(str(js_velocity)[testvel+1:testvel1])
    velocity = testvel2 * (10 ** testvel3)

    if str(js_enablecommdelays) == "True":
        if str(js_enablepropagationdelay) == "True":
            propagation_delay_time = float(js_propagationdelay)
        elif str(js_enablepropagationdelay) == "False":
            propagation_delay_time = distance/velocity
    elif str(js_enablecommdelays) == "False":
        propagation_delay_time = 0

    #queuing delay
    #queuing_delay_time= 0.001 #seconds
    queuing_delay_time=float(js_queuingdelay)
    #processing delay
    #processing_delay_time= 0.001 #seconds
    processing_delay_time=float(js_processingdelay)

    total_delay_time=communication_nodes*(transmission_delay_time+propagation_delay_time+queuing_delay_time+processing_delay_time)

    if str(js_enablecommdelays) == "False":
        queuing_delay_time=0
        processing_delay_time=0
        total_delay_time=0

    #charge_change_time=2 #seconds
    charge_change_time=float(js_setpointdelay)
    total_delay_system=total_delay_time+charge_change_time
    #EV_penetration=0.9
    EV_penetration=float(js_evpenetration)
    #PV_penetration=0.9
    PV_penetration=float(js_pvpenetration)

    #Montecarlo_simulations=1
    Montecarlo_simulations=int(js_montecarlo)

    #run_ev_penetration=0 #1 enabled
    if str(js_enableevpenetration)=="True":
        run_ev_penetration = 1
    elif str(js_enableevpenetration)=="False":
        run_ev_penetration = 0

    #run_pv_penetration=0 #1 enabled
    if str(js_enablepvpenetration)=="True":
        run_pv_penetration = 1
    elif str(js_enablepvpenetration)=="False":
        run_pv_penetration = 0

    #vehicles_enabled=1 #1 enabled
    if str(js_enableev)=="True":
        vehicles_enabled = 1
    elif str(js_enableev)=="False":
        vehicles_enabled = 0

    #pvsystems_enabled=1 #1 enabled
    if str(js_enablepv)=="True":
        pvsystems_enabled = 1
    elif str(js_enablepv)=="False":
        pvsystems_enabled = 0
    
    #charging_peak1=6 #hours
    testchp = str(js_firstpeak).find("h")
    testchp1 = len(str(js_firstpeak))
    testchp2 = int(str(js_firstpeak)[0:testchp])
    testchp3 = int(str(js_firstpeak)[testchp + 1:testchp1])
    charging_peak1 = testchp2

    #charging_peak2=18 #hours
    testchpf = str(js_secondpeak).find("h")
    testchpf1 = len(str(js_secondpeak))
    testchpf2 = int(str(js_secondpeak)[0:testchpf])
    testchpf3 = int(str(js_secondpeak)[testchpf + 1:testchpf1])
    charging_peak2 = testchpf2
    
    #***************GENERATE FIRST CHARGING CONNECTION PDF*********************
    # example of kernel density estimation for a bimodal data sample
    
    # generate a sample
    
    charging_peak1_hours=int(60/simulation_time)*charging_peak1
    charging_peak2_hours=int(60/simulation_time)*charging_peak2
    modals=2
    normal1_size=10
    normal2_size=10
    charging_peak1_scale=10
    charging_peak2_scale=8
        
    sample1 = np.random.normal(size=int(time_steps)*normal1_size,loc=charging_peak1_hours,scale=charging_peak1_scale)
    sample2 = np.random.normal(size=int(time_steps)*normal2_size,loc=charging_peak2_hours,scale=charging_peak2_scale)
    sample = np.hstack((sample1, sample2))
    # fit density
    model = KernelDensity(bandwidth=2, kernel='gaussian')
    sample = sample.reshape((len(sample), 1))
    model.fit(sample)
    # sample probabilities for a range of outcomes
    values = np.asarray([value for value in range(0,time_steps)])
    values = values.reshape((len(values), 1))
    probabilities = model.score_samples(values)
    probabilities = np.exp(probabilities)
    
    
    
    #start fresh and compile
    dss.dss_clearall()
    dss.text("compile {}".format(dss_file))
    
    #**************READ FILES FOR FILTERED RESPONSE********
    EV_Power1 = pd.read_csv(filter_control_path, delimiter=' ', header=None)
    EV_Power=EV_Power1.to_numpy()
    EV_Power1_size=len(EV_Power)
    
    
    
    #*********************GIVEN PDF*********************************
    First_charge_pdf1 = pd.read_csv(First_charge_pdf_path, delimiter=',', header=None)
    First_charge_pdf=First_charge_pdf1.to_numpy()
    First_charge_pdf_size=len(First_charge_pdf)
    First_charge_probability=First_charge_pdf[:,1]
    
    
    
    
    
    
    #**************CREATE MONITORS ON ELEMENTS**************************
    #POWER
    #LINES
    lines_names = dss.lines_allnames()
    lines_names_size_size = len(lines_names)
    for i in range(lines_names_size_size):
        dss.text("new monitor." + lines_names[i] +"_P" + " element=line." + lines_names[i] + " terminal=1 mode=1 ppolar=no")
    
    #TRANSFORMERS
    transformers_names = dss.transformers_allNames()
    transformers_size = len(transformers_names)
    for i in range(transformers_size):
        dss.text("new monitor." + transformers_names[i] +"_P"  + " element=transformer." + transformers_names[i] + " terminal=1 mode=1 ppolar=no")
    
    #LOADS
    circuit_loads_names = dss.loads_allnames()
    circuit_loads_size = len(circuit_loads_names)
    for i in range(circuit_loads_size):
        dss.text("new monitor." + circuit_loads_names[i] +"_P"  + " element=load." + circuit_loads_names[i] + " terminal=1 mode=1 ppolar=no")
    
    #PVs
    circuit_PV_names = dss.pvsystems_allnames()
    circuit_PV_size = len(circuit_PV_names)
    for i in range(circuit_PV_size):
        dss.text("new monitor." + circuit_PV_names[i] +"_P"   + " element=PVSystem." + circuit_PV_names[i] + " terminal=1 mode=0 ppolar=no")
    
    #STORAGES
    dss.circuit_setactiveclass("Storage")
    circuit_evs_names = dss.activeclass_allnames()
    circuit_evs_size = len(circuit_evs_names)
    for i in range(circuit_evs_size):
        dss.text("new monitor." + circuit_evs_names[i] +"_P"  + " element=storage." + circuit_evs_names[i] + " terminal=1 mode=1 ppolar=no")
        
    #VOLTAGES
    #LINES
    lines_names = dss.lines_allnames()
    lines_names_size_size = len(lines_names)
    for i in range(lines_names_size_size):
        dss.text("new monitor." + lines_names[i] +"_V" + " element=line." + lines_names[i] + " terminal=1 mode=0 ppolar=no")
    
    #TRANSFORMERS
    transformers_names = dss.transformers_allNames()
    transformers_size = len(transformers_names)
    for i in range(transformers_size):
        dss.text("new monitor." + transformers_names[i] +"_V"  + " element=transformer." + transformers_names[i] + " terminal=1 mode=0 ppolar=no")
    
    #LOADS
    circuit_loads_names = dss.loads_allnames()
    circuit_loads_size = len(circuit_loads_names)
    for i in range(circuit_loads_size):
        dss.text("new monitor." + circuit_loads_names[i] +"_V"  + " element=load." + circuit_loads_names[i] + " terminal=1 mode=0 ppolar=no")
    
    #PVs
    circuit_PV_names = dss.pvsystems_allnames()
    circuit_PV_size = len(circuit_PV_names)
    for i in range(circuit_PV_size):
        dss.text("new monitor." + circuit_PV_names[i] +"_V"   + " element=PVSystem." + circuit_PV_names[i] + " terminal=1 mode=0 ppolar=no")
    
    #STORAGES
    dss.circuit_setactiveclass("Storage")
    circuit_evs_names = dss.activeclass_allnames()
    circuit_evs_size = len(circuit_evs_names)
    for i in range(circuit_evs_size):
        dss.text("new monitor." + circuit_evs_names[i] +"_V"  + " element=storage." + circuit_evs_names[i] + " terminal=1 mode=1 ppolar=no")
    
    
    #***************ENABLE/DISABLE VEHICLES FOR SIMULATION**********
    
    if vehicles_enabled==0:
        dss.circuit_setactiveclass("Storage")
        getpowerev_names = dss.activeclass_allnames()
        getpowerev_size = len(getpowerev_names)
        for i in range(getpowerev_size):
            dss.activeclass_write_name(str(getpowerev_names[i]))
            dss.cktelement_write_enabled(0)
    elif vehicles_enabled==1:
        dss.circuit_setactiveclass("Storage")
        getpowerev_names = dss.activeclass_allnames()
        getpowerev_size = len(getpowerev_names)
        for i in range(getpowerev_size):
            dss.activeclass_write_name(str(getpowerev_names[i]))
            dss.cktelement_write_enabled(1)
    
    if pvsystems_enabled==0:
        dss.circuit_setactiveclass("PVSystem")
        getpowerpv_names= dss.pvsystems_allnames()
        getpowerpv_size= len(getpowerpv_names)
        for i in range(getpowerpv_size):
            dss.activeclass_write_name(str(getpowerpv_names[i]))
            dss.cktelement_write_enabled(0)
            #dss.text("PVSystem."+getpowerev_names[i]+".enabled="+str(0))
    elif pvsystems_enabled==1:
        dss.circuit_setactiveclass("PVSystem")
        getpowerpv_names= dss.pvsystems_allnames()
        getpowerpv_size= len(getpowerpv_names)
        for i in range(getpowerpv_size):
            dss.activeclass_write_name(str(getpowerpv_names[i]))
            dss.cktelement_write_enabled(1)
            #dss.text("PVSystem."+getpowerev_names[i]+".enabled="+str(0))
        
    #************GET TOTAL LOAD POWER************
    #TOTAL LOAD=LOAD+EV
    dss.circuit_setactiveclass("Load")
    getpowerload_names= dss.loads_allnames()
    getpowerload_size= len(getpowerload_names)
    powerload=0
    evload=0
    for i in range(getpowerload_size):
        #powerload=powerload+dss.loads_read_kw()
        powerload=powerload+float(dss.text("? Load."+getpowerload_names[i]+".kW"))
    
    dss.circuit_setactiveclass("Storage")
    getpowerev_names = dss.activeclass_allnames()
    getpowerev_size = len(getpowerev_names)
    for i in range(getpowerev_size):
        evload=evload+float(dss.text("? Storage."+getpowerev_names[i]+".kWrated"))
    
    totalpowerloads=powerload+evload
    
    #***********Get PV power**********
    dss.circuit_setactiveclass("PVSystem")
    getpowerpv_names= dss.pvsystems_allnames()
    getpowerpv_size= len(getpowerpv_names)
    powerpv=0
    for i in range(getpowerpv_size):
        dss.pvsystems_write_name(getpowerpv_names[i])
        powerpv=powerpv+dss.pvsystems_read_kvarated()
    
    #**********CALCULATED PENETRATIONS*****
    EV_penetration_calculated=evload/totalpowerloads
    PV_penetration_calculated=powerpv/totalpowerloads
    
    if run_pv_penetration==1:
        powerpv=0
        for i in range(getpowerpv_size):
            dss.pvsystems_write_name(getpowerpv_names[i])
            print(str(dss.activeclass_read_name()))
            print(dss.pvsystems_read_kvarated())
            
        while PV_penetration > PV_penetration_calculated:
            
            dss.circuit_setactiveclass("PVSystem")
            getpowerpv_names= dss.pvsystems_allnames()
            getpowerpv_size= len(getpowerpv_names)
            for i in range(getpowerpv_size):
                pvkw=0
                dss.pvsystems_write_name(getpowerpv_names[i])
                pvkw=dss.pvsystems_read_kvarated()
                pvkw=pvkw+1
                dss.pvsystems_write_kvarated(pvkw)
                dss.text('PVSystem.'+getpowerpv_names[i]+'.kVA='+str(pvkw))
                powerpv=powerpv+1
            PV_penetration_calculated=powerpv/totalpowerloads
        print("run pv penetration")
        dss.circuit_setactiveclass("PVSystem")
        getpowerpv_names= dss.pvsystems_allnames()
        getpowerpv_size= len(getpowerpv_names)
        for i in range(getpowerpv_size):
            dss.pvsystems_write_name(getpowerpv_names[i])
            print(str(dss.activeclass_read_name()))
            print(dss.pvsystems_read_kvarated())
            
    elif run_ev_penetration==1:
        evload=0
        while EV_penetration > EV_penetration_calculated:
            
            dss.circuit_setactiveclass("Storage")
            getpowerev_names = dss.activeclass_allnames()
            getpowerev_size = len(getpowerev_names)
            for i in range(getpowerev_size):
                evkw=0
                dss.activeclass_write_name(getpowerev_names[i])
                evkw=float(dss.text("? Storage."+getpowerev_names[i]+".kWrated"))
                evkw=evkw+1
                dss.text("Storage."+getpowerev_names[i]+".kWrated="+str(evkw))
                evload=evload+1
            EV_penetration_calculated=evload/totalpowerloads
        print("run ev penetration")
        #chech updates
       
        dss.circuit_setactiveclass("Storage")
        getpowerev_names = dss.activeclass_allnames()
        getpowerev_size = len(getpowerev_names)
        for i in range(getpowerev_size):
            print(str(dss.activeclass_read_name()))
            print(dss.text("? Storage."+getpowerev_names[i]+".kWrated"))
        
    
    #*************check probability
    
    First_charge_probability_all=np.multiply(First_charge_probability,14*14/100)
    
    vehiclesconnected=np.round(First_charge_probability_all,0)
    
    
    
    
    
    #############################################################################
    ###############SIMULATION START####################################

    if Montecarlo_simulations==1:
        #norand
        vec=np.array([1.4144,22.4013,7.63813,12.3265,4.62854,3.59742,8.21634,16.5266,4.33863,2.92903,19.5853,21.8325,11.6189,18.744,1.4144, 22.4013, 7.63813, 12.3265, 4.62854, 3.59742, 8.21634, 16.5266, 4.33863, 2.92903, 19.5853, 21.8325, 11.6189, 18.744,1.4144, 22.4013, 7.63813, 12.3265, 4.62854, 3.59742, 8.21634, 16.5266, 4.33863, 2.92903, 19.5853, 21.8325, 11.6189, 18.744,1.4144, 22.4013, 7.63813, 12.3265, 4.62854, 3.59742, 8.21634, 16.5266, 4.33863, 2.92903, 19.5853, 21.8325, 11.6189])
        vecc=np.reshape(vec, (1,len(vec)))
        vecdisconect=np.add(vec,average_charging_time+total_delay_system)
    
    
        dss.circuit_setactiveclass("Storage")
        getpowerev_names = dss.activeclass_allnames()
        getpowerev_size = len(getpowerev_names)
        for i in range(getpowerev_size):
            dss.text("Storage."+getpowerev_names[i]+".TimeChargeTrig="+str(vec[i]))
        m=0
        i=0
        j=0
        time_simulation=0
        
        testpower_i=int(len(EV_Power)/time_steps)
        
        testpower=EV_Power[::testpower_i]
        dss.text('Set ControlMode = time') #Defines the control mode
        dss.text('Reset') #Resets all energy meters and monitors
        dss.text("Set Mode = daily stepsize = " + str(simulation_time) +simulation_time_units+" number = 1")
        output_charging_events=[]
        for time_simulation in range(time_steps):
            dss.solution_solve()
            
            #here goes control
            actualtime=dss.solution_read_dblhour()
            dss.circuit_setactiveclass("Storage")
            getpowerev_names = dss.activeclass_allnames()
            getpowerev_size = len(getpowerev_names)
            
                    
            for i in range(getpowerev_size):
                evtime=float(dss.text("? Storage."+getpowerev_names[i]+".TimeChargeTrig"))
                if actualtime>evtime:
                    dss.activeclass_write_name(getpowerev_names[i])
                    dss.text("Storage."+getpowerev_names[i]+".state=charging")
                if actualtime>vecdisconect[i]:
                    dss.text("Storage."+getpowerev_names[i]+".state=idling")
                    dss.activeclass_write_name(str(getpowerev_names[i]))
                    dss.cktelement_write_enabled(0)
        
                check_name=dss.activeclass_read_name()
                check_state=dss.text("? Storage."+getpowerev_names[i]+".state")
                check_enabled=dss.text("? Storage."+getpowerev_names[i]+".enabled")
                check_soc=dss.text("? Storage."+getpowerev_names[i]+".%stored")
                dss.text("Storage."+getpowerev_names[i]+".kWrated="+str(testpower[time_simulation]))
                check_EV_Power=dss.text("? Storage."+getpowerev_names[i]+".kWrated")
                output_charging_events.append([actualtime,check_name,check_state,check_soc,check_enabled,check_EV_Power])
    
        cccc=pd.DataFrame(output_charging_events)
        #obtain data from monitor
        dss.circuit_setactiveclass("Monitor")
        dss.monitors_write_name("TR1_P")
        monitors_header=dss.monitors_header()
        a0=dss.monitors_channel(0)
        a1=dss.monitors_channel(1)
        a2=dss.monitors_channel(2)
        a3=dss.monitors_channel(3)
        a4=dss.monitors_channel(4)
        a5=dss.monitors_channel(5)
        a6=dss.monitors_channel(6)
        
        dss.circuit_setactiveclass("Monitor")
        dss.monitors_write_name("LOAD1_V")
        monitors_header=dss.monitors_header()
        b0=dss.monitors_channel(0)
        b1=dss.monitors_channel(1)
        b2=dss.monitors_channel(2)
        b3=dss.monitors_channel(3)
        b4=dss.monitors_channel(4)
        b5=dss.monitors_channel(5)
        b6=dss.monitors_channel(6)


    ##########################################################################
    #########################################################################
    
    dss.loadshapes_write_name('radiacion')
    pvloadshape=dss.loadshapes_read_pmult()
    dss.loadshapes_write_name('Shape_1')
    residentialloadshape=dss.loadshapes_read_pmult()
    
    pvloadshape=np.multiply(pvloadshape,1000) #0.7
    
    #pvloadshapefiltered = savgol_filter(pvloadshape, 101, 3)# window size 51, polynomial order 3
    pvloadshapefiltered = savgol_filter(pvloadshape, int(js_golaywindowsize), int(js_golaypoliorder))# window size 51, polynomial order 3
    
    signal1=pvloadshape-pvloadshapefiltered
    signal2=pvloadshapefiltered-signal1
    
    
    yhat=pvloadshapefiltered
    V1=0
    V2=0
    S2=0
    S1=0
    S=0
    E2=0
    E1=0
    E=0
    Carga=0
    
    P_iny=[]
    P_ev=[]
    min_err=0#0.5
    for i in range(len(yhat)):
        E=yhat[i]
        S=1.824*S1-0.8389*S2+(E*0.003615)+(E1*0.007231)+(E2*0.003615)
    
        err=E-S
        if err<min_err:
            S=E-min_err
    
        if E<min_err:
            S=0
    
        S1=S
        S2=S1
        E1=E
        E2=E1
    
        Carga=E-S
    
        if Carga> 7.2:
            Carga=7.2
        if Carga<0:
            Carga=0
            
        P_iny.append(E-Carga)
        P_ev.append(Carga)
    
    
    
    
    # #**************DRAW FIGURES**************
    
    
    #mp.plot(yhat,color='blue',label='yhat')
    #mp.twinx()
    
    
    t=np.arange(0,24,24/time_steps)
    mp.plot(t,a1,t,a3,t,a5)
    mp.xlabel("Time [Hours]")
    mp.ylabel("Power [kW]")
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.grid()
    mp.legend(['Phase A', 'Phase B', 'Phase C'])
    mp.rcParams["font.family"]="Times"
    mp.show()
    
    
    t2=np.arange(0,24,24/len(pvloadshape))
    mp.plot(t2,pvloadshape)
    mp.plot(t2,pvloadshapefiltered)
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.xlim([6,15])
    mp.grid()
    mp.ylabel("Solar irradiance [W/"+r'$m^3$'+"]")
    mp.xlabel("Time [Hours]")
    mp.show()
    
    t3=np.arange(0,24,24/len(P_ev))
    mp.plot(t3,P_ev, color='orange',label='P_ev')
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.xlim([6,15])
    mp.grid()
    mp.ylabel("EV Power [kW]")
    mp.xlabel("Time [Hours]")
    mp.show()
    
    
    t=np.arange(0,24,24/time_steps)
    mp.plot(t,b1,t,b2,t,b3)
    mp.xlabel("Time [Hours]")
    mp.ylabel("Voltage [V]")
    mp.xticks(range(25))
    mp.xlim(0,24)
    mp.grid()
    mp.legend(['Phase A', 'Phase B', 'Phase C'])
    mp.rcParams["font.family"]="Times"
    mp.show()

    b11=np.array(a1)
    b22=np.array(a3)
    b33=np.array(a5)
    b=b11+b22+b33
    t4 = np.arange(0, 24, 24 / time_steps)
    mp.plot(t4, b)
    mp.xlabel("Time [Hours]")
    mp.ylabel("Power [kW]")
    mp.xticks(range(25))
    mp.xlim(0, 24)
    mp.grid()
    mp.legend(['TR1'])
    mp.rcParams["font.family"] = "Times"
    mp.show()


    print(b1)
    #numpy.set_printoptions(threshold=sys.maxsize)
    print(numpy.ndarray.tolist(b))

eel.init("www")
eel.start("index.html")