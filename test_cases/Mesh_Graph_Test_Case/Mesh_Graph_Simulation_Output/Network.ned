package tsn;

import inet.networks.base.TsnNetworkBase;
import inet.node.ethernet.Eth1G;
import inet.node.tsn.TsnDevice;
import inet.node.tsn.TsnSwitch;

network TSN_Network extends TsnNetworkBase{
    @display("bgb=1000,1000");
    submodules:
        Switch_1: TsnSwitch {}
        Switch_2: TsnSwitch {}
        Switch_3: TsnSwitch {}
        Switch_4: TsnSwitch {}
        Switch_5: TsnSwitch {}
        Switch_6: TsnSwitch {}
        ES_1: TsnDevice {}
        ES_2: TsnDevice {}
        ES_3: TsnDevice {}
        ES_4: TsnDevice {}
        ES_5: TsnDevice {}
        ES_6: TsnDevice {}
        ES_7: TsnDevice {}
        ES_8: TsnDevice {}
        ES_9: TsnDevice {}
        ES_10: TsnDevice {}
        ES_11: TsnDevice {}
        ES_12: TsnDevice {}
        ES_13: TsnDevice {}
        ES_14: TsnDevice {}
        ES_15: TsnDevice {}
        ES_16: TsnDevice {}
        ES_17: TsnDevice {}
        ES_18: TsnDevice {}
        ES_19: TsnDevice {}
        ES_20: TsnDevice {}
        ES_21: TsnDevice {}
        ES_22: TsnDevice {}
        ES_23: TsnDevice {}
        ES_24: TsnDevice {}
    connections:
        ES_1.ethg++ <--> Eth1G <--> Switch_1.ethg++;
        ES_2.ethg++ <--> Eth1G <--> Switch_1.ethg++;
        ES_3.ethg++ <--> Eth1G <--> Switch_1.ethg++;
        ES_4.ethg++ <--> Eth1G <--> Switch_1.ethg++;
        ES_5.ethg++ <--> Eth1G <--> Switch_2.ethg++;
        ES_6.ethg++ <--> Eth1G <--> Switch_2.ethg++;
        ES_7.ethg++ <--> Eth1G <--> Switch_2.ethg++;
        ES_8.ethg++ <--> Eth1G <--> Switch_2.ethg++;
        ES_9.ethg++ <--> Eth1G <--> Switch_3.ethg++;
        ES_10.ethg++ <--> Eth1G <--> Switch_3.ethg++;
        ES_11.ethg++ <--> Eth1G <--> Switch_3.ethg++;
        ES_12.ethg++ <--> Eth1G <--> Switch_3.ethg++;
        ES_13.ethg++ <--> Eth1G <--> Switch_4.ethg++;
        ES_14.ethg++ <--> Eth1G <--> Switch_4.ethg++;
        ES_15.ethg++ <--> Eth1G <--> Switch_4.ethg++;
        ES_16.ethg++ <--> Eth1G <--> Switch_4.ethg++;
        ES_17.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        ES_18.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        ES_19.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        ES_20.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        ES_21.ethg++ <--> Eth1G <--> Switch_6.ethg++;
        ES_22.ethg++ <--> Eth1G <--> Switch_6.ethg++;
        ES_23.ethg++ <--> Eth1G <--> Switch_6.ethg++;
        ES_24.ethg++ <--> Eth1G <--> Switch_6.ethg++;
        Switch_1.ethg++ <--> Eth1G <--> Switch_4.ethg++;
        Switch_1.ethg++ <--> Eth1G <--> Switch_2.ethg++;
        Switch_2.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        Switch_2.ethg++ <--> Eth1G <--> Switch_3.ethg++;
        Switch_3.ethg++ <--> Eth1G <--> Switch_6.ethg++;
        Switch_4.ethg++ <--> Eth1G <--> Switch_5.ethg++;
        Switch_5.ethg++ <--> Eth1G <--> Switch_6.ethg++;
}