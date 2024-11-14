sim_data means should be as follows:
// for endpoints!!!
// min, max values are weird need to sanity check this!!!
ES_1 = 0.000036 = MIN = MAX = MEAN
ES_2 = 0.000044 = MIN = MAX = MEAN
ES_3 = 0.000018 = MIN = MAX = MEAN
ES_4, MEAN = 0.000039, MIN = 0.000033, MAX = 0.000046
ES_5 = 0.000019 = MIN = MAX = MEAN
ES_6 = 0.000028 = MIN = MAX = MEAN
ES_7, MEAN = MIN = 0.000044, MAX = 0.000047
ES_8 = 0.000031 = MIN = MAX = MEAN

IMNETPP.INI changes:
// maybe play more with these later!!!
*.*.bridging.directionReverser.delayer.delay = 0us
*.Switch*.bridging.streamFilter.ingress.meter[0].committedInformationRate = 1000Mbps
*.Switch*.bridging.streamFilter.ingress.meter[0].committedBurstSize = 5000B