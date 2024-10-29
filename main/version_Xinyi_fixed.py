import csv
from collections import defaultdict
import heapq
from collections import deque

# Removed unused item for reading correctly
# fixed index error caused by topology name not starting with 0 anymore... -_-
def read_topology(filename):
    shaped_queue = {}
    graph = defaultdict(dict)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'LINK':
                _, link_id, source, source_port, dest, dest_port = row
                graph[source][dest] = (int(source_port), int(dest_port))
                graph[dest][source] = (int(dest_port), int(source_port))  
            if row[0] == 'SW':
                _, switch, ports = row
                shaped_queue[switch] = {port: dict() for port in range(1, int(ports)+1)}
                for port in shaped_queue[switch]:
                    shaped_queue[switch][port] = {priority: list() for priority in range(8)}
            if row[0] == 'EN':
                _,end, ports = row
                shaped_queue[end] = {port: dict() for port in range(1, int(ports)+1)}
                for port in shaped_queue[switch]:
                    shaped_queue[end][port] = {priority: list() for priority in range(8)}
    return graph, shaped_queue

# Changed to ignore header
def read_streams(filename):
    streams = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
                pcp, name, stream_type, source, dest, size, period, deadline = row
                if (pcp != "PCP"): # Could maybe be done more elegant, im lazy 
                    stream = {
                        'pcp': int(pcp),
                        'name': name,
                        'type': stream_type,
                        'source': source,
                        'dest': dest,
                        'size': int(size),
                        'period': int(period),
                        'deadline': int(deadline),
                        'path': [],
                        'priority': int(pcp),  
                        'frame_length': int(size),  
                        'reserved_rate': int(size) / int(period),  # r_f = size / period
                        'burst_size': int(size)  # b_f = size
                    }
                    streams.append(stream)
    return streams


def dijkstra(graph, stream, shaped_queue):
    queue = [(stream['source'], [])]
    visited = set()

    while queue:
        (node, path) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]

            if node == stream['dest']:
                for i in range(len(path)-1):
                    (source_port, dest_port) = graph[path[i]][path[i+1]]
                    if path[i+1] in shaped_queue.keys():
                        shaped_queue[path[i+1]][dest_port][stream['priority']].append(stream) 
                        # implement qar: streams from different ports and have different priorities are allocated into different shaped queues
                return path, shaped_queue

            for neighbor, (source_port, dest_port) in graph[node].items():
                if neighbor not in visited:
                    heapq.heappush(queue, (neighbor, path))

    return float('inf'), []


def cal_hop_delay(streams_same_shaped_queue, streams_high_pri, streams_same_pri, streams_low_pri, link_rate):
    delay = 0
    if streams_high_pri:
        b_H = sum(stream['burst_size'] for stream in streams_high_pri)
        r_H = sum(stream['reserved_rate'] for stream in streams_high_pri)
    else: 
        b_H = 0
        r_H = 0
    if streams_same_pri:
        b_C = sum(stream['burst_size'] for stream in streams_same_pri)
    else:
        b_C = 0
    if streams_low_pri:
        l_L = max(stream['frame_length'] for stream in streams_low_pri)
    else:
        l_L = 0
    
    for stream in streams_same_shaped_queue:
        b_j = stream['burst_size']
        l_j = stream['frame_length']
        stream_delay = (b_H + b_C + b_j - l_j + l_L) / (link_rate - r_H) + l_j / link_rate
        delay = max(delay, stream_delay)
    return delay * 1e6


def calculate_worst_case_delay(stream, shaped_queue, link_rate, topology):
    total_delay = 0
    for i in range(len(stream['path'])-2): # why minus 2?, need to consider this
        node = stream['path'][i]
        next_node = stream['path'][i+1]
        (source_port, dest_port) = topology[node][next_node]
        streams_same_shaped_queue = shaped_queue[next_node][dest_port][stream['priority']]
        streams_high_pri = []
        for port in shaped_queue[next_node]:
            for priority in range(stream['priority']+1, 8):
                if shaped_queue[next_node][port][priority]:
                    streams_high_pri.extend(shaped_queue[next_node][port][priority])
        streams_same_pri = []
        for port in shaped_queue[next_node]:
            if shaped_queue[next_node][port][stream['priority']]:
                streams_same_pri.extend(shaped_queue[next_node][port][stream['priority']])
        streams_low_pri = []
        for port in shaped_queue[next_node]:
            for priority in range(stream['priority']):
                if shaped_queue[next_node][port][priority]:
                    streams_low_pri.extend(shaped_queue[next_node][port][priority])
        hop_delay = cal_hop_delay(streams_same_shaped_queue, streams_high_pri, 
                              streams_same_pri, streams_low_pri, link_rate)
        total_delay += hop_delay

    return total_delay



def main():
    topology, shaped_queue = read_topology('../test_cases/test_case_biomial/topology.csv')
    streams = read_streams('../test_cases/test_case_biomial/streams.csv')
    link_rate = 1e8

    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            path, shaped_queue = dijkstra(topology, stream, shaped_queue)
            stream['path'] = path
        
            max_e2e = calculate_worst_case_delay(stream, shaped_queue, link_rate, topology)

            writer.writerow([
                stream['name'],
                f"{max_e2e:.1f}",
                stream['deadline'],
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()