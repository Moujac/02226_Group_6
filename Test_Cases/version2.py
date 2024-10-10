import csv
from collections import defaultdict
import heapq


def read_topology(filename):
    graph = defaultdict(dict)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'LINK':
                _, link_id, source, _, dest, _, _ = row
                graph[source][dest] = 1
                graph[dest][source] = 1  
    return graph


def read_streams(filename):
    streams = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            pcp, name, stream_type, source, dest, size, period, deadline = row
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
                'min_frame_length': int(size),  
                'max_frame_length': int(size),  
                'reserved_data_rate': int(size) / int(period),  # r_f = size / period
                'burst_size': int(size)  # b_f = size
            }
            streams.append(stream)
    return streams


def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]

            if node == end:
                return cost, path

            for neighbor, weight in graph[node].items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path))

    return float('inf'), []


def calculate_per_hop_delay(stream, link_rate, interfering_streams, higher_priority_rate):

    b_H = sum(s['burst_size'] for s in interfering_streams if s['priority'] > stream['priority'])  
    b_C = sum(s['burst_size'] for s in interfering_streams if s['priority'] == stream['priority'] and s != stream)  
    b_f = stream['burst_size']
    l_f = stream['min_frame_length']
    L_f = stream['max_frame_length']
    r = link_rate
    r_H = higher_priority_rate  


    accumulated_burst = b_H + b_C + b_f - l_f
    remaining_bandwidth = r - r_H
    lower_frame_trans = L_f
    transmission_delay = L_f / r


    delay = (accumulated_burst + lower_frame_trans) / remaining_bandwidth + transmission_delay
    return delay * 1e6  


def calculate_worst_case_delay(stream, path, graph, link_rate):
    total_delay = 0
    for i in range(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        
  
        interfering_streams = find_interfering_streams(stream, graph[source], path) # find set I
        

        higher_priority_rate = sum(s['reserved_data_rate'] for s in interfering_streams if s['priority'] > stream['priority'])
        

        total_delay += calculate_per_hop_delay(stream, link_rate, interfering_streams, higher_priority_rate)
        
    return total_delay


def find_interfering_streams(current_stream, neighbors, path):
    interfering_streams = []
    for stream in streams:
        if stream['source'] in neighbors and stream['dest'] in path:
            interfering_streams.append(stream)
    return interfering_streams

def main():
    topology = read_topology('example_topology.csv')
    streams = read_streams('example_streams.csv')
    link_rate = 1e9

    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            _, path = dijkstra(topology, stream['source'], stream['dest'])
            stream['path'] = path

        
            max_e2e = calculate_worst_case_delay(stream, path, topology, link_rate)

            writer.writerow([
                stream['name'],
                f"{max_e2e:.1f}",
                stream['deadline'],
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()
