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


def calculate_per_hop_delay(link_rate, interfering_streams, higher_priority_rate):
    delay = 0
    for stream in interfering_streams:
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

        delay = max(delay, (accumulated_burst + lower_frame_trans) / remaining_bandwidth + transmission_delay)

    return delay * 1e6  


def calculate_worst_case_delay(all_streams, stream, path, graph, link_rate):
    total_delay = 0
    interfering_streams = find_interfering_streams(all_streams, stream, path)  # find set I

    for i in range(len(path) - 1):
        source = path[i]
        # dest = path[i + 1]

        interfering_streams_at_node = interfering_streams.get(source, [])
        higher_priority_rate = sum(s['reserved_data_rate'] for s in interfering_streams_at_node if s['priority'] > stream['priority'])
        
        total_delay += calculate_per_hop_delay(link_rate, interfering_streams_at_node, higher_priority_rate)
    
    return total_delay

def find_interfering_streams(streams, current_stream, path):
    interfering_streams_per_node = {}  

    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        
        interfering_streams_per_node[current_node] = []

        for stream in streams:
            if stream == current_stream:
                continue  

            if stream['source'] == current_stream['source'] and stream['dest'] == current_stream['dest']:
                continue  

            stream_path = stream.get('path', [])
            if current_node not in stream_path or next_node not in stream_path:
                continue  

            if not violates_qars(current_stream, stream, current_node):
                interfering_streams_per_node[current_node].append(stream)

    return interfering_streams_per_node


def violates_qars(current_stream, interfering_stream, current_node):

    # QAR1: two stream come from different port
    if current_stream['source'] != interfering_stream['source']:
        return True

    # QAR2: two streams coming from the same port but have different priorities
    if current_stream['source'] == interfering_stream['source'] and current_stream['priority'] != interfering_stream['priority']:
        return True

    # QAR3: two streams sent by the same node but have different priorities
    if current_node == current_stream['source'] or current_node == interfering_stream['source']:
        if current_stream['priority'] != interfering_stream['priority']:
            return True

    return False


def main():
    topology = read_topology('./Small_Case/small-topology.csv')
    streams = read_streams('./Small_Case/small-streams.csv')
    link_rate = 1e9

    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            _, path = dijkstra(topology, stream['source'], stream['dest'])
            stream['path'] = path

        
            max_e2e = calculate_worst_case_delay(streams, stream, path, topology, link_rate)

            writer.writerow([
                stream['name'],
                f"{max_e2e:.1f}",
                stream['deadline'],
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()
