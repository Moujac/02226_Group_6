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
                # Assuming unit weight for simplicity. Can be modified for weighted edges.
                graph[source][dest] = 1
                graph[dest][source] = 1  # Assuming bidirectional links
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
                'priority': int(pcp),  # Assuming PCP represents priority
                'min_frame_length': int(size),  # Assuming minimum frame length is the given size
                'max_frame_length': int(size),  # Assuming maximum frame length is the given size
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

def calculate_per_hop_delay(stream, link_rate):
    # This is a simplified version of the formula. In a real implementation,
    # you would need to consider all streams sharing the same link and their interactions.
    b_H = 1500  # Assuming maximum frame size of higher priority traffic
    b_C = 1500  # Assuming maximum frame size of same priority traffic
    b_f = stream['burst_size']
    l_f = stream['min_frame_length']
    L_f = stream['max_frame_length']
    r = link_rate
    r_H = link_rate / 2  # Assuming higher priority traffic takes half the link rate

    delay = (b_H + b_C + b_f - l_f + L_f) / (r - r_H) + L_f / r
    return delay * 1e6  # Convert to microseconds

def calculate_worst_case_delay(stream, path, link_rate):
    total_delay = 0
    for _ in range(len(path) - 1):  # For each hop
        total_delay += calculate_per_hop_delay(stream, link_rate)
    return total_delay

def main():
    topology = read_topology('example_topology.csv')
    streams = read_streams('example_streams.csv')
    link_rate = 1e9  # Assuming 1 Gbps links

    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            _, path = dijkstra(topology, stream['source'], stream['dest'])
            stream['path'] = path
            
            # Calculate worst-case delay
            max_e2e = calculate_worst_case_delay(stream, path, link_rate)

            writer.writerow([
                stream['name'],
                f"{max_e2e:.1f}",
                stream['deadline'],
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()