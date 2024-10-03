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
                'deadline': int(deadline)
            }
            # Calculate ATS-specific parameters
            stream['burst_size'] = stream['size']  # b = size
            stream['committed_rate'] = stream['size'] / stream['period']  # r = size / period
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

def calculate_worst_case_delay(stream, path_length):
    # This is a simplified calculation and should be replaced with a more accurate model
    # based on network calculus or other appropriate methods for ATS streams
    transmission_time = stream['size'] * 8 / (10 ** 9)  # Assuming 1 Gbps links
    propagation_delay = 0.1 * path_length  # Assuming 0.1 ms propagation delay per hop
    queuing_delay = stream['size'] / stream['committed_rate']
    
    return (transmission_time + propagation_delay + queuing_delay) * 10**6  # Convert to microseconds

def main():
    topology = read_topology('example_topology.csv')
    streams = read_streams('example_streams.csv')

    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            cost, path = dijkstra(topology, stream['source'], stream['dest'])
            
            # Calculate worst-case delay
            max_e2e = calculate_worst_case_delay(stream, len(path) - 1)

            writer.writerow([
                stream['name'],
                f"{max_e2e:.1f}",
                stream['deadline'],
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()