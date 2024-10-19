import pandas as pd  # For data manipulation
import networkx as nx  # For graph-based operations
from dataclasses import dataclass, field  # For structured data with default behavior
from typing import List  # For type hinting lists
from enum import Enum  # For defining enumerations

# Define stream types (ATS, AVB, etc.) using an enumeration
class StreamType(Enum):
    ATS = "ATS"
    AVB = "AVB"

# Data class to represent a Stream
@dataclass
class Stream:
    pcp: int  # Priority Code Point (PCP)
    name: str  # Stream name
    type: StreamType  # Type of stream (ATS, AVB, etc.)
    source: str  # Source node in the network
    dest: str  # Destination node in the network
    size: int  # Size of the stream in bytes
    period: int  # Transmission period in microseconds
    deadline: int  # End-to-end deadline in microseconds
    path: List[str] = field(default_factory=list)  # Network path (empty by default)
    worst_case_delay: float = 0.0  # Calculated worst-case latency (default is 0)

class StreamTokenBucket:
    def __init__(self, rate, capacity):
        self.tokens = 0  # Current tokens
        self.rate = rate  # Rate of token accumulation
        self.capacity = capacity  # Maximum capacity of tokens

    def replenish(self, time_passed):
        """Replenish tokens based on the rate and time passed."""
        self.tokens += self.rate * time_passed
        self.tokens = min(self.tokens, self.capacity)  # Cap tokens at the maximum capacity

    def consume(self, size):
        """Consume tokens for the given frame size."""
        if self.tokens >= size:
            self.tokens -= size
            return True
        return False

class ShapedQueue:
    def __init__(self):
        self.queue = []  # Frames waiting for transmission
        self.stream_buckets = {}  # Token buckets for each stream

    def add_frame(self, frame):
        """Add a frame to the queue based on its stream token bucket."""
        if frame.name not in self.stream_buckets:
            # Initialize a token bucket for the stream if it doesn't exist
            self.stream_buckets[frame.name] = StreamTokenBucket(rate=frame.period, capacity=frame.size)  # Adjust rate and capacity as needed

        # Attempt to consume tokens for the frame
        if self.stream_buckets[frame.name].consume(frame.size):
            self.queue.append(frame)  # If tokens were consumed, add to the queue
        else:
            # Frame must wait if tokens aren't sufficient
            self.queue.append(frame)

    def replenish_tokens(self, time_passed):
        """Replenish tokens for all streams in the queue based on their specific rates."""
        for frame in self.queue:
            bucket = self.stream_buckets[frame.name]
            bucket.replenish(time_passed)

    def get_next_frame(self):
        """Retrieve the next frame from the queue, considering head-of-line blocking."""
        if not self.queue:
            return None

        head_frame = self.queue[0]  # Get the frame at the head of the queue
        if self.stream_buckets[head_frame.name].consume(head_frame.size):
            return self.queue.pop(0)  # If it can be transmitted, remove and return it
        return None  # Return None if the head cannot be transmitted

# Read stream data from a CSV file and convert each row into a Stream object
def read_streams(file_path):
    # Define CSV column headers
    streams_columns = ["PCP", "StreamName", "StreamType", "SourceNode", "DestinationNode", "Size", "Period", "Deadline"]
    streams_df = pd.read_csv(file_path, names=streams_columns, header=None)  # Read CSV into DataFrame
    streams_list = []
    
    # Convert each DataFrame row into a Stream object
    for _, row in streams_df.iterrows():
        stream = Stream(
            pcp=int(row['PCP']),
            name=row['StreamName'],
            type=StreamType(row['StreamType']),
            source=row['SourceNode'],
            dest=row['DestinationNode'],
            size=int(row['Size']) + 42,  # Add Ethernet overhead
            period=int(row['Period']),
            deadline=int(row['Deadline'])
        )
        streams_list.append(stream)  # Add to list
    return streams_list

# Read network topology from a CSV file and build the graph
def read_topology(file_path):
    G = nx.Graph()  # Initialize graph
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line in the file
    for line in lines:
        parts = line.strip().split(',')
        if parts[0] == 'LINK':
            # If line represents a link, add an edge between nodes
            _, link_id, source, source_port, dest, dest_port, domain = parts
            G.add_edge(source, dest, weight=1)  # Add edge with default weight
        else:
            # Otherwise, line represents a device (node)
            device_type, device_name, ports, domain = parts
            G.add_node(device_name, type=device_type, ports=int(ports), domain=domain)  # Add node
    return G

# Compute shortest path in the network for each stream using Dijkstra's algorithm
def find_shortest_path(streams, graph):
    for stream in streams:
        try:
            # Find shortest path from source to destination
            path = nx.shortest_path(graph, source=stream.source, target=stream.dest, weight='weight')
            stream.path = path  # Store the computed path
        except nx.NetworkXNoPath:
            print(f"No path found for {stream.name}")  # Log if no path exists

# Initialize shaped queues and place streams into their corresponding queues based on QAR rules
def initialize_queues(streams, topology_graph):
    queues = {}
    m = 8  # Number of priority levels (0-7)
    
    for node in topology_graph.nodes():
        queues[node] = {}

        # Initialize all priority levels and shaped queues for the node
        for pcp in range(m):
            queues[node][pcp] = {'shaped_queues': {}, 'ready_queue': ShapedQueue()}

            # Assume the number of ingress ports is equal to the number of adjacent nodes
            ingress_ports = list(topology_graph.neighbors(node))
            for ingress_port in ingress_ports:
                # Initialize a shaped queue for each ingress port (per priority level)
                queues[node][pcp]['shaped_queues'][ingress_port] = ShapedQueue()

            # Initialize a shaped queue for local streams (per priority level)
            queues[node][pcp]['shaped_queues']['local'] = ShapedQueue()

        # Now, place the streams into their corresponding queues
        for stream in streams:
            if node in stream.path:  # If this node is part of the stream's path
                pcp = stream.pcp
                upstream_node = stream.source if stream.path[0] == node else None  # Determine the upstream node

                # QAR1: Different upstream nodes should have separate shaped queues even with the same priority
                if upstream_node:
                    if upstream_node not in queues[node][pcp]['shaped_queues']:
                        queues[node][pcp]['shaped_queues'][upstream_node] = ShapedQueue()
                    # Add the stream to the corresponding queue based on its upstream node
                    queues[node][pcp]['shaped_queues'][upstream_node].add_frame(stream)

                # QAR3: Streams originating from the current node
                if stream.source == node:
                    if 'local' not in queues[node][pcp]['shaped_queues']:
                        queues[node][pcp]['shaped_queues']['local'] = ShapedQueue()
                    # Add the stream to the local shaped queue
                    queues[node][pcp]['shaped_queues']['local'].add_frame(stream)

    return queues

def per_hop_delay(stream, streams_list, current_node, next_node, link_rate, queues):
    pcp = stream.pcp
    upstream_node = stream.source if stream.path[0] == current_node else None
    shaped_queue = None

    # QAR1: Use shaped queue from the corresponding upstream node
    if upstream_node in queues[current_node][pcp]['shaped_queues']:
        shaped_queue = queues[current_node][pcp]['shaped_queues'][upstream_node]
    # QAR3: If stream originates from the current node, use the local queue
    elif 'local' in queues[current_node][pcp]['shaped_queues']:
        shaped_queue = queues[current_node][pcp]['shaped_queues']['local']

    # Replenish tokens for all queues (assuming time has passed)
    shaped_queue.replenish_tokens(1)  # Assuming time passed is 1 microsecond

    # Burst sizes for higher, same, and lower priority streams
    bH = sum(s.size for s in streams_list if s.pcp > stream.pcp and current_node in s.path and next_node in s.path)
    bCj = sum(s.size for s in streams_list if s.pcp == stream.pcp and s != stream and current_node in s.path and next_node in s.path)
    bj = stream.size  # Current stream size
    lj = stream.size  # Transmission delay for stream 'j'
    
    # Compute the transmission delay for frame j
    lj_over_r = lj / link_rate

    # Rate consumed by higher priority streams
    rH = sum(s.size / s.period for s in streams_list if s.pcp > stream.pcp and current_node in s.path and next_node in s.path)

    # Calculate the used bandwidth by current streams in the queue
    used_bandwidth = sum(s.size / s.period for s in streams_list if current_node in s.path)

    # Remaining bandwidth considering currently used streams
    remaining_bandwidth = link_rate - used_bandwidth
    
    # Ensure remaining bandwidth isn't negative
    remaining_bandwidth = max(remaining_bandwidth, 0)
    
    # Maximum size of a lower priority frame in transmission
    lL = max((s.size for s in streams_list if s.pcp < stream.pcp and current_node in s.path and next_node in s.path), default=0)

    # Total potential interference (no need for lj since it's already included in bj)
    total_interference = bH + bCj + bj + lL  # Total potential interference

    # Per-hop delay calculation based on the total interference
    hop_delay = max(total_interference / remaining_bandwidth + lj_over_r, lj_over_r)
    
    return hop_delay

# Function to calculate worst-case delays
def calc_worst_case_delay(streams_list, link_rate, queues):
    for stream in streams_list:
        total_delay = 0
        for i in range(len(stream.path) - 1):
            current_node = stream.path[i]
            next_node = stream.path[i + 1]
            hop_delay = per_hop_delay(stream, streams_list, current_node, next_node, link_rate, queues)
            total_delay += hop_delay
        stream.worst_case_delay = total_delay

# Write the output (stream delay and path) to a CSV file
def write_output(file_path, streams):
    output_columns = ["Flow", "maxE2E (us)", "Deadline (us)", "Path"]  # Output column headers
    output_data = []
    
    # Prepare output data for each stream
    for stream in streams:
        path_str = '->'.join(f"{node}:{index}:0" for index, node in enumerate(stream.path))
        output_data.append([stream.name, stream.worst_case_delay, stream.deadline, path_str])
    
    output_df = pd.DataFrame(output_data, columns=output_columns)  # Create DataFrame
    output_df.to_csv(file_path, index=False)  # Save to CSV file

# Main function with example input handling
def main():
    link_rate = 100e6  # 100 Mbps

    # Read input streams and network topology
    streams_list = read_streams('./Test_Cases/Small_Case/small-streams.csv')
    topology_graph = read_topology('./Test_Cases/Small_Case/small-topology.csv')

    # Initialize shaped queues following QAR rules
    queues = initialize_queues(streams_list, topology_graph)

    # Find shortest paths for each stream
    find_shortest_path(streams_list, topology_graph)

    # Calculate the worst-case delay for each stream
    calc_worst_case_delay(streams_list, link_rate, queues)

    # Output the results
    write_output('output.csv', streams_list)

if __name__ == "__main__":
    main()