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

class ShapedQueue:
    def __init__(self, capacity):
        self.queue = []  # List to store incoming frames waiting to be shaped
        self.ready_queue = []  # List to store frames ready for transmission
        self.tokens = 0  # Initialize token bucket
        self.capacity = capacity  # Maximum tokens the bucket can hold

    def add_frame(self, frame):
        # Adds a frame to the queue. If enough tokens are available, the frame is moved to the ready queue.
        if self.tokens >= frame.size:
            self.ready_queue.append(frame)
            self.tokens -= frame.size
        else:
            self.queue.append(frame)

    def replenish_tokens(self, rate):
        # Replenish tokens in the bucket based on the data rate (rf).
        self.tokens += rate
        if self.tokens > self.capacity:  # Ensure we don't exceed capacity (ˆbf)
            self.tokens = self.capacity
        self.move_to_ready_queue()

    def move_to_ready_queue(self):
        # Move frames from the main queue to the ready queue if enough tokens are available.
        for frame in self.queue[:]:
            if self.tokens >= frame.size:
                self.ready_queue.append(frame)
                self.tokens -= frame.size
                self.queue.remove(frame)

    def get_next_frame(self):
        # Retrieve the next frame from the ready queue for transmission.
        return self.ready_queue.pop(0) if self.ready_queue else None

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
            size=int(row['Size']),
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

    for line in lines:
        parts = line.strip().split(',')
        if parts[0] == 'LINK':
            # If line represents a link, add an edge between nodes
            _, link_id, source, source_port, dest, dest_port, domain = parts
            G.add_edge(source, dest, weight=1)  # Add edge with default weight
        else:
            # Otherwise, line represents a device (node)
            device_type, device_name, ports, domain = parts
            G.add_node(device_name, type=device_type, ports=int(ports), domain=domain, queues={})  # Add node with queues attribute
    
    return G

# Are ports correct???
def init_queues(G, streams_list):
    # Initialize queues for each node based on the nodes and streams
    for node in G.nodes():
        priority_levels = set(range(8))  # Initialize all priority levels 0-7
        input_ports = set()  # Identify input ports based on topology links

        for src, dest in G.edges():
            if dest == node:
                input_ports.add(src)
            if src == node:
                input_ports.add(dest)

        for stream in streams_list:
            for i in range(len(stream.path)):  # Include all nodes, starting from 0
                if stream.path[i] == node:
                    if i > 0:
                        input_ports.add(stream.path[i - 1])  # Add the previous node as input port
                    else:
                        input_ports.add(stream.source)  # For the first node, add source as input port

        #print(f"Node: {node}, Priority Levels: {priority_levels}, Input Ports: {input_ports}")

        node_queues = {}
        for pcp in priority_levels:
            node_queues[pcp] = {'shaped_queues': {}, 'ready_queue': ShapedQueue(capacity=streams_list[0].size)}
            for input_port in input_ports:
                node_queues[pcp]['shaped_queues'][input_port] = ShapedQueue(capacity=streams_list[0].size)
                #print(f"Initializing shaped queue for Node: {node}, PCP: {pcp}, Input Port: {input_port}")

        G.nodes[node]['queues'] = node_queues  # Assign the initialized queues to the node

    # Add streams to appropriate queues and replenish tokens
    for stream in streams_list:
        for i in range(len(stream.path) - 1):
            current_node = stream.path[i]
            pcp = stream.pcp
            input_port = stream.path[i - 1] if i > 0 else stream.source

            if input_port in G.nodes[current_node]['queues'][pcp]['shaped_queues']:
                shaped_queue = G.nodes[current_node]['queues'][pcp]['shaped_queues'][input_port]
                shaped_queue.add_frame(stream)
                shaped_queue.replenish_tokens(stream.size / stream.period)  # Replenish tokens based on frame's rate
            else:
                print(f"Input port {input_port} not found in shaped queues for node {current_node}")

    return G

# Compute shortest path in the network for each stream using Dijkstra's algorithm
def find_shortest_path(streams, graph):
    for stream in streams:
        # Find shortest path from source to destination
        path = nx.shortest_path(graph, source=stream.source, target=stream.dest, weight='weight')
        stream.path = path  # Store the computed path

# Did I understand the math correctly ?? need to sanity check 
def per_hop_delay(stream, streams_list, current_node, next_node, link_rate, graph):
    pcp = stream.pcp
    input_port = None

    # Determine the input port by finding the previous node in the path
    if current_node != stream.path[0]:  # Check if it's not the source node
        for i in range(1, len(stream.path)):
            if stream.path[i] == current_node:
                input_port = stream.path[i - 1]
                break

    if input_port is None:
        # Handle the special case where current_node is the source node itself
        if current_node == stream.path[0]:
            input_port = current_node
        else:
            raise KeyError(f"Input port for node {current_node} not found in stream path {stream.path}")

    # Access the correct shaped queue
    shaped_queue = graph.nodes[current_node]['queues'][pcp]['shaped_queues'][input_port]

    # Access set I directly from the shaped queue
    I = shaped_queue.queue

    # Calculate delays iterating over set I
    delays = []
    for s in I:
        # Calculate burst sizes and transmission delays
        bH = sum(st.size for st in streams_list if st.pcp > s.pcp and current_node in st.path and next_node in st.path)
        bCj = sum(st.size for st in streams_list if st.pcp == s.pcp and st != s and current_node in st.path and next_node in st.path)
        bj = s.size
        lj = s.size
        lL = max((st.size for st in streams_list if st.pcp < s.pcp and current_node in st.path and next_node in st.path), default=0)
        rH = sum(st.size / st.period for st in streams_list if st.pcp > s.pcp and current_node in st.path and next_node in st.path)
        transDelay = lj / link_rate
        Denominator = link_rate - rH

        # Calculate per-hop delay
        delay = ((bH + bCj + bj - lj + lL) / Denominator) + transDelay

        delays.append(delay)

    # Calculate per-hop delay considering the maximum delay from the loop
    hop_delay = max(delays)

    return hop_delay

# Function to calculate worst-case delays
def calc_worst_case_delay(streams_list, link_rate, graph):
    for stream in streams_list:
        total_delay = 0
        for i in range(len(stream.path) - 1):
            current_node = stream.path[i]
            next_node = stream.path[i + 1]
            hop_delay = per_hop_delay(stream, streams_list, current_node, next_node, link_rate, graph)
            total_delay += hop_delay
        stream.worst_case_delay = total_delay

# Write the output (stream delay and path) to a CSV file
def write_output(file_path, streams):
    output_columns = ["Flow", "maxE2E (us)", "Deadline (us)", "Path"]  # Output column headers
    output_data = []
    
    # Prepare output data for each stream
    for stream in streams:
        path_str = '->'.join(f"{node}:{index}" for index, node in enumerate(stream.path))
        output_data.append([stream.name, stream.worst_case_delay, stream.deadline, path_str])
    
    output_df = pd.DataFrame(output_data, columns=output_columns)  # Create DataFrame
    output_df.to_csv(file_path, index=False)  # Save to CSV file

def main():
    link_rate = 100e6  # Define link rate (data rate), 100 Mbps in this case

    # Handle input
    streams_list = read_streams('./Test_Cases/Small_Case/small-streams.csv')
    topology_graph = read_topology('./Test_Cases/Small_Case/small-topology.csv')

    # Calculate shortest paths for streams
    find_shortest_path(streams_list, topology_graph)

    # Initialize queues based on shortest paths
    topology_graph = init_queues(topology_graph, streams_list)

    # Calculate worst-case delay
    calc_worst_case_delay(streams_list, link_rate, topology_graph)

    # Generate output file
    write_output('output.csv', streams_list)

    print("Still work in progress!!!!")

if __name__ == "__main__":
    main()