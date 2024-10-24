import csv
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

Link_speed = 1e9  # Assume 1 Gbps link speed
Propagation_delay = 1e-6  # Assuming 1 microsecond propagation delay

@dataclass
class Stream:
    id: str
    pcp: int
    stream_type: str
    source: str
    destination: str
    size: int
    period: int
    deadline: int
    path: List[str] = field(default_factory=list)
    max_e2e: float = 0

@dataclass
class Link:
    id: str
    source: str
    source_port: int
    destination: str
    destination_port: int

@dataclass
class Queue:
    priority: int
    ingress_node: str  # Add source node for QAR1 compliance
    streams: List[Stream] = field(default_factory=list)

class ATSAnalysisTool:
    def __init__(self):
        self.streams: List[Stream] = []
        self.topology: nx.Graph = nx.Graph()
        self.links: Dict[str, Link] = {}
        self.queues: Dict[str, Dict[str, List[Queue]]] = {}  

    def read_streams(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.streams.append(Stream(
                    id=f"Flow_{row[1][5:]}",  
                    pcp=int(row[0]),
                    stream_type=row[2],
                    source=row[3],
                    destination=row[4],
                    size=int(row[5]),
                    period=int(row[6]),
                    deadline=int(row[7])
                ))

    def read_topology(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'SW' or row[0] == 'ES':
                    node_name = row[1]
                    self.topology.add_node(node_name, type=row[0], ports=int(row[2]))
                    self.queues[node_name] = {}
                elif row[0] == 'LINK':
                    link = Link(row[1], row[2], int(row[3]), row[4], int(row[5]))
                    self.links[link.id] = link
                    self.topology.add_edge(link.source, link.destination, link_id=link.id)
                    
                    if link.source_port not in self.queues[link.source]:
                        self.queues[link.source][str(link.source_port)] = []

    def calculate_shortest_paths(self):
        for stream in self.streams:
            try:
                stream.path = nx.shortest_path(self.topology, stream.source, stream.destination)
            except nx.NetworkXNoPath:
                print(f"No path found for stream {stream.id} from {stream.source} to {stream.destination}")
                stream.path = []  

    def assign_streams_to_queues(self):
        for stream in self.streams:
            if not stream.path:
                continue  
            for i in range(len(stream.path) - 1):
                current_node = stream.path[i]
                next_node = stream.path[i + 1]
                edge = self.topology[current_node][next_node]
                link_id = edge['link_id']
                output_port = str(self.links[link_id].source_port)
                self.assign_stream_to_queue(stream, current_node, output_port)

    def assign_stream_to_queue(self, stream: Stream, node: str, output_port: str):
        if node not in self.queues or output_port not in self.queues[node]:
            self.queues.setdefault(node, {})[output_port] = []
        
        queue_list = self.queues[node][output_port]
        # Ensure streams from different ingress nodes are placed in different queues (QAR1)
        appropriate_queue = next((q for q in queue_list if q.priority == stream.pcp and q.ingress_node == stream.source), None)
        
        if not appropriate_queue:
            # Create a new queue if none matches both the priority and ingress node
            appropriate_queue = Queue(priority=stream.pcp, ingress_node=stream.source)
            queue_list.append(appropriate_queue)
        appropriate_queue.streams.append(stream)

    def calculate_hop_delay(self, stream: Stream, node: str, output_port: str) -> float:
        transmission_delay = (stream.size * 8)/Link_speed  
        propagation_delay = Propagation_delay  
        
        if node in self.queues and output_port in self.queues[node]:
            queue_list = self.queues[node][output_port]
            
            # Worst-case delay calculation based on the formula components
            b_H = sum((s.size * 8) / Link_speed for q in queue_list for s in q.streams if s.pcp > stream.pcp)
            b_C = sum((s.size * 8) / Link_speed for q in queue_list for s in q.streams if s.pcp == stream.pcp and s != stream)
            l_L = max(((s.size * 8) / Link_speed for q in queue_list for s in q.streams if s.pcp < stream.pcp), default=0)
            r_H = 0  # Reserved bandwidth for higher priority streams (simplified)
            
            available_bandwidth = Link_speed - r_H
            if available_bandwidth <= 0:
                raise ValueError("Link capacity exceeded by reserved higher priority rate")
            
            hop_delay = (b_H + b_C + l_L) / available_bandwidth
            
        else:
            hop_delay = 0  # No queues mean no delay at end systems
        
        return transmission_delay + propagation_delay + hop_delay

    def calculate_worst_case_delay(self, stream: Stream) -> float:
        if not stream.path:
            return float('inf')  # Return infinity for streams with no valid path
        
        total_delay = 0
        for i in range(len(stream.path) - 1):
            current_node = stream.path[i]
            next_node = stream.path[i + 1]
            edge = self.topology[current_node][next_node]
            link_id = edge['link_id']
            output_port = str(self.links[link_id].source_port)
            hop_delay = self.calculate_hop_delay(stream, current_node, output_port)
            total_delay += hop_delay
        return total_delay * 1e6  # Convert to microseconds

    def analyze(self):
        self.calculate_shortest_paths()
        self.assign_streams_to_queues()
        for stream in self.streams:
            stream.max_e2e = self.calculate_worst_case_delay(stream)

    def write_solution(self, filename: str):
        with open(filename, 'w', newline='') as f:
            f.write("Flow, maxE2E (us), Deadline (us), Path SourceName:LinkID:QueueNumber->...\n")
            for stream in self.streams:
                if not stream.path:
                    path_str = "No valid path"
                else:
                    path_str = "->".join(f"{node}:{self.get_link_and_queue(stream, i)}"
                                        for i, node in enumerate(stream.path[:-1]))
                    path_str += f"->{stream.path[-1]}"
                f.write(f"{stream.id},{stream.max_e2e:.3f},{stream.deadline},{path_str}\n")
                
    def get_link_and_queue(self, stream: Stream, index: int) -> str:
        current_node = stream.path[index]
        next_node = stream.path[index + 1]
        edge = self.topology[current_node][next_node]
        link_id = edge['link_id']
        output_port = str(self.links[link_id].source_port)
        queue_list = self.queues[current_node][output_port]
        queue_number = next(i for i, q in enumerate(queue_list) if stream in q.streams)
        return f"{link_id}:{queue_number}"

    def print_total_max_delay(self):
        total_max_delay = sum(stream.max_e2e for stream in self.streams)
        print(f"Total Maximum Delay for all streams: {total_max_delay:.3f} microseconds, if the Link Speed is : {Link_speed} and the Propagation Delay is : {Propagation_delay}")

def main():
    tool = ATSAnalysisTool()
    tool.read_streams('small-streams.csv')
    tool.read_topology('small-topology.csv')
    tool.analyze()
    tool.print_total_max_delay()
    tool.write_solution('small-solution-test.csv')

if __name__ == "__main__":
    main()
