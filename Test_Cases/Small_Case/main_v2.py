import csv
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Stream:
    pcp: int
    name: str
    stream_type: str
    source: str
    destination: str
    size: int
    period: int
    deadline: int
    path: List[str] = None
    max_e2e: float = None

@dataclass
class Link:
    id: str
    source: str
    source_port: int
    destination: str
    destination_port: int
    domain: str

class ATSAnalysisTool:
    def __init__(self):
        self.streams: List[Stream] = []
        self.topology: nx.Graph = nx.Graph()
        self.links: Dict[str, Link] = {}

    def read_streams(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.streams.append(Stream(
                    int(row[0]), row[1], row[2], row[3], row[4],
                    int(row[5]), int(row[6]), int(row[7])
                ))

    def read_topology(self, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] in ['ES', 'SW']:
                    self.topology.add_node(row[1], type=row[0], ports=int(row[2]))
                elif row[0] == 'LINK':
                    link = Link(row[1], row[2], int(row[3]), row[4], int(row[5]), row[6])
                    self.links[link.id] = link
                    self.topology.add_edge(link.source, link.destination, link_id=link.id)

    def calculate_shortest_paths(self):
        for stream in self.streams:
            stream.path = nx.shortest_path(self.topology, stream.source, stream.destination)

    def calculate_worst_case_delay(self, stream: Stream) -> float:
      
        total_delay = 0
        for i in range(len(stream.path) - 1):
            current_node = stream.path[i]
            next_node = stream.path[i + 1]
            
            node_delay = self.calculate_node_delay(stream, current_node, next_node)
            total_delay += node_delay

        return total_delay

    def calculate_node_delay(self, stream: Stream, current_node: str, next_node: str) -> float:
       
        base_delay = stream.size / 1000  
        interfering_streams = self.get_interfering_streams(stream, current_node, next_node)
        interference_delay = sum(s.size / 1000 for s in interfering_streams)
        
        return base_delay + interference_delay

    def get_interfering_streams(self, stream: Stream, current_node: str, next_node: str) -> List[Stream]:
        interfering_streams = []
        for s in self.streams:
            if s != stream and current_node in s.path and next_node in s.path:
                if s.pcp >= stream.pcp:  
                    interfering_streams.append(s)
        return interfering_streams

    def analyze(self):
        self.calculate_shortest_paths()
        for stream in self.streams:
            stream.max_e2e = self.calculate_worst_case_delay(stream)

    def write_solution(self, filename: str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["StreamName", "MaxE2E(us)", "Deadline(us)", "Path"])
            for stream in self.streams:
                path_str = "->".join(stream.path)
                writer.writerow([stream.name, f"{stream.max_e2e:.3f}", stream.deadline, path_str])

def main():
    tool = ATSAnalysisTool()
    tool.read_streams('small-streams.csv')
    tool.read_topology('small-topology.csv')
    tool.analyze()
    tool.write_solution('solution-alt.csv')

if __name__ == "__main__":
    main()