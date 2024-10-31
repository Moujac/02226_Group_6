import csv
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Stream:
    pcp: int
    name: str
    type: str
    source: str
    dest: str
    size: int
    period: int
    deadline: int
    path: List[str]
    priority: int
    frame_length: int
    reserved_rate: float
    burst_size: int

    @classmethod
    def from_row(cls, row: List[str]) -> 'Stream':
        pcp, name, stream_type, source, dest, size, period, deadline = row
        size_int = int(size)
        return cls(
            pcp=int(pcp),
            name=name,
            type=stream_type,
            source=source,
            dest=dest,
            size=size_int,
            period=int(period),
            deadline=int(deadline),
            path=[],
            priority=int(pcp),
            frame_length=size_int,
            reserved_rate=size_int / int(period),
            burst_size=size_int
        )

class NetworkTopology:
    def __init__(self, topology_file: Path):
        self.graph = nx.Graph()
        self.shaped_queue = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._read_topology(topology_file)

    def _read_topology(self, filename: Path) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                    
                record_type = row[0]
                if record_type == 'LINK':
                    self._process_link(row)
                elif record_type in ('SW', 'EN'):
                    self._process_node(row)

    def _process_link(self, row: List[str]) -> None:
        _, link_id, source, source_port, dest, dest_port = row
        self.graph.add_edge(
            source, 
            dest, 
            source_port=int(source_port),
            dest_port=int(dest_port)
        )

    def _process_node(self, row: List[str]) -> None:
        _, node, ports = row
        for port in range(1, int(ports) + 1):
            self.shaped_queue[node][port] = {priority: [] for priority in range(8)}

    def find_shortest_path(self, stream: Stream) -> List[str]:
        try:
            path = nx.shortest_path(self.graph, stream.source, stream.dest)
            self._update_shaped_queue(stream, path)
            return path
        except nx.NetworkXNoPath:
            return []

    def _update_shaped_queue(self, stream: Stream, path: List[str]) -> None:
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i + 1]]
            next_node = path[i + 1]
            dest_port = edge_data['dest_port']
            
            if next_node in self.shaped_queue:
                self.shaped_queue[next_node][dest_port][stream.priority].append(stream)

class ATS_Delay_Calculator:
    def __init__(self, link_rate: float):
        self.link_rate = link_rate

    def calculate_hop_delay(
        self,
        streams_same_queue: List[Stream],
        streams_high_pri: List[Stream],
        streams_same_pri: List[Stream],
        streams_low_pri: List[Stream]
    ) -> float:
        b_H = sum(s.burst_size for s in streams_high_pri)
        r_H = sum(s.reserved_rate for s in streams_high_pri)
        b_C = sum(s.burst_size for s in streams_same_pri) if streams_same_pri else 0
        l_L = max((s.frame_length for s in streams_low_pri), default=0)

        max_delay = 0
        for stream in streams_same_queue:
            stream_delay = self._calculate_stream_delay(
                stream, b_H, b_C, l_L, r_H
            )
            max_delay = max(max_delay, stream_delay)
        
        return max_delay * 1e6

    def _calculate_stream_delay(
        self,
        stream: Stream,
        b_H: int,
        b_C: int,
        l_L: int,
        r_H: float
    ) -> float:
        return (
            (b_H + b_C + stream.burst_size - stream.frame_length + l_L) /
            (self.link_rate - r_H) +
            stream.frame_length / self.link_rate
        )

    def calculate_total_delay(
        self,
        stream: Stream,
        topology: NetworkTopology
    ) -> float:
        total_delay = 0
        path = stream.path

        for i in range(len(path) - 2):
            node = path[i]
            next_node = path[i + 1]
            edge_data = topology.graph[node][next_node]
            dest_port = edge_data['dest_port']

            # Get streams in same queue
            streams_same_queue = topology.shaped_queue[next_node][dest_port][stream.priority]

            # Get higher priority streams across all ports
            streams_high_pri = []
            for port in topology.shaped_queue[next_node]:
                for priority in range(stream.priority + 1, 8):
                    streams_high_pri.extend(
                        topology.shaped_queue[next_node][port][priority]
                    )

            # Get same priority streams across all ports
            streams_same_pri = []
            for port in topology.shaped_queue[next_node]:
                streams_same_pri.extend(
                    topology.shaped_queue[next_node][port][stream.priority]
                )

            # Get lower priority streams across all ports
            streams_low_pri = []
            for port in topology.shaped_queue[next_node]:
                for priority in range(stream.priority):
                    streams_low_pri.extend(
                        topology.shaped_queue[next_node][port][priority]
                    )

            hop_delay = self.calculate_hop_delay(
                streams_same_queue,
                streams_high_pri,
                streams_same_pri,
                streams_low_pri
            )
            total_delay += hop_delay

        return total_delay

def read_streams(filename: Path) -> List[Stream]:
    streams = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [Stream.from_row(row) for row in reader]

def main():
    base_path = Path('../test_cases/test_case_random_geometric')
    topology = NetworkTopology(base_path / 'topology.csv')
    streams = read_streams(base_path / 'streams.csv')
    delay_calculator = ATS_Delay_Calculator(link_rate=1e8)

    with open(base_path/'solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StreamName', 'MaxE2E(us)', 'Deadline(us)', 'Path'])

        for stream in streams:
            path = topology.find_shortest_path(stream)
            stream.path = path
            
            if not path:
                continue
                
            max_e2e = delay_calculator.calculate_total_delay(stream, topology)

            writer.writerow([
                stream.name,
                f"{max_e2e:.1f}",
                stream.deadline,
                '->'.join(path)
            ])

if __name__ == "__main__":
    main()