from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SimConfig:
    folder_name: str
    add_file: str
    net_file: str
    rou_file: str
    route_ids: List[str] = field(init=False)
    routes_map: Dict[str, List[str]]
    description: str = ""

    def __post_init__(self):
        all_lists = self.routes_map.values()
        unique_routes = {r for route_list in all_lists for r in route_list}
        self.route_ids = sorted(list(unique_routes), key=lambda x: int(x.replace("route", "")))


CONFIG_4WAY_160M = SimConfig(
    folder_name="4way_crossing_160m",
    add_file="4way_crossing_160m.add.xml",
    net_file="4way_crossing_160m.net.xml",
    rou_file="4way_crossing_160m.rou.xml",

    routes_map={
        "NS_Straight": ["route11", "route6"], 
        "NS_Right":    ["route5", "route10"], 
        "NS_Left":     ["route4", "route12"],
        "EW_Straight": ["route2", "route7"],
        "EW_Right":    ["route1", "route9"],
        "EW_Left":     ["route3", "route8"]
    },

    description="Incrocio a 4 vie. Ogni braccio è lungo 160m ed ha due corsie per senso di marcia"
)