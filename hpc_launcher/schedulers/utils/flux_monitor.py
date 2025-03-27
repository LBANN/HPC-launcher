# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
#
# Code provuded by FLUX Team member Mark Grondona [grondo] in
# https://github.com/flux-framework/flux-core/discussions/6732

import time
from collections import namedtuple, deque

import flux
from flux.idset import IDset
from flux.hostlist import Hostlist

OfflineEvent = namedtuple("OfflineEvent", "timestamp, rank, name")


class NodeMonitor:
    def __init__(self, handle):
        self.handle = handle
        self.hostlist = Hostlist(handle.attr_get("hostlist"))
        self.backlog = deque()
        self.rpc = None
        self.last_online = IDset()

    def start(self):
        self.rpc = self.handle.rpc(
            "groups.get",
            {"name": "broker.online"},
            nodeid=0,
            flags=flux.constants.FLUX_RPC_STREAMING,
        )
        return self

    def poll(self, timeout=-1.0):
        if self.rpc is None:
            raise RuntimeError("poll() called before start()")

        while not self.backlog:
            resp = self.rpc.wait_for(timeout).get()
            self.__online_group_update(resp)
            self.rpc.reset()

        return self.__next_event()

    def __next_event(self):
        return self.backlog.popleft()

    def __online_group_update(self, resp):

        # All ranks leaving in this update share an event timestamp
        timestamp = time.time()

        # Calculate the ranks that left the online group by subtracting
        # the current set from the previous set. This returns only those
        # ranks that left:
        online = IDset(resp["members"])
        leave = self.last_online - online

        # Append a single event to the backlog for each offline rank
        for rank in leave:
            self.__append_event(timestamp, rank)

        # Update last_online
        self.last_online = online

    def __append_event(self, timestamp, rank):
        self.backlog.append(OfflineEvent(timestamp, rank, self.hostlist[rank]))

def main():
    handle = flux.Flux()
    nodemon = NodeMonitor(handle).start()

    while True:
        timestamp, rank, hostname = nodemon.poll()
        print(f"rank {rank} ({hostname}) lost at {timestamp}")

if __name__ == "__main__":
    main()
    
