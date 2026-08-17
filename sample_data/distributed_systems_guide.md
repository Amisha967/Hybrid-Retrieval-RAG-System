# Distributed Systems Engineering & Consensus Protocols

Distributed systems consist of autonomous computing nodes that communicate over network channels and coordinate actions by passing messages.

## Consensus Algorithms: Raft vs. Paxos

Consensus algorithms enable a collection of machines to work as a coherent group that can survive failures of some of its members.

### Raft Consensus
Raft achieves consensus by electing a distinguished leader, then giving the leader complete responsibility for managing the replicated log.
The algorithm decomposes consensus into three independent subproblems:
1. Leader Election: A new leader must be chosen when an existing leader fails. Raft uses randomized election timers to avoid split votes.
2. Log Replication: The leader accepts log entries from clients, replicates them across other servers via AppendEntries RPCs, and forces them to agree.
3. Safety: If any server has applied a particular log entry to its state machine, no other server may apply a different log entry for the same log index.

### Byzantine Fault Tolerance (BFT)
In adversarial networks where nodes may act maliciously or send conflicting messages, crash-fault-tolerant protocols like Raft are insufficient. PBFT (Practical Byzantine Fault Tolerance) requires \(3f + 1\) total replicas to tolerate \(f\) Byzantine nodes, maintaining safety across a three-phase commit process: Pre-Prepare, Prepare, and Commit.

## CAP Theorem & PACELC

Eric Brewer's CAP theorem states that a distributed data store can simultaneously provide at most two of the following three guarantees:
- Consistency (Linearizability): Every read receives the most recent write or an error.
- Availability: Every non-failing node returns a non-error response for every request.
- Partition Tolerance: The system continues to operate despite an arbitrary number of dropped or delayed messages between nodes.

Daniel Abadi extended this into the PACELC theorem: If there is a **P**artition, how does the system trade off **A**vailability and **C**onsistency; **E**lse, when the system is running normally without partitions, how does it trade off **L**atency and **C**onsistency?
